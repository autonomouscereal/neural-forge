#!/usr/bin/env python3
"""Render the matrix campaign ledger into something a human can actually read.

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` §3 makes `ledger.jsonl` the
append-only spine of the campaign and §5 rule 8 makes "the rendered matrix" the
terminal deliverable.  This is that renderer.  It reads only the ledger, never a
receipt, so the ledger is provably sufficient -- if a number is missing from
this report it is missing from the ledger, and the receipt schema is what needs
fixing.

The report has five surfaces, in decreasing altitude:

1. **The matrix.**  Component (rows) x technique (columns), one glyph per cell
   position showing how deep the row got: the deepest level that passed, or the
   shallowest that failed and how.  This is the single picture the campaign
   exists to produce.
2. **Level ladders.**  Per component x technique, the M16 -> M8 -> M4 -> X4
   ladder with each rung's status, because §5 rule 2 grades a technique by the
   level it reaches *and every shallower level*, and rule 2's `non-monotone`
   flag is only visible in the ladder.
3. **Failure classes.**  A histogram over `first_divergent_metric`.  §5 rule 5
   forbids narrative causes, so this histogram *is* the campaign's failure
   analysis.
4. **Band-versus-behavior agreement.**  §5 rule 4 makes "does the S8 <=0.4 band
   predict the behavioral outcome" a campaign output in its own right; it is
   tallied here across every sealed cell as a 2x2.
5. **Per-cell drill-down.**  Every sealed instance with its numbers and its
   receipt digest, newest first, grouped by cell id so re-runs sit together.

Usage::

    python experiments/render_matrix_ledger.py \
        --ledger reports/matrix-campaign-20260810/ledger.jsonl \
        --out    reports/matrix-campaign-20260810/LEDGER.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    CHAIN_LINKS,
    COMPONENTS,
    TECHNIQUES,
    FREE_RUN_MIN_TOKEN_MATCH,
    read_ledger,
)

LEVEL_ORDER = ("M16", "M8", "M4", "M2", "X4", "NA")

#: The marginal ladder §5 rule 2 walks, shallowest first.  ``M2`` joined it in
#: wave 2 as the family's floor and the break-even point for every technique
#: whose stored form is a basis plus a solved factor (`NOTES-MX-E` §4.1).
LEVEL_LADDER = ("M16", "M8", "M4", "M2", "X4")

#: One glyph per status.  Chosen so the matrix reads at a glance in a terminal
#: and in a Markdown viewer: a filled mark is a real behavioral pass, a hollow
#: one is anything short of it.
STATUS_GLYPH = {
    "pass": "PASS",
    "pending-review": "REVW",
    "fail": "FAIL",
    "infra": "INFR",
    "timeout": "TIME",
}

FAILURE_SHORT = {
    "infra": "infra",
    "solve-conditioning": "cond",
    "local-solve": "solve",
    "macrocell-band": "band",
    "composition": "comp",
    "free-run": "free",
    "behavior": "behv",
}


def _sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    level = row.get("level", "NA")
    return (
        str(row.get("component_id", "")),
        str(row.get("technique_id", "")),
        LEVEL_ORDER.index(level) if level in LEVEL_ORDER else 99,
        str(row.get("site", "")),
        str(row.get("sealed_at", "")),
    )


def _latest_by_cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Newest sealed instance per cell id.

    The ledger is append-only and a re-run gets a new instance id rather than an
    overwrite (§3), so "the state of a cell" is its newest instance and every
    older one stays visible in the drill-down.
    """

    latest: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: str(r.get("sealed_at", ""))):
        latest[str(row["cell_id"])] = dict(row)
    return latest


def _cell_text(value: Any, limit: int = 110) -> str:
    """Collapse a free-text note into something a Markdown table row survives.

    Receipt notes carry tracebacks and JSON tails verbatim -- that is the point
    of them -- but a newline or a pipe inside a table cell silently destroys the
    table, which is exactly the kind of quiet unreadability this renderer exists
    to prevent.
    """

    text = " ".join(str(value or "").split()).replace("|", "\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _quote(value: Any, limit: int = 1200) -> list[str]:
    """Render a note as a real Markdown blockquote, newlines intact."""

    text = str(value or "").strip()
    if not text:
        return []
    if len(text) > limit:
        dropped = len(text) - limit
        text = text[:limit] + f"\n... [{dropped} more characters in the receipt]"
    return [f"> {line}" if line.strip() else ">" for line in text.splitlines()]


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "nan"
    if number in (float("inf"), float("-inf")):
        return "inf"
    return f"{number:.{digits}f}"


def _sites(entries: Sequence[Mapping[str, Any]]) -> str:
    """The comma-joined site set of a group of ledger rows, in site order.

    ADDENDUM 5 rule 2-S: a level claim is never written without the sites it
    was sealed at, so every claim this renderer prints has to be able to say
    which sites it is made of.
    """

    return ",".join(sorted({str(entry.get("site", "?")) for entry in entries}))


def _passing_sites(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(e.get("site", "?")) for e in entries if e.get("status") == "pass"})


def _cell_summary(entries: Sequence[Mapping[str, Any]]) -> str:
    """The glyph for one (component, technique) matrix position.

    Reports how deep the row got **and where**: the deepest level with a
    passing site and the sites it passed at, or -- when nothing passed -- the
    shallowest attempt and the link it broke at, because §4.2's escalation
    policy keys entirely off *which* link failed.

    ADDENDUM 5 changed the first half of that sentence.  This used to require
    *every* attempted site at a level to pass before it would print
    ``PASS@<level>``, which is site-blind in both directions: it hid
    `C06.T1.X4`'s real L12 pass behind two shallower failures, and where it did
    print a pass it named no site at all.  Rule 2-S makes the site set part of
    the claim, so the glyph is ``PASS@<level>@<sites>``.
    """

    if not entries:
        return "."
    by_level: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_level[str(entry.get("level", "NA"))].append(entry)
    deepest_pass = None
    deepest_sites: list[str] = []
    for level in LEVEL_ORDER:
        rows = by_level.get(level, [])
        passing = _passing_sites(rows)
        if passing:
            deepest_pass = level
            deepest_sites = passing
    if deepest_pass is not None:
        return f"PASS@{deepest_pass}@{','.join(deepest_sites)}"
    for level in LEVEL_ORDER:
        rows = by_level.get(level, [])
        if not rows:
            continue
        statuses = {str(r.get("status")) for r in rows}
        if "fail" in statuses:
            links = [
                FAILURE_SHORT.get(str(r.get("first_divergent_metric")), "?")
                for r in rows
                if r.get("status") == "fail"
            ]
            return f"{level}:{Counter(links).most_common(1)[0][0]}"
        if "infra" in statuses:
            return f"{level}:infra"
        if "timeout" in statuses:
            return f"{level}:time"
        if "pending-review" in statuses:
            return f"{level}:revw"
    return "."


def render(rows: Sequence[Mapping[str, Any]], *, ledger_path: Path) -> str:
    latest = _latest_by_cell(rows)
    body = sorted(latest.values(), key=_sort_key)
    lines: list[str] = []
    add = lines.append

    sealed_at = max((str(r.get("sealed_at", "")) for r in body), default="never")
    statuses = Counter(str(r.get("status")) for r in body)
    add("# Stage-1 matrix campaign -- ledger")
    add("")
    add(f"Rendered {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} "
        f"from `{ledger_path}`.")
    add("")
    add(
        f"**{len(body)} cells sealed** ({len(rows)} ledger rows including re-runs); "
        f"newest seal {sealed_at}."
    )
    add("")
    add("| status | cells | meaning |")
    add("|---|---:|---|")
    meaning = {
        "pass": "free-run >= %.2f and decode smoke reviewed coherent (§5 rule 1)"
        % FREE_RUN_MIN_TOKEN_MATCH,
        "pending-review": "cleared every mechanical link; awaiting the frozen review",
        "fail": "crossed a chain threshold; see first_divergent_metric",
        "infra": "calibration refused the cell -- void, not evidence",
        "timeout": "exceeded the 20-minute hard cap and yielded (§2.3)",
    }
    for status in ("pass", "pending-review", "fail", "infra", "timeout"):
        if statuses.get(status):
            add(f"| `{status}` | {statuses[status]} | {meaning[status]} |")
    add("")

    # ------------------------------------------------------------------ 1
    add("## 1. The matrix")
    add("")
    add(
        "Rows are components, columns are techniques.  A position shows the "
        "deepest level with a passing site **and the sites it passed at** "
        "(ADDENDUM 5 rule 2-S), or the shallowest level attempted and the chain "
        "link it broke at.  `.` is untested."
    )
    add("")
    technique_ids = [t for t in TECHNIQUES if t != "copy_exact"]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in body:
        if row.get("cell_kind") == "control":
            continue
        grouped[(str(row["component_id"]), str(row["technique_id"]))].append(row)
    header = "| component | " + " | ".join(technique_ids) + " |"
    add(header)
    add("|---|" + "---|" * len(technique_ids))
    for component in COMPONENTS:
        cells = [
            _cell_summary(grouped.get((component, technique), []))
            for technique in technique_ids
        ]
        if all(cell == "." for cell in cells):
            continue
        label = f"{component} {COMPONENTS[component]['name']}"
        add(f"| {label} | " + " | ".join(cells) + " |")
    if not grouped:
        add("| _(no component cells sealed yet)_ | " + " | ".join("." for _ in technique_ids) + " |")
    add("")

    control = [r for r in body if r.get("cell_kind") == "control"]
    calibration = [r for r in body if r.get("cell_kind") == "calibration"]
    if control or calibration:
        add("### Calibration row")
        add("")
        add("| cell | status | free-run vs control | notes |")
        add("|---|---|---:|---|")
        for row in control + calibration:
            add(
                f"| `{row['cell_id']}` | {STATUS_GLYPH.get(str(row['status']), '?')} "
                f"| {_number(row.get('token_match_vs_control'))} "
                f"| {_cell_text(row.get('notes'))} |"
            )
        add("")

    # ------------------------------------------------------------------ 2
    add("## 2. Level ladders")
    add("")
    add(
        "ADDENDUM 5 rule 2-S: a technique+component works *at level L at sites "
        "S*, where S is exactly the set of sites with a sealed passing receipt "
        "at L **and at every shallower level attempted at that site**.  The "
        "claim is never printed without its site set; the `good/attempted` "
        "count survives as a breadth annotation, never as the licence for an "
        "unqualified claim.  (This supersedes §5 rule 2, whose site-blind "
        "reading `NOTES-MX-G` §4.6 showed to be ambiguous.)"
    )
    add("")
    add("| component | technique | M16 | M8 | M4 | M2 | X4 | verdict (site-scoped) |")
    add("|---|---|---|---|---|---|---|---|")
    for (component, technique), entries in sorted(grouped.items()):
        rungs: list[str] = []
        by_level = defaultdict(list)
        for entry in entries:
            by_level[str(entry.get("level"))].append(entry)
        # Rule 2-S walks each *site's* own ladder: a site holds a claim at level
        # L only when it passed at L and at every shallower level it attempted.
        # Aggregating first and asking about sites afterwards is exactly the
        # site-blind step the addendum retires.
        site_claim: dict[str, str | None] = {}
        for site in sorted({str(e.get("site", "?")) for e in entries}):
            deepest: str | None = None
            # ``NA`` is walked with the ladder because C14 has no level axis at
            # all (norms are diagonal) and its sealed PASS used to fall through
            # every branch below and print `awaiting review` over a reviewed,
            # passing cell.
            for level in (*LEVEL_LADDER, "NA"):
                rows_here = [
                    r
                    for r in by_level.get(level, [])
                    if str(r.get("site", "?")) == site
                ]
                if not rows_here:
                    continue
                if all(r.get("status") == "pass" for r in rows_here):
                    deepest = level
                else:
                    break
            site_claim[site] = deepest
        for level in LEVEL_LADDER:
            rows_at = by_level.get(level, [])
            if not rows_at:
                rungs.append(".")
                continue
            good = _passing_sites(rows_at)
            rungs.append(
                f"{len(good)}/{len(rows_at)}"
                + (f" {','.join(good)}" if good else "")
            )
        verdict = "untested"
        claimed = {s: lv for s, lv in site_claim.items() if lv is not None}
        if claimed:
            ladder = (*LEVEL_LADDER, "NA")
            deepest_level = max(claimed.values(), key=ladder.index)
            at = sorted(s for s, lv in claimed.items() if lv == deepest_level)
            shallower = sorted(
                f"{s}:{lv}" for s, lv in claimed.items() if lv != deepest_level
            )
            attempted = len({str(e.get("site", "?")) for e in entries})
            verdict = (
                f"works at NA (no level axis) at {','.join(at)} "
                f"({len(at)}/{attempted} sites)"
                if deepest_level == "NA"
                else (
                    f"works to {deepest_level} at {','.join(at)} "
                    f"({len(at)}/{attempted} sites)"
                )
            )
            if shallower:
                verdict += f"; shallower elsewhere: {'; '.join(shallower)}"
        elif entries:
            links = Counter(
                str(r.get("first_divergent_metric"))
                for r in entries
                if r.get("status") == "fail"
            )
            statuses = Counter(str(r.get("status")) for r in entries)
            if links:
                verdict = f"fails at {links.most_common(1)[0][0]}"
            elif statuses.get("infra") == len(entries):
                # An infra row is a void cell, not a negative result (§3 link 1);
                # calling it "fails" would put a technique in the closed-path
                # ledger on the strength of a broken harness run.
                verdict = "void (infra) -- no evidence"
            elif statuses.get("timeout"):
                verdict = "timed out"
            else:
                verdict = "awaiting review"
        add(
            f"| {component} | {technique} | " + " | ".join(rungs) + f" | {verdict} |"
        )
    add("")

    # ------------------------------------------------------------------ 3
    add("## 3. Failure classes")
    add("")
    add(
        "The chain is walked mechanically (§5 rule 5); this histogram is the "
        "campaign's failure analysis in full.  `free-run` is the F3 signature -- "
        "a cell that cleared every numeric link and still lost behavior -- and is "
        "the most information-dense class in the matrix."
    )
    add("")
    counted = Counter(
        str(r.get("first_divergent_metric"))
        for r in body
        if r.get("first_divergent_metric")
    )
    add("| first divergent metric | cells | share |")
    add("|---|---:|---:|")
    total_failed = sum(counted.values())
    for link in CHAIN_LINKS:
        if counted.get(link):
            share = counted[link] / total_failed if total_failed else 0.0
            add(f"| `{link}` | {counted[link]} | {share:.0%} |")
    if not counted:
        add("| _(no divergences sealed)_ | 0 | -- |")
    add("")

    # ------------------------------------------------------------------ 4
    add("## 4. Band-versus-behavior agreement")
    add("")
    add(
        "§5 rule 4: boundary metrics are diagnostic only, and whether the S8 "
        "`<= 0.4` macrocell band *predicts* the behavioral outcome is itself a "
        "campaign output.  Rows are the band's prediction, columns the behavior."
    )
    add("")
    table = Counter()
    for row in body:
        band = row.get("endpoint_band_le_0p4")
        match = row.get("token_match_vs_control")
        if band is None or match is None:
            continue
        behaved = float(match) >= FREE_RUN_MIN_TOKEN_MATCH
        table[(bool(band), behaved)] += 1
    add("| band | behaved | did not behave |")
    add("|---|---:|---:|")
    add(f"| in band (<= 0.4) | {table[(True, True)]} | {table[(True, False)]} |")
    add(f"| out of band | {table[(False, True)]} | {table[(False, False)]} |")
    agree = table[(True, True)] + table[(False, False)]
    scored = sum(table.values())
    add("")
    add(
        f"Agreement {agree}/{scored}"
        + (f" ({agree / scored:.0%})" if scored else " (no scored cells yet)")
        + "."
    )
    add("")

    # ------------------------------------------------------------------ 4b
    add("## 4b. Storage honesty")
    add("")
    add(
        "A level names a *rank*, and a rank is not a size.  A technique whose "
        "stored form is a `w x r` basis plus an `r x w` solved factor stores "
        "`r * (in + out)` against a dense `in * out`, so on a square component "
        "it does not contract at all until `r < w/2`.  Cells above 1.0 below "
        "cost more storage than the tensor they replaced."
    )
    add("")
    stored_rows = [
        row for row in _latest_by_cell(rows).values()
        if row.get("stored_parameter_ratio") is not None
    ]
    if stored_rows:
        add("| cell | technique | level | stored | dense parent | stored/dense |")
        add("|---|---|---|---:|---:|---:|")
        for row in sorted(stored_rows, key=_sort_key):
            add(
                f"| `{row['cell_id']}` | {row.get('technique_id')} "
                f"| {row.get('level')} | {row.get('stored_parameter_count')} "
                f"| {row.get('parent_parameter_count')} "
                f"| {_number(row.get('stored_parameter_ratio'), 4)} |"
            )
        contracting = sum(
            1 for row in stored_rows if float(row["stored_parameter_ratio"]) < 1.0
        )
        add("")
        add(
            f"{contracting} of {len(stored_rows)} sealed cells with a storage "
            "reading actually contract."
        )
    else:
        add("_(no sealed cell reports a stored-parameter count yet)_")
    add("")

    # ------------------------------------------------------------------ 5
    add("## 5. Per-cell drill-down")
    add("")
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[str(row["cell_id"])].append(row)
    for cell_id in sorted(by_cell, key=lambda c: _sort_key(by_cell[c][-1])):
        instances = sorted(
            by_cell[cell_id], key=lambda r: str(r.get("sealed_at", "")), reverse=True
        )
        newest = instances[0]
        add(f"### `{cell_id}` -- {STATUS_GLYPH.get(str(newest['status']), '?')}")
        add("")
        add(
            f"{newest.get('component')} / {newest.get('technique')} / "
            f"level {newest.get('level')} / site {newest.get('site')} / "
            f"wave {newest.get('campaign_wave')}"
        )
        add("")
        add(
            "| instance | arch | status | link | write cos | macrocell relL2 "
            "| vs control | free-run | lcp | stored/dense | cond | s | receipt |"
        )
        add("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for entry in instances:
            add(
                "| `{instance}` | {arch} | {status} | {link} | {cos} | {band} "
                "| {delta} | {free} | {lcp} | {stored} | {cond} | {secs} "
                "| `{sha}` |".format(
                    instance=str(entry.get("cell_instance_id", ""))[:8],
                    # A cell sealed before wave 2 has no architecture field; the
                    # dash is the honest reading, not an assumption about which
                    # GPU it ran on.
                    arch=str(entry.get("compute_architecture") or "--").split(" ")[0],
                    status=str(entry.get("status")),
                    link=str(entry.get("first_divergent_metric") or "--"),
                    cos=_number(entry.get("write_heldout_cos")),
                    band=_number(entry.get("post_operator_heldout_relL2")),
                    delta=_number(entry.get("vs_copy_exact_control_delta")),
                    free=_number(entry.get("token_match_vs_control"), 3),
                    lcp=_number(entry.get("mean_lcp"), 3),
                    stored=_number(entry.get("stored_parameter_ratio"), 3),
                    cond=_number(entry.get("condition_number"), 1),
                    secs=_number(entry.get("wall_time_seconds"), 1),
                    sha=str(entry.get("receipt_sha", ""))[:12],
                )
            )
        quoted = _quote(newest.get("notes"))
        if quoted:
            add("")
            for line in quoted:
                add(line)
        add("")
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    ledger = Path(args.ledger).expanduser().absolute()
    rows = read_ledger(ledger)
    report = render(rows, ledger_path=ledger)
    if args.out is None:
        sys.stdout.write(report)
        return 0
    destination = Path(args.out).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(destination)
    print(f"rendered {len(rows)} ledger rows -> {destination}")
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
