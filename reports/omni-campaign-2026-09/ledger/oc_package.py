#!/usr/bin/env python3
"""Build (and re-verify) the review package Cyril actually reads.

The LFM2 campaign's `package.sh` copied the artifacts a wave's claims are read
off into `human-review/` and then committed them, because a claim that lives
only on the server is a claim nobody can check.  This is the same package for
one checkpoint, plus the four things the omni campaign added: BOTH review
columns in the per-case table, the canonical three-box reading beside the
sealed grade, decoded media artifacts with their machine diagnostics, and a
manifest that digests every file so the package can be verified after it has
been copied into the repo mirror.

The one-page README is the top of the folder because §10's per-child markdown
is long by design -- a reader who only wants to know what this thing is, what
it cost and whether it held anything should not have to scroll a dossier.

Nothing here grades anything.  Sealed counts come from the per-case file the
reviewer wrote, canonical boxes from the canonical re-read, media scores from
Cyril; this only lays them beside each other with the verbatim text.

Usage::

    oc_package.py build --registry R --checkpoint-id ID --out DIR \\
        [--dossier D] [--dossier-b2 D] [--per-case P] [--per-case-b2 P] \\
        [--canonical-reread C] [--receipt R] [--verdict V] [--build-receipt B] \\
        [--clone-receipt C]... [--bench B] [--media FILE_OR_DIR]... [--extra F]...
    oc_package.py verify --package DIR
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import wave
from array import array
from pathlib import Path
from typing import Any, Mapping, Sequence

from oc_checkpoint import OcCheckpointError, history
from oc_ledger import read_rows
from oc_seal import (
    CAMPAIGN,
    OcSealError,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    human_bytes,
    now_utc,
    oc_root,
)

MANIFEST_SCHEMA = "neural-forge-omni-review-package-manifest-v1"
MANIFEST_NAME = "MANIFEST.json"

#: Canonical three-box readings (ACTUAL_MATRIX_2026-08-28's lens).
CANONICAL_BOXES = ("signal", "artifact-miss", "content-miss")


# --------------------------------------------------------------------------
# dossier rendering (the READ files)
# --------------------------------------------------------------------------


def collapsed(text: str) -> str:
    """Consecutive repeated words and phrases removed -- `mx45_read_dossier`'s
    measure, carried verbatim so a `<350 COLLAPSED` marker means in this
    campaign exactly what it meant in the last one."""

    words = text.split()
    out: list[str] = []
    for word in words:
        if out and out[-1] == word:
            continue
        out.append(word)
    joined = " ".join(out)
    for period in range(2, 13):
        pattern = re.compile(r"(?:\b(\w+(?:\s+\w+){%d})\b\s+)(?:\1\s+)+" % (period - 1))
        joined = pattern.sub(r"\1 ", joined)
    return joined


def render_read(dossier: Mapping[str, Any], which: str) -> str:
    """One dossier side, case by case, full verbatim."""

    records = dossier.get(f"{which}_responses") or []
    label = dossier.get("cell_id") or f"PROBE.{dossier.get('probe', 'unknown')}"
    seal_digest = dossier.get("dossier_sha256") or ""
    lines = [f"# {label}  {which}  dossier {seal_digest[:16]}"]
    shape = dossier.get("standard_shape_child")
    if shape:
        fields = [d.get("field") for d in shape.get("config_diff", [])]
        lines.append(f"# net_ratio {shape.get('net_ratio')}  config_diff {fields}")
    caps = 0
    for index, record in enumerate(records, start=1):
        text = record.get("response_verbatim", record.get("response", ""))
        squeezed = collapsed(text)
        if record.get("termination") == "censored-at-cap":
            caps += 1
        lines.append(
            f"\n--- {index:2d} {record.get('id', record.get('case_id'))} "
            f"[{record.get('termination')}] {len(text)} chars, "
            f"{len(squeezed)} collapsed"
            + ("  <350 COLLAPSED" if len(squeezed) < 350 else "")
        )
        lines.append(text)
    lines.append(f"\n# cap-hits {caps}/{len(records)}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# media diagnostics (machine columns only; the score is Cyril's)
# --------------------------------------------------------------------------


def media_diagnostics(path: Path) -> dict[str, Any]:
    """Non-empty, size, digest, and -- for WAV -- duration and RMS.

    Deliberately shallow: `ffprobe` is not on the workstation and a diagnostic
    that only runs in one place is a diagnostic that stops being run.  Video
    duration and A/V sync are recorded by the media runner into the dossier's
    `artifacts[]`; this fills in what the standard library can measure and says
    `null` for the rest rather than guessing.
    """

    stat = path.stat()
    out: dict[str, Any] = {
        "path": path.name,
        "bytes": stat.st_size,
        "sha256": file_sha256(path),
        "non_empty": stat.st_size > 0,
        "suffix": path.suffix.lower(),
        "duration_s": None,
        "sample_rate": None,
        "channels": None,
        "frames": None,
        "audio_rms": None,
    }
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                width = handle.getsampwidth()
                out["frames"] = frames
                out["sample_rate"] = rate
                out["channels"] = handle.getnchannels()
                out["duration_s"] = (frames / rate) if rate else None
                raw = handle.readframes(min(frames, rate * 30))
            if width == 2 and raw:
                samples = array("h")
                samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
                if samples:
                    mean_square = sum(float(s) * float(s) for s in samples) / len(samples)
                    out["audio_rms"] = (mean_square ** 0.5) / 32768.0
        except (wave.Error, OSError, EOFError) as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def manifest_entries(root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.name == MANIFEST_NAME:
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return entries


def write_manifest(root: Path, head: Mapping[str, Any]) -> dict[str, Any]:
    entries = manifest_entries(root)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "campaign": CAMPAIGN,
        "checkpoint_id": head["checkpoint_id"],
        "tag": head["checkpoint"].get("tag"),
        "record_instance_id": head["record_instance_id"],
        "record_sha": head["receipt_sha"],
        "generated_at": now_utc(),
        "file_count": len(entries),
        "bytes_total": sum(int(e["bytes"]) for e in entries),
        "files": entries,
        "package_sha256": canonical_sha256(
            sorted([e["path"], e["sha256"]] for e in entries)
        ),
    }
    path = root / MANIFEST_NAME
    path.write_bytes(canonical_json_bytes(manifest))
    reread = json.loads(path.read_text(encoding="utf-8"))
    if reread["package_sha256"] != manifest["package_sha256"]:
        raise OcSealError(f"{path} does not read back as written")
    return manifest


def read_manifest(root: str | Path) -> dict[str, Any]:
    """The package's own MANIFEST.json, or a refusal.

    `verify_manifest` answers "is this folder internally consistent"; it cannot
    answer "whose folder is this", because every digest in it is recomputed
    from the manifest's own list.  Anything that needs the second question --
    R4(v), which packages the evidence before deleting the weights -- reads the
    manifest here and compares `checkpoint_id` / `record_sha` / `package_sha256`
    against the checkpoint it is about to destroy.
    """

    path = Path(root).expanduser().absolute() / MANIFEST_NAME
    if not path.is_file():
        raise OcSealError(f"no {MANIFEST_NAME} in {path.parent}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise OcSealError(f"{path} is not a {MANIFEST_SCHEMA}")
    return manifest


def verify_manifest(root: str | Path) -> list[str]:
    """Every problem with a package, as sentences.  Empty list = it verifies.

    NOTE what this does NOT establish: the digests are recomputed from the
    manifest's own file list, so a well-formed package for a DIFFERENT
    checkpoint verifies here too.  Use `read_manifest` and compare the identity
    fields when the answer has to be about a particular checkpoint.
    """

    base = Path(root).expanduser().absolute()
    path = base / MANIFEST_NAME
    if not path.is_file():
        return [f"no {MANIFEST_NAME} in {base}"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path} is not JSON: {exc}"]
    if manifest.get("schema") != MANIFEST_SCHEMA:
        return [f"{path} schema is {manifest.get('schema')!r}"]
    problems: list[str] = []
    declared = {e["path"]: e for e in manifest.get("files", [])}
    on_disk = {e["path"]: e for e in manifest_entries(base)}
    for name, entry in declared.items():
        actual = on_disk.get(name)
        if actual is None:
            problems.append(f"missing: {name}")
        elif actual["sha256"] != entry["sha256"]:
            problems.append(
                f"digest mismatch: {name} {actual['sha256'][:16]} != "
                f"{entry['sha256'][:16]}"
            )
        elif actual["bytes"] != entry["bytes"]:
            problems.append(f"size mismatch: {name}")
    for name in on_disk:
        if name not in declared:
            problems.append(f"undeclared: {name}")
    recomputed = canonical_sha256(
        sorted([e["path"], e["sha256"]] for e in manifest.get("files", []))
    )
    if recomputed != manifest.get("package_sha256"):
        problems.append(
            f"package_sha256 recomputes {recomputed[:16]}, manifest claims "
            f"{str(manifest.get('package_sha256'))[:16]}"
        )
    return problems


# --------------------------------------------------------------------------
# the markdown
# --------------------------------------------------------------------------


def _load(path: Path | None) -> Any:
    """Read a JSON input, or REFUSE by name.

    A mistyped `--dossier` / `--bench` / `--per-case` used to raise
    FileNotFoundError out of here, which `main` does not catch: the CLI's
    contract is `REFUSED: <why>` and exit 3, and a traceback is neither.
    """

    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        raise OcCheckpointError(
            f"{target} does not exist (or is not a file); every input named on "
            "the command line is read and copied into the package verbatim"
        )
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OcCheckpointError(f"{target} is not JSON: {exc}") from exc


def _canonical_lanes(
    payload: Any, keys: Sequence[str | None]
) -> tuple[dict[str, str], str | None]:
    """Accept `{lane: box}` or `{cell_id: {lane: box, intel: ...}}`.

    Returns `(lanes, source_key)`.  `source_key` is the key the block was found
    under, `"<flat>"` for the single-cell flat form, and None when the file
    names no block for THIS checkpoint.

    It used to fall back to `next(iter(payload))` -- the first dict-valued key
    in the file -- when the id lookup missed.  The wave-level re-read files are
    keyed by CELL id (`ARCH.C2S4L03`) and a checkpoint id is
    `OC01-F-Q35-4B-C2S4L03`, so the id lookup missed EVERY time and the
    fallback always fired: with one cell in the file the arbitrary block
    happened to be the right one, and with two the wrong cell's three-box
    readings were printed as this checkpoint's in the document Cyril decides
    on.  There is no defensible guess here.  If none of the candidate keys
    matches, the canonical column is UNREAD and REVIEW_<TAG>.md says so.
    """

    if not isinstance(payload, dict):
        return {}, None
    if any(v in CANONICAL_BOXES for v in payload.values() if isinstance(v, str)):
        # the flat single-cell form: the file IS this checkpoint's block, and
        # there is no second cell for it to be confused with.
        return {k: v for k, v in payload.items() if isinstance(v, str)}, "<flat>"
    for key in keys:
        if not key:
            continue
        block = payload.get(key)
        if isinstance(block, dict):
            return {k: v for k, v in block.items() if v in CANONICAL_BOXES}, key
    return {}, None


def _count_source(source: Any) -> str:
    """How a parameter count got its value, in words a reader can act on."""

    if source == "derived-from-safetensors-headers":
        return "read from the safetensors headers"
    if source == "asserted-by-operator":
        return "**asserted by the operator, not measured**"
    return "source not recorded"


def _ratio_basis(checkpoint: Mapping[str, Any]) -> str:
    """Whether the compression number in front of Cyril was measured or typed.

    The ratio is the claim the whole package is about.  A reader is entitled to
    know, on the same line as the number, whether both of its terms came off the
    disk.
    """

    sources = (
        checkpoint.get("parameter_count_source"),
        checkpoint.get("parent_parameter_count_source"),
    )
    if all(x == "derived-from-safetensors-headers" for x in sources):
        return "both counts read from the safetensors headers"
    if any(x == "asserted-by-operator" for x in sources):
        return "**one or both counts were ASSERTED, not measured**"
    return "count provenance not recorded (registered before it was tracked)"


def readme_text(
    head: Mapping[str, Any], files: Sequence[str], ledger_rows: int
) -> str:
    checkpoint = head["checkpoint"]
    review = head.get("review") or {}
    bank1 = review.get("bank1") or {}
    columns = review.get("columns") or {}
    bank2 = review.get("bank2") or {}
    media = review.get("media_out") or []
    ratio = checkpoint.get("net_ratio")
    lines = [
        f"# {head['checkpoint_id']} — review package",
        "",
        f"Campaign `{CAMPAIGN}`. Stage **{checkpoint['stage']}**, parent "
        f"`{checkpoint.get('parent_code')}`, wave `{checkpoint.get('wave')}`.",
        f"Status **{head['status']}** — {head['status_reason']}",
        "",
        "**The verdict and every count in this folder are ANNOTATIONS. The "
        "decision is yours.**",
        "",
        "| what | value |",
        "|---|---|",
        f"| parameters | {checkpoint.get('parameter_count'):,} "
        f"({_count_source(checkpoint.get('parameter_count_source'))}) |"
        if checkpoint.get("parameter_count") is not None else "| parameters | - |",
        f"| parent parameters | {checkpoint.get('parent_parameter_count'):,} "
        f"({_count_source(checkpoint.get('parent_parameter_count_source'))}) |"
        if checkpoint.get("parent_parameter_count") is not None
        else "| parent parameters | - |",
        f"| net stored ratio | {ratio:.10f} ({1 / ratio:.4f}x), "
        f"{_ratio_basis(checkpoint)} |"
        if ratio else "| net stored ratio | - |",
        f"| quantization | {checkpoint.get('quantization')} |",
        f"| directory digest | `{str(checkpoint.get('directory_sha256'))[:32]}…` |",
        f"| recipe digest | `{str((checkpoint.get('recipe') or {}).get('recipe_sha256'))[:32]}…` |",
        f"| sealed 15-case | {bank1.get('useful_or_partial', '-')} held "
        f"({bank1.get('useful', '-')}U/{bank1.get('partial', '-')}P/"
        f"{bank1.get('observed_miss', '-')}M) vs parent "
        f"{bank1.get('parent_useful_or_partial', '-')} |",
        f"| machine verdict | {bank1.get('sealed_verdict', '-')} |",
        f"| canonical (three-box) | {bank1.get('canonical_signal', '-')} signal / "
        f"{bank1.get('canonical_artifact_miss', '-')} artifact / "
        f"{bank1.get('canonical_content_miss', '-')} content |",
        f"| column A (copy-backs count) | {columns.get('column_a', '-')} |",
        f"| column B (copy-backs refused) | {columns.get('column_b', '-')} |",
        f"| bank v2 | sealed {bank2.get('sealed_useful_or_partial', '-')} / "
        f"canonical {bank2.get('canonical_signal', '-')} |",
        f"| media artifacts | {len(media)} ({sum(1 for m in media if m.get('cyril_score') is not None)} scored) |",
        f"| ledger rows | {ledger_rows} |",
        "",
        "## What to read, in order",
        "",
        f"1. `REVIEW_{checkpoint.get('tag') or head['checkpoint_id']}.md` — the full "
        "form: what was folded or trained, benchmarks, both review surfaces "
        "verbatim, media, export.",
        "2. `READ-*.txt` — the raw text, child and parent, every case, nothing "
        "summarised.",
        "3. `record-*.json` / `ledger-rows-*.jsonl` — the sealed record chain and "
        "the ledger rows that index it.",
        "",
        "## Files",
        "",
    ]
    lines += [f"- `{name}`" for name in files]
    lines += [
        "",
        "## Scoring this yourself",
        "",
        "Write `cyril-score-<TAG>.json` beside this README:",
        "",
        "```json",
        '{"tag": "<TAG>", "date": "<yyyy-mm-dd>", "per_case_overrides": {},',
        ' "media_scores": {"<prompt id>": 3}, "ruling_text": "<your words>"}',
        "```",
        "",
        "then seal it as a new ledger instance with `oc_checkpoint.py "
        "record-review ... --status <yours> --cyril-ruling \"<your words>\"`. "
        "Your read governs; the machine columns stay beside it as diagnostics.",
        "",
    ]
    return "\n".join(lines)


def review_markdown(
    head: Mapping[str, Any],
    *,
    dossier: Mapping[str, Any] | None,
    dossier_b2: Mapping[str, Any] | None,
    per_case: Mapping[str, Any] | None,
    per_case_b2: Mapping[str, Any] | None,
    canonical: Mapping[str, str],
    canonical_source: str | None = None,
    canonical_supplied: bool = False,
    bench: Any,
    media: Sequence[Mapping[str, Any]],
    export: Mapping[str, Any] | None,
    chain: Sequence[Mapping[str, Any]],
) -> str:
    checkpoint = head["checkpoint"]
    review = head.get("review") or {}
    bank1 = review.get("bank1") or {}
    columns = review.get("columns") or {}
    tag = checkpoint.get("tag") or head["checkpoint_id"]
    ratio = checkpoint.get("net_ratio")
    out: list[str] = []
    add = out.append

    add(f"# REVIEW {tag} — {head['checkpoint_id']}")
    add("")
    add(
        f"net stored ratio **{ratio:.10f}**"
        if ratio else "net stored ratio **-**"
    )
    add(
        f" · stage **{checkpoint['stage']}** · machine verdict "
        f"**{bank1.get('sealed_verdict', '-')}** · sealed "
        f"**{bank1.get('useful_or_partial', '-')}/15** vs parent "
        f"**{bank1.get('parent_useful_or_partial', '-')}/15** · canonical "
        f"**{bank1.get('canonical_signal', '-')}/15** · columns "
        f"**A={columns.get('column_a', '-')} B={columns.get('column_b', '-')}**"
    )
    add("")
    add("*The verdict and the counts are ANNOTATIONS — the decision is Cyril's.*")
    add("")
    add("## 1. What was folded or trained")
    add("")
    add("| component | parent params | child params | ratio | >=4x | counts |")
    add("|---|---:|---:|---:|---|---|")
    for component in checkpoint.get("per_component_ratios") or []:
        add(
            f"| {component['component']} | {component['parent_params']:,} | "
            f"{component['child_params']:,} | {component['ratio']:.6f} | "
            f"{'yes' if component['ratio'] <= 0.25 else 'NO'} | "
            f"{'derived from the safetensors headers' if component.get('tensor_prefix') else '**asserted by the operator**'} |"
        )
    if checkpoint.get("passed_through"):
        add("")
        add(f"Passed through unchanged: {', '.join(checkpoint['passed_through'])}.")
    accounting = checkpoint.get("fold_accounting")
    if accounting:
        add("")
        add(
            f"Accounting: the model lost **{accounting['measured_drop']:,}** "
            f"parameters (read from the headers) and the components above claim "
            f"**{accounting['claimed_saving']:,}** of that; "
            f"{accounting['unexplained']:,} is unexplained by the named components"
            + (f", declared slack {accounting['accounting_slack']:,}."
               if accounting.get("accounting_slack") else ".")
        )
    add("")
    add(f"Recipe: `{(checkpoint.get('recipe') or {}).get('command')}` "
        f"(sha256 `{(checkpoint.get('recipe') or {}).get('recipe_sha256')}`).")
    add("")
    add("Lineage (root first):")
    add("")
    for step in checkpoint.get("lineage") or []:
        add(f"- `{step.get('checkpoint_id')}` dir "
            f"`{str(step.get('directory_sha256'))[:16]}…` record "
            f"`{str(step.get('record_sha'))[:16]}…`")
    add("")
    add("Record chain:")
    add("")
    for record in chain:
        add(f"- {record['record_seq']:>2} `{record['event']}` "
            f"{record['sealed_at']} → {record['status']} "
            f"(`{record['receipt_sha'][:16]}…`)")
    add("")

    add("## 2. Benchmarks vs parent")
    add("")
    if bench:
        add("```json")
        add(json.dumps(bench, indent=1)[:20000])
        add("```")
        add("")
        add("*PARTIAL: sampled, `--limit 200 --seed 1234`; the parent row is re-run "
            "every wave and must reproduce to every digit.*")
    else:
        add("Not measured for this checkpoint.")
    add("")

    add("## 3. Frozen review (bank v1), in full")
    add("")
    # The canonical column is a SECOND reading of the same responses, and it
    # only means anything if it is this checkpoint's reading.  Say plainly
    # which block it came from -- or that there is none -- rather than letting
    # a dash in the column read as "no signal on that lane".
    if canonical_supplied and canonical_source is None:
        add("> **The canonical re-read file names no block for this checkpoint** "
            f"(looked for `{head['checkpoint_id']}`, `{tag}` and the dossier's "
            "`cell_id`). The *canonical box* column below is UNREAD — every "
            "entry is `-` because nothing was matched, not because nothing "
            "was found.")
        add("")
    elif canonical_source and canonical_source != "<flat>":
        add(f"*Canonical boxes read from block `{canonical_source}`.*")
        add("")
    if dossier:
        add("| # | lane | sealed | canonical box | termination | chars |")
        add("|---:|---|---|---|---|---:|")
        for index, record in enumerate(dossier.get("child_responses") or [], start=1):
            lane = record.get("id")
            text = record.get("response_verbatim", "")
            add(
                f"| {index} | {lane} | {(per_case or {}).get(lane, '-')} | "
                f"{canonical.get(lane, '-')} | {record.get('termination')} | "
                f"{len(text)} |"
            )
        add("")
        parents = {r.get("id"): r for r in dossier.get("parent_responses") or []}
        for record in dossier.get("child_responses") or []:
            lane = record.get("id")
            add(f"### {lane}")
            add("")
            add("**Prompt**")
            add("")
            for message in record.get("prompt_messages") or []:
                add(f"> *{message.get('role')}*: {message.get('content')}")
            add("")
            add(f"**Child** [{record.get('termination')}, "
                f"{record.get('generated_tokens')} tokens, "
                f"sha `{str(record.get('response_sha256'))[:16]}…`]")
            add("")
            add("```text")
            add(record.get("response_verbatim", ""))
            add("```")
            add("")
            parent = parents.get(lane)
            if parent:
                add(f"**Parent** [{parent.get('termination')}, "
                    f"{parent.get('generated_tokens')} tokens]")
                add("")
                add("```text")
                add(parent.get("response_verbatim", ""))
                add("```")
                add("")
    else:
        add("No dossier supplied to the packager.")
    add("")

    add("## 4. Bank v2, in full")
    add("")
    if dossier_b2:
        for record in dossier_b2.get("child_responses") or []:
            lane = record.get("id")
            add(f"### {lane}  ({(per_case_b2 or {}).get(lane, '-')})")
            add("")
            add("```text")
            add(record.get("response_verbatim", ""))
            add("```")
            add("")
    else:
        add("Bank v2 was not run on this checkpoint.")
    add("")

    add("## 5. Media artifacts")
    add("")
    if media:
        add("| file | bytes | duration | audio RMS | sha256 | Cyril 1-5 |")
        add("|---|---:|---:|---:|---|---|")
        # A file is joined to its review entry by prompt_id, and ALSO by any
        # filename the entry carries: joining on `Path(name).stem == prompt_id`
        # alone rendered a silent "-" for every artifact whose file name is not
        # its prompt id, which is most of them once a runner adds a seed or a
        # take number.  What still does not match is named under the table
        # rather than dashed out of sight.
        scores: dict[str, Any] = {}
        for entry in review.get("media_out") or []:
            for key in (
                entry.get("prompt_id"),
                entry.get("filename"),
                entry.get("file"),
                entry.get("path"),
            ):
                if key:
                    scores[str(key)] = entry.get("cyril_score")
                    scores[Path(str(key)).name] = entry.get("cyril_score")
                    scores[Path(str(key)).stem] = entry.get("cyril_score")
        unmatched: list[str] = []
        for item in media:
            name = str(item["path"])
            if name in scores:
                score = scores[name]
            elif Path(name).stem in scores:
                score = scores[Path(name).stem]
            else:
                score, _ = None, unmatched.append(name)
            add(
                f"| `{item['path']}` | {item['bytes']:,} | "
                f"{item['duration_s'] if item['duration_s'] is not None else '-'} | "
                f"{round(item['audio_rms'], 5) if item['audio_rms'] is not None else '-'} | "
                f"`{item['sha256'][:16]}…` | "
                f"{score if score is not None else '—'} |"
            )
        add("")
        if unmatched:
            add(f"**No `media_out` entry matched {len(unmatched)} of these files** "
                f"({', '.join('`' + n + '`' for n in unmatched)}): their score "
                "column is blank because nothing in the review names them, not "
                "because they were scored 0. The review's prompt ids are "
                + ", ".join(
                    f"`{m.get('prompt_id')}`" for m in (review.get("media_out") or [])
                ) + ".")
            add("")
        add("Machine columns are diagnostics only (non-empty, duration, RMS, "
            "digest). The score is Cyril's blind 1-5 against the parent artifact "
            "on the same prompt.")
    else:
        add("No decoded media artifacts in this package.")
    add("")

    add("## 6. Export")
    add("")
    if export:
        add("```json")
        add(json.dumps(export, indent=1))
        add("```")
    else:
        add(f"HF checkpoint `{checkpoint.get('directory')}` "
            f"(directory sha256 `{checkpoint.get('directory_sha256')}`). "
            "No GGUF export recorded for this checkpoint yet.")
    add("")

    add("## 7. Reviewer notes, and what is not known")
    add("")
    add(head.get("notes") or review.get("notes") or "(none recorded)")
    add("")
    add("Not known from this package: anything not in the files it lists. In "
        "particular the counts above are one reader's; the verbatim text in "
        "§3/§4 is what a second reader should grade before any of them is "
        "believed.")
    add("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    registry = Path(args.registry)
    chain = history(registry, args.checkpoint_id)
    if not chain:
        raise OcCheckpointError(f"{args.checkpoint_id} is not in the registry")
    head = chain[-1]
    checkpoint = head["checkpoint"]
    tag = checkpoint.get("tag") or args.checkpoint_id
    out = Path(args.out).expanduser().absolute()
    # Building into a folder that already has files used to overwrite the
    # README, the REVIEW and the MANIFEST in place while every stale file from
    # the previous build silently joined the new manifest -- and changed
    # `package_sha256`, which is the number a review commits to.  A non-empty
    # --out is now a refusal, and --overwrite MOVES the old package aside
    # rather than writing over it.
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise OcSealError(
                f"{out} is not empty.  A package is written once; re-running into "
                "a used folder mixes two builds into one manifest.  Pass "
                "--overwrite to move the existing one aside first."
            )
        backup = out.with_name(f"{out.name}.superseded-{now_utc().replace(':', '')}")
        shutil.move(str(out), str(backup))
        print(f"backed up the previous package to {backup}")
    out.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []

    def place(source: Path | None, name: str) -> Path | None:
        """Copy an input into the package, or REFUSE by name.

        `shutil.copy2` on a path that does not exist raises FileNotFoundError,
        which `main` does not catch: a mistyped `--dossier` came out of the
        build as a traceback instead of `REFUSED: <why>` and exit 3, and it did
        so AFTER some of the package had already been written.
        """

        if source is None:
            return None
        if not source.is_file():
            raise OcCheckpointError(
                f"{name}: {source} does not exist (or is not a file).  Every input "
                "named on the command line is copied into the package verbatim, "
                "so a path that is not there is a refusal, not an omission."
            )
        target = out / name
        shutil.copy2(source, target)
        copied.append(name)
        return target

    # the record chain and its ledger rows travel with the package: a package
    # whose provenance is only on the server is exactly the defect the LFM2
    # campaign fixed by committing every wave's human-review folder.
    (out / f"record-{tag}.json").write_bytes(canonical_json_bytes(head))
    copied.append(f"record-{tag}.json")
    (out / f"record-chain-{tag}.json").write_bytes(canonical_json_bytes(chain))
    copied.append(f"record-chain-{tag}.json")

    ledger_rows = 0
    if args.ledger and Path(args.ledger).is_file():
        rows = [
            row for row in read_rows(args.ledger)
            if row.get("checkpoint_id") == args.checkpoint_id
        ]
        ledger_rows = len(rows)
        (out / f"ledger-rows-{tag}.jsonl").write_bytes(
            b"".join(canonical_json_bytes(r) + b"\n" for r in rows)
        )
        copied.append(f"ledger-rows-{tag}.jsonl")

    dossier = _load(args.dossier)
    dossier_b2 = _load(args.dossier_b2)
    per_case = _load(args.per_case)
    per_case_b2 = _load(args.per_case_b2)
    # A wave-level re-read file is keyed by CELL id, not by checkpoint id, so
    # the dossier's own `cell_id` is the key that actually matches in practice.
    # All three candidates are tried; none matching is reported, never guessed.
    canonical_payload = _load(args.canonical_reread)
    canonical, canonical_source = _canonical_lanes(
        canonical_payload,
        [args.checkpoint_id, tag, (dossier or {}).get("cell_id")],
    )
    if canonical_payload is not None and canonical_source is None:
        print(
            "WARNING: the canonical re-read file names no block for "
            f"{args.checkpoint_id} (tried the checkpoint id, tag {tag!r} and "
            f"cell_id {(dossier or {}).get('cell_id')!r}); the canonical "
            "column in REVIEW_" + str(tag) + ".md is marked UNREAD."
        )
    bench = _load(args.bench)
    export = _load(args.export)

    place(args.dossier, f"dossier-{tag}.json")
    place(args.dossier_b2, f"dossier-b2-{tag}.json")
    place(args.per_case, f"per-case-{tag}.json")
    place(args.per_case_b2, f"per-case-b2-{tag}.json")
    place(args.canonical_reread, f"canonical-reread-{tag}.json")
    place(args.receipt, f"receipt-{tag}.json")
    place(args.verdict, f"verdict-{tag}.json")
    place(args.build_receipt, f"build-receipt-{tag}.json")
    place(args.bench, f"bench-{tag}.json")
    place(args.export, f"export-{tag}.json")
    for index, path in enumerate(args.clone_receipt or [], start=1):
        place(Path(path), f"clone-receipt-{index}-{Path(path).name}")
    recipe = (checkpoint.get("recipe") or {}).get("recipe_path")
    if recipe and Path(recipe).is_file():
        place(Path(recipe), f"recipe-{tag}{Path(recipe).suffix}")
    for path in args.extra or []:
        place(Path(path), Path(path).name)

    if dossier:
        (out / f"READ-{tag}.txt").write_text(render_read(dossier, "child"), encoding="utf-8")
        copied.append(f"READ-{tag}.txt")
        if dossier.get("parent_responses"):
            (out / f"READ-PARENT-{tag}.txt").write_text(
                render_read(dossier, "parent"), encoding="utf-8"
            )
            copied.append(f"READ-PARENT-{tag}.txt")
    if dossier_b2:
        (out / f"READ-B2-{tag}.txt").write_text(
            render_read(dossier_b2, "child"), encoding="utf-8"
        )
        copied.append(f"READ-B2-{tag}.txt")

    media: list[dict[str, Any]] = []
    if args.media:
        media_dir = out / "media"
        media_dir.mkdir(exist_ok=True)
        sources: list[Path] = []
        for item in args.media:
            path = Path(item)
            sources.extend(sorted(p for p in path.rglob("*") if p.is_file())
                           if path.is_dir() else [path])
        for source in sources:
            target = media_dir / source.name
            shutil.copy2(source, target)
            media.append(media_diagnostics(target))
            copied.append(f"media/{source.name}")
        (media_dir / "media-diagnostics.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema": "neural-forge-omni-media-diagnostics-v1",
                    "checkpoint_id": args.checkpoint_id,
                    "artifacts": media,
                }
            )
        )
        copied.append("media/media-diagnostics.json")

    review_md = review_markdown(
        head,
        dossier=dossier,
        dossier_b2=dossier_b2,
        per_case=per_case,
        per_case_b2=per_case_b2,
        canonical=canonical,
        canonical_source=canonical_source,
        canonical_supplied=canonical_payload is not None,
        bench=bench,
        media=media,
        export=export,
        chain=chain,
    )
    (out / f"REVIEW_{tag}.md").write_text(review_md, encoding="utf-8")
    copied.append(f"REVIEW_{tag}.md")

    files = sorted({e["path"] for e in manifest_entries(out)})
    (out / "README.md").write_text(
        readme_text(head, files + ["README.md"], ledger_rows), encoding="utf-8"
    )

    manifest = write_manifest(out, head)
    problems = verify_manifest(out)
    if problems:
        raise OcSealError(f"the package does not verify as written: {problems[:5]}")
    print(f"package     {out}")
    print(f"files       {manifest['file_count']}  "
          f"{human_bytes(manifest['bytes_total'])}")
    print(f"package_sha256 {manifest['package_sha256']}")
    print(f"manifest re-verified OK ({len(copied)} artifacts placed)")
    package_sha = manifest["package_sha256"]
    print("\nNEXT: record the package on the checkpoint with "
          "`oc_checkpoint.py record-review ... --review-json <rev.json>` where "
          "the review names package_path=" + str(out) + " and package_sha256="
          + package_sha + ", then regenerate OC/REVIEW_QUEUE.md with "
          "oc_dashboard.py.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    problems = verify_manifest(args.package)
    manifest = json.loads((Path(args.package) / MANIFEST_NAME).read_text(encoding="utf-8")) \
        if (Path(args.package) / MANIFEST_NAME).is_file() else {}
    print(f"package  {args.package}")
    print(f"files    {manifest.get('file_count', '-')}")
    print(f"digest   {manifest.get('package_sha256', '-')}")
    for line in problems:
        print(f"  PROBLEM {line}")
    print("VERIFIED" if not problems else f"{len(problems)} PROBLEM(S)")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="write the review package for one checkpoint")
    p.add_argument("--registry", type=Path, default=None)
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dossier", type=Path, default=None)
    p.add_argument("--dossier-b2", type=Path, default=None)
    p.add_argument("--per-case", type=Path, default=None)
    p.add_argument("--per-case-b2", type=Path, default=None)
    p.add_argument("--canonical-reread", type=Path, default=None)
    p.add_argument("--receipt", type=Path, default=None)
    p.add_argument("--verdict", type=Path, default=None)
    p.add_argument("--build-receipt", type=Path, default=None)
    p.add_argument("--clone-receipt", type=Path, action="append", default=[])
    p.add_argument("--bench", type=Path, default=None)
    p.add_argument("--export", type=Path, default=None)
    p.add_argument("--media", type=Path, action="append", default=[])
    p.add_argument("--extra", type=Path, action="append", default=[])
    p.add_argument("--overwrite", action="store_true",
                   help="move an existing package aside (never write over it)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("verify", help="re-check every digest in a package")
    p.add_argument("--package", type=Path, required=True)
    p.set_defaults(func=cmd_verify)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = oc_root(args.root)
    if getattr(args, "registry", None) is None:
        args.registry = root / "registry"
    if getattr(args, "ledger", None) is None and hasattr(args, "ledger"):
        args.ledger = root / "ledger.jsonl"
    try:
        return int(args.func(args))
    except (OcCheckpointError, OcSealError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
