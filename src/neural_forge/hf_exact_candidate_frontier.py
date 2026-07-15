"""Bounded retention for exact-stage Qwen research candidates.

This module is intentionally downstream of the authenticated stage runner.  It
does not train, bake, evaluate, or mutate a model.  It authenticates one
terminal run, archives either its factor-only selected state or its already
accepted standalone child, and maintains at most three behavior-ranked records
per comparable evaluation contract.

Conservative v1 does not delete artifacts.  It durably records dominated
bundles in ``pending_prune`` for manual review, but exposes no recursive
deletion implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


FRONTIER_SCHEMA = "neural-forge-exact-candidate-frontier-v1"
RECORD_SCHEMA = "neural-forge-exact-candidate-record-v1"
RUN_SCHEMA = "neural-forge-qwen-exact-target-native-stage-v1"
PREREGISTRATION_SCHEMA = "neural-forge-qwen-exact-target-native-preregistration-v1"
GUIDED_RUN_SCHEMA = "neural-forge-qwen-exact-target-native-guided-continuation-v1"
GUIDED_WINNER_PUBLICATION_RUN_SCHEMA = (
    "neural-forge-qwen-guided-winner-publication-v2"
)
GUIDED_PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-exact-target-native-guided-continuation-preregistration-v1"
)
GUIDED_WINNER_PUBLICATION_PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-guided-winner-publication-preregistration-v2"
)
GUIDED_GUD_WINNER_PUBLICATION_RUN_SCHEMA = (
    "neural-forge-qwen-guided-gud-winner-publication-v1"
)
GUIDED_GUD_WINNER_PUBLICATION_PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-guided-gud-winner-publication-preregistration-v1"
)
GUIDED_GUD_TOURNAMENT_PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-exact-target-native-guided-gud-candidate-tournament-"
    "preregistration-v1"
)
GUIDED_CANDIDATE_TOURNAMENT_PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-exact-target-native-guided-candidate-tournament-"
    "preregistration-v1"
)
GUIDED_GUD_SEALED_CONSUMPTION_SCHEMA = (
    "neural-forge-guided-gud-sealed-consumption-v1"
)
EXACT_TARGET_COMPILER_TRANSACTION_INSTALLATION_SCHEMA = (
    "neural-forge-exact-target-compiler-transaction-installation-v1"
)
EXACT_TARGET_COMPILER_TRANSACTION_PUBLICATION_SCHEMA = (
    "neural-forge-exact-target-compiler-transaction-publication-v1"
)
EXTERNAL_BEHAVIOR_GATE_SCHEMA = "neural-forge-external-behavior-gate-receipt-v1"
GUIDED_QKVO_INSTALLATION_SCHEMA = (
    "neural-forge-guided-dense-gqa-qkvo-residual-installation-v1"
)
GUIDED_QKVO_BAKE_SCHEMA = "neural-forge-guided-dense-gqa-qkvo-residual-bake-v1"
DENSE_QKVO_INSTALLATION_SCHEMA = (
    "neural-forge-dense-gqa-qkvo-residual-installation-v2"
)
GUIDED_GUD_INSTALLATION_SCHEMA = (
    "neural-forge-guided-dense-swiglu-projection-residual-installation-v1"
)
GUIDED_GUD_BAKE_SCHEMA = (
    "neural-forge-guided-dense-swiglu-projection-residual-bake-v1"
)
DENSE_GUD_INSTALLATION_SCHEMA = (
    "neural-forge-dense-swiglu-projection-residual-installation-v2"
)
LEXICAL_BAKE_SCHEMA = "neural-forge-dense-lexical-residual-bake-v1"
SOURCE_V2_FACTOR_SCHEMA = "neural-forge-target-native-factor-pack-v2"
MAX_RETAINED_PER_CONTRACT = 3
QWEN3_1P4B_EXACT_PARAMETERS = 1_410_567_168
GUIDED_QKVO_FACTOR_PARAMETERS = 7_127_040
GUIDED_GUD_FACTOR_PARAMETERS = 4_792_320
SOURCE_V2_FACTOR_COUNT = 18

ArtifactKind = Literal["research-factor-pack", "accepted-standalone-child"]
TruthLabel = Literal["research-factor-only-candidate", "accepted-standalone"]
FactorAuthenticator = Callable[[Path, int | None], Mapping[str, Any]]
PreregistrationAuthenticator = Callable[[Path, str], Mapping[str, Any]]
AcceptedChildAuthenticator = Callable[[Path, int], Mapping[str, Any]]
MethodGateAuthenticator = Callable[
    [Mapping[str, Any], Mapping[str, Any] | None],
    tuple[bool, Mapping[str, Any]],
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FACTOR_FILENAME = "rejected_factor_pack.pt"
_SELECTED_FACTOR_FILENAME = "selected_factor_pack.pt"
_GUIDED_SELECTED_FACTOR_FILENAME = "selected_guided_factor_pack.pt"
_CHILD_DIRECTORY = "accepted_child"
_GUIDED_FACTOR_COUNT = 26
_GUIDED_FACTOR_PARAMETERS = 28_868_608
_BANNED_METRIC_WORDS = (
    "weight",
    "activation",
    "hidden_state",
    "attention",
    "residual",
    "projection",
    "similarity",
    "numeric_matching",
)


class ExactCandidateFrontierError(RuntimeError):
    """A candidate, frontier, or deletion proof is invalid."""


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ExactCandidateFrontierError(
            "Frontier provenance is not finite canonical JSON."
        ) from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ExactCandidateFrontierError(f"{name} must be a SHA-256 string.")
    normalized = value.removeprefix("sha256:").lower()
    if not _SHA256.fullmatch(normalized):
        raise ExactCandidateFrontierError(f"{name} must be a SHA-256 string.")
    return normalized


def strict_json(path: Path | str) -> Any:
    source = Path(path)

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ExactCandidateFrontierError(
                    f"Duplicate JSON key {key!r} in {source}."
                )
            result[key] = value
        return result

    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ExactCandidateFrontierError(
                    f"Non-finite JSON constant {value!r} in {source}."
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExactCandidateFrontierError(
            f"Cannot read strict JSON {source}."
        ) from error


def file_sha256(path: Path | str) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ExactCandidateFrontierError(f"Cannot hash artifact {source}.") from error
    return digest.hexdigest()


def _has_reparse_flag(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError as error:
        raise ExactCandidateFrontierError(f"Cannot inspect path {path}.") from error
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(value, "st_file_attributes", 0)
    return bool(path.is_symlink() or (flag and attributes & flag))


def _assert_ordinary_file(path: Path, name: str) -> None:
    if not path.is_file() or _has_reparse_flag(path):
        raise ExactCandidateFrontierError(f"{name} must be an ordinary file: {path}")


def _assert_ordinary_directory(path: Path, name: str) -> None:
    if not path.is_dir() or _has_reparse_flag(path):
        raise ExactCandidateFrontierError(
            f"{name} must be an ordinary directory: {path}"
        )


def _assert_ordinary_protected_path(path: Path, name: str) -> None:
    if _has_reparse_flag(path) or not (path.is_file() or path.is_dir()):
        raise ExactCandidateFrontierError(
            f"{name} must be an ordinary protected file or directory: {path}"
        )


def _tree_records(root: Path) -> list[dict[str, Any]]:
    _assert_ordinary_directory(root, "artifact root")
    try:
        root_device = int(root.stat().st_dev)
    except OSError as error:
        raise ExactCandidateFrontierError(
            f"Cannot inspect artifact root device {root}."
        ) from error
    records: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise ExactCandidateFrontierError(
                f"Cannot enumerate artifact directory {directory}."
            ) from error
        for entry in entries:
            path = Path(entry.path)
            if _has_reparse_flag(path):
                raise ExactCandidateFrontierError(
                    f"Artifact tree contains a symlink, junction, or reparse point: {path}"
                )
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
                entry_device = int(entry.stat(follow_symlinks=False).st_dev)
            except OSError as error:
                raise ExactCandidateFrontierError(
                    f"Cannot classify artifact entry {path}."
                ) from error
            if os.path.ismount(path) or (
                os.name == "posix" and entry_device != root_device
            ):
                raise ExactCandidateFrontierError(
                    f"Artifact tree crosses a nested mount or filesystem device: {path}"
                )
            if is_directory:
                visit(path)
                continue
            if not is_file:
                raise ExactCandidateFrontierError(
                    f"Artifact contains a non-file entry: {path}"
                )
            relative = path.relative_to(root).as_posix()
            records.append(
                {
                    "path": relative,
                    "size_bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )

    visit(root)
    if not records:
        raise ExactCandidateFrontierError(f"Artifact directory is empty: {root}")
    return records


def artifact_tree(root: Path | str) -> dict[str, Any]:
    directory = Path(root)
    records = _tree_records(directory)
    return {
        "file_count": len(records),
        "total_bytes": sum(int(item["size_bytes"]) for item in records),
        "tree_sha256": canonical_sha256(records),
        "files": records,
    }


def _record_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return canonical_sha256(payload)


def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and (_has_reparse_flag(path) or not path.is_file()):
            raise ExactCandidateFrontierError(
                f"Refusing to replace a non-ordinary manifest: {path}"
            )
        os.replace(temporary, path)
        if os.name == "posix":
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            try:
                directory_descriptor = os.open(path.parent, flags)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError as error:
                raise ExactCandidateFrontierError(
                    f"Cannot durably fsync manifest directory {path.parent}."
                ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _finite_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExactCandidateFrontierError(f"{name} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ExactCandidateFrontierError(f"{name} must be in [0, 1].")
    return number


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExactCandidateFrontierError(f"{name} must be a nonnegative integer.")
    return value


def rollout_first_rank(metrics: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the canonical ascending research-retention key.

    This is the target-native trainer's rollout-first acquisition policy.  No
    fit, sealed, weight, activation, hidden-state, or similarity measurement is
    accepted by this function.
    """

    lowered_names = tuple(str(name).lower() for name in metrics)
    if any(word in name for name in lowered_names for word in _BANNED_METRIC_WORDS):
        raise ExactCandidateFrontierError(
            "Candidate retention metrics contain an internal numeric diagnostic."
        )
    required_counts = (
        "exact_rollout_count",
        "lcp_at_least_4_count",
        "lcp_token_count",
        "token_match_count",
        "longest_run_excess",
        "action_agreement_count",
        "rollout_count",
        "rollout_token_count",
        "action_count",
    )
    counts = {
        name: _nonnegative_int(metrics.get(name), f"candidate_metrics.{name}")
        for name in required_counts
    }
    if (
        counts["action_count"] <= 0
        or counts["rollout_count"] <= 0
        or counts["rollout_token_count"] <= 0
    ):
        raise ExactCandidateFrontierError(
            "Candidate behavior counts must include actions and free rollouts."
        )
    if counts["exact_rollout_count"] > counts["rollout_count"]:
        raise ExactCandidateFrontierError(
            "candidate_metrics exact rollout count exceeds rollout count."
        )
    if counts["lcp_at_least_4_count"] > counts["rollout_count"]:
        raise ExactCandidateFrontierError(
            "candidate_metrics LCP-at-least-four count exceeds rollout count."
        )
    if (
        counts["lcp_token_count"] > counts["rollout_token_count"]
        or counts["token_match_count"] > counts["rollout_token_count"]
    ):
        raise ExactCandidateFrontierError(
            "candidate_metrics token counts exceed rollout token count."
        )
    if counts["action_agreement_count"] > counts["action_count"]:
        raise ExactCandidateFrontierError(
            "candidate_metrics action agreements exceed action count."
        )
    if counts["lcp_token_count"] < 4 * counts["lcp_at_least_4_count"]:
        raise ExactCandidateFrontierError(
            "candidate_metrics LCP-at-least-four count exceeds its token evidence."
        )
    if counts["token_match_count"] < counts["lcp_token_count"]:
        raise ExactCandidateFrontierError(
            "candidate_metrics token matches cannot be fewer than LCP tokens."
        )
    repeat = _finite_fraction(
        metrics.get("repeat_fraction_excess"),
        "candidate_metrics.repeat_fraction_excess",
    )
    adjacent = _finite_fraction(
        metrics.get("adjacent_repeat_fraction_excess"),
        "candidate_metrics.adjacent_repeat_fraction_excess",
    )
    _finite_fraction(
        metrics.get("parent_action_topk_fraction"),
        "candidate_metrics.parent_action_topk_fraction",
    )
    mean_logprob = metrics.get("parent_action_mean_logprob")
    mean_rank = metrics.get("parent_action_mean_rank")
    worst_rank = metrics.get("parent_action_worst_rank")
    if (
        isinstance(mean_logprob, bool)
        or not isinstance(mean_logprob, (int, float))
        or not math.isfinite(float(mean_logprob))
        or float(mean_logprob) > 1e-9
        or isinstance(mean_rank, bool)
        or not isinstance(mean_rank, (int, float))
        or not math.isfinite(float(mean_rank))
        or float(mean_rank) < 1.0
        or isinstance(worst_rank, bool)
        or not isinstance(worst_rank, int)
        or worst_rank < 1
        or float(mean_rank) > worst_rank
    ):
        raise ExactCandidateFrontierError(
            "Candidate parent-action log-probability/rank metrics are inconsistent."
        )
    require_sha256(metrics.get("report_sha256"), "candidate_metrics.report_sha256")
    if metrics.get("split") != "calibration-selection":
        raise ExactCandidateFrontierError(
            "Candidate retention requires calibration-selection output metrics."
        )
    for numerator, denominator, fraction in (
        ("exact_rollout_count", "rollout_count", "exact_rollout_fraction"),
        ("token_match_count", "rollout_token_count", "token_match_fraction"),
        ("lcp_token_count", "rollout_token_count", "lcp_fraction"),
        ("action_agreement_count", "action_count", "action_accuracy"),
    ):
        observed = _finite_fraction(
            metrics.get(fraction), f"candidate_metrics.{fraction}"
        )
        denominator_value = counts[denominator]
        expected = (
            0.0 if denominator_value == 0 else counts[numerator] / denominator_value
        )
        if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ExactCandidateFrontierError(
                f"candidate_metrics.{fraction} disagrees with its counts."
            )
    return (
        -counts["exact_rollout_count"],
        -counts["lcp_at_least_4_count"],
        -counts["lcp_token_count"],
        -counts["token_match_count"],
        round(repeat * 1_000_000),
        round(adjacent * 1_000_000),
        counts["longest_run_excess"],
        -counts["action_agreement_count"],
    )


_GUIDED_METHOD_CHECKS = (
    "selected_step_acquired",
    "candidate_has_one_lcp_at_least_4_rollout",
    "candidate_repeat_fraction_nonregression_vs_step0",
    "candidate_adjacent_repeat_fraction_nonregression_vs_step0",
)

_GUIDED_TERMINAL_COMMON_FIELDS = frozenset(
    {
        "schema",
        "status",
        "preregistration",
        "preregistration_sha256",
        "evaluation_contract_sha256",
        "parent",
        "target",
        "compiler_split",
        "splits",
        "thresholds",
        "guided_qkvo",
        "source",
        "audit",
        "implementation_authentication",
        "selection_domain",
        "method_gate",
        "internal_numeric_matching_can_promote",
        "sealed_evaluation_locked_until_candidate_passes",
        "models_loaded",
        "gpu_work_launched",
        "output_root",
        "compiler_report",
        "inherited_installation",
        "source_v2_restore",
        "guided_installation",
        "last_completed_phase",
        "trainer_evidence_sha256",
        "trainer_evidence",
        "factor_acquisition",
        "selected_factor_pack",
        "pre_bake_behavior_decisions",
        "guided_method_decision",
        "selection_used_internal_numeric_matching",
        "standalone_compaction_passed",
        "model_checkpoint_written",
        "rollback_exact",
        "retained_artifacts",
    }
)
_GUIDED_TERMINAL_OPTIONAL_FIELDS = frozenset(
    {
        "compiled_child_cache",
        "reason",
        "rollback_state_sha256",
        "guided_publication_handoff",
        "publication",
        "post_bake_behavior_decisions",
        "post_bake_guided_method_decision",
        "prepublication_native_restored",
        "fresh_reload_certificate",
        "fresh_reload_guided_method_gate",
        "model_save_count",
    }
)
_GUIDED_ACCEPTED_REQUIRED_FIELDS = frozenset(
    {
        "guided_publication_handoff",
        "publication",
        "post_bake_behavior_decisions",
        "post_bake_guided_method_decision",
        "fresh_reload_certificate",
        "fresh_reload_guided_method_gate",
        "model_save_count",
    }
)
_GUIDED_REJECTED_REQUIRED_FIELDS = frozenset({"reason", "rollback_state_sha256"})


def _validate_guided_terminal_report_schema(
    report: Mapping[str, Any], status: str
) -> None:
    """Reject fields outside the real v1 runner's terminal report domains."""

    fields = set(report)
    allowed = _GUIDED_TERMINAL_COMMON_FIELDS | _GUIDED_TERMINAL_OPTIONAL_FIELDS
    required = _GUIDED_TERMINAL_COMMON_FIELDS | (
        _GUIDED_ACCEPTED_REQUIRED_FIELDS
        if status == "accepted"
        else _GUIDED_REJECTED_REQUIRED_FIELDS
    )
    if fields - allowed or required - fields:
        raise ExactCandidateFrontierError(
            "Guided terminal run report does not have the exact v1 field schema."
        )
    accepted_only = {
        "fresh_reload_certificate",
        "fresh_reload_guided_method_gate",
        "model_save_count",
    }
    if status == "accepted":
        if "reason" in fields or "rollback_state_sha256" in fields:
            raise ExactCandidateFrontierError(
                "Accepted guided terminal report contains rejected-only fields."
            )
    elif fields & accepted_only:
        raise ExactCandidateFrontierError(
            "Rejected guided terminal report contains accepted-only fields."
        )


def _dynamic_method_rank(decision: Mapping[str, Any] | None) -> tuple[int, ...]:
    """Rank only authenticated output-behavior method checks.

    The guided method gate is dynamic because its repetition ceilings are the
    selected run's authenticated step-zero behavior, not fixed internal tensor
    targets.  Only the four boolean output-behavior checks influence ranking.
    Diagnostic magnitudes, weights, activations, and similarity values never do.
    """

    if decision is None:
        return ()
    checks = decision.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != set(_GUIDED_METHOD_CHECKS)
        or any(not isinstance(checks.get(name), bool) for name in _GUIDED_METHOD_CHECKS)
        or not isinstance(decision.get("passed"), bool)
        or decision.get("passed") is not all(bool(checks[name]) for name in checks)
        or decision.get("internal_numeric_matching_consulted") is not False
    ):
        raise ExactCandidateFrontierError(
            "Authenticated guided method decision is malformed."
        )
    declared = require_sha256(
        decision.get("decision_sha256"), "guided_method_decision.decision_sha256"
    )
    payload = dict(decision)
    payload.pop("decision_sha256", None)
    if canonical_sha256(payload) != declared:
        raise ExactCandidateFrontierError(
            "Authenticated guided method decision hash does not verify."
        )
    # A complete method-positive result ranks first, then partial progress.  No
    # numeric value from the decision is used in this key.
    return (
        -int(bool(decision["passed"])),
        -sum(int(bool(checks[name])) for name in _GUIDED_METHOD_CHECKS),
    )


def behavior_rank(
    metrics: Mapping[str, Any],
    dynamic_method_decision: Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    """Return the complete behavior-only frontier key for one run schema."""

    return (
        *_dynamic_method_rank(dynamic_method_decision),
        *rollout_first_rank(metrics),
    )


def _candidate_metric_geometry(
    report: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> tuple[int, int]:
    splits = report.get("splits")
    candidate_split = splits.get("candidate") if isinstance(splits, Mapping) else None
    training = preregistration.get("training")
    if not isinstance(candidate_split, Mapping) or not isinstance(training, Mapping):
        raise ExactCandidateFrontierError(
            "Authenticated candidate split or training schedule is missing."
        )
    rollout_count = _nonnegative_int(
        candidate_split.get("row_count"), "splits.candidate.row_count"
    )
    horizon = _nonnegative_int(
        training.get("evaluation_horizon"), "training.evaluation_horizon"
    )
    if rollout_count <= 0 or horizon <= 0:
        raise ExactCandidateFrontierError(
            "Candidate split size and evaluation horizon must be positive."
        )
    return rollout_count, horizon


def _validate_candidate_metric_geometry(
    metrics: Mapping[str, Any], *, rollout_count: int, horizon: int
) -> None:
    rollout_first_rank(metrics)
    expected_tokens = rollout_count * horizon
    expected = {
        "rollout_count": rollout_count,
        "rollout_token_count": expected_tokens,
        "action_count": expected_tokens,
    }
    if any(metrics.get(name) != value for name, value in expected.items()):
        raise ExactCandidateFrontierError(
            "Candidate behavior denominators disagree with the authenticated split "
            "and evaluation horizon."
        )
    exact = int(metrics["exact_rollout_count"])
    lcp4 = int(metrics["lcp_at_least_4_count"])
    lcp_tokens = int(metrics["lcp_token_count"])
    token_matches = int(metrics["token_match_count"])
    action_matches = int(metrics["action_agreement_count"])
    longest_run_excess = int(metrics["longest_run_excess"])
    if horizon < 4:
        if lcp4 != 0:
            raise ExactCandidateFrontierError(
                "LCP-at-least-four evidence is impossible below a four-token horizon."
            )
    elif lcp4 < exact:
        raise ExactCandidateFrontierError(
            "Every exact rollout must also be an LCP-at-least-four rollout."
        )
    exact_tokens = exact * horizon
    if (
        lcp_tokens < exact_tokens
        or token_matches < exact_tokens
        or action_matches < exact_tokens
        or longest_run_excess > max(0, horizon - 1)
    ):
        raise ExactCandidateFrontierError(
            "Candidate exact-rollout or repetition counts are logically inconsistent."
        )


def _resolve_binding(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ExactCandidateFrontierError(f"{name} must be a nonempty path.")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _stable_artifact_authentication(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactCandidateFrontierError(
            f"artifact_authentication.{name} must be an object."
        )
    rows = value.get("required_files")
    tree = value.get("tree")
    if (
        not isinstance(rows, list)
        or value.get("authenticated_before_model_access") is not True
    ):
        raise ExactCandidateFrontierError(
            f"artifact_authentication.{name} is incomplete."
        )
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ExactCandidateFrontierError(
                f"artifact_authentication.{name}.required_files[{index}] is malformed."
            )
        relative = row.get("path")
        if not isinstance(relative, str) or not relative:
            raise ExactCandidateFrontierError("Authenticated artifact path is invalid.")
        normalized_rows.append(
            {
                "path": relative,
                "sha256": require_sha256(
                    row.get("sha256"), f"artifact_authentication.{name}.sha256"
                ),
                "size_bytes": _nonnegative_int(
                    row.get("size_bytes"),
                    f"artifact_authentication.{name}.size_bytes",
                ),
            }
        )
    normalized_tree = None
    if tree is not None:
        if not isinstance(tree, Mapping) or set(tree) != {
            "file_count",
            "total_bytes",
            "tree_sha256",
        }:
            raise ExactCandidateFrontierError(
                f"artifact_authentication.{name}.tree is malformed."
            )
        normalized_tree = {
            "file_count": _nonnegative_int(
                tree.get("file_count"), f"artifact_authentication.{name}.file_count"
            ),
            "total_bytes": _nonnegative_int(
                tree.get("total_bytes"), f"artifact_authentication.{name}.total_bytes"
            ),
            "tree_sha256": require_sha256(
                tree.get("tree_sha256"),
                f"artifact_authentication.{name}.tree_sha256",
            ),
        }
    if not normalized_rows and normalized_tree is None:
        raise ExactCandidateFrontierError(
            f"artifact_authentication.{name} has no content identity."
        )
    return {
        "required_files": sorted(normalized_rows, key=lambda item: item["path"]),
        "tree": normalized_tree,
    }


def _trainer_evidence(report: Mapping[str, Any]) -> Mapping[str, Any]:
    evidence = report.get("trainer_evidence")
    if not isinstance(evidence, Mapping):
        raise ExactCandidateFrontierError("Terminal report lacks trainer evidence.")
    expected = require_sha256(
        report.get("trainer_evidence_sha256"), "trainer_evidence_sha256"
    )
    if canonical_sha256(evidence) != expected:
        raise ExactCandidateFrontierError("Trainer evidence hash does not verify.")
    return evidence


def _decision_metrics(
    report: Mapping[str, Any],
    *,
    post_bake: bool,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    field = (
        "post_bake_behavior_decisions" if post_bake else "pre_bake_behavior_decisions"
    )
    decisions = report.get(field)
    if decisions is None and not post_bake:
        decisions = report.get("behavior_decisions")
    if not isinstance(decisions, Mapping):
        raise ExactCandidateFrontierError(f"Terminal report lacks {field}.")
    candidate = decisions.get("candidate")
    if not isinstance(candidate, Mapping) or not isinstance(
        candidate.get("passed"), bool
    ):
        raise ExactCandidateFrontierError(
            f"Terminal report {field}.candidate is malformed."
        )
    metrics = candidate.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ExactCandidateFrontierError(
            f"Terminal report {field}.candidate lacks metrics."
        )
    payload = dict(metrics)
    rollout_first_rank(payload)
    return payload, decisions


def _selected_factor_state(
    report: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    expected_rollout_count: int,
    expected_horizon: int,
) -> tuple[str, int, dict[str, Any], dict[str, Any]]:
    acquisition = report.get("factor_acquisition")
    if (
        not isinstance(acquisition, Mapping)
        or acquisition.get("target_native_factors_acquired_behavior") is not True
    ):
        raise ExactCandidateFrontierError(
            "Terminal report does not prove behavior-acquired factors."
        )
    selected_step = acquisition.get("selected_step")
    if (
        isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or selected_step <= 0
    ):
        raise ExactCandidateFrontierError(
            "Step-zero blank states are not retention candidates."
        )
    if evidence.get("selected_step") != selected_step:
        raise ExactCandidateFrontierError(
            "Trainer evidence and factor acquisition selected steps disagree."
        )
    selected_sha = require_sha256(
        acquisition.get("selected_factor_state_sha256"),
        "factor_acquisition.selected_factor_state_sha256",
    )
    initial_sha = require_sha256(
        acquisition.get("step0_factor_state_sha256"),
        "factor_acquisition.step0_factor_state_sha256",
    )
    if selected_sha == initial_sha:
        raise ExactCandidateFrontierError(
            "Selected factor state is byte-identical to step zero."
        )
    manifest = evidence.get("factor_pack_manifest")
    checkpoints = evidence.get("checkpoints")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("factor_state_sha256") != selected_sha
        or not isinstance(checkpoints, list)
    ):
        raise ExactCandidateFrontierError(
            "Trainer factor manifest does not bind the selected state."
        )
    checkpoint_steps: list[int] = []
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise ExactCandidateFrontierError(
                f"Trainer checkpoint {index} is not an object."
            )
        step = checkpoint.get("step")
        metrics = checkpoint.get("metrics")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or not isinstance(metrics, Mapping)
        ):
            raise ExactCandidateFrontierError(
                f"Trainer checkpoint {index} has invalid step or metrics."
            )
        require_sha256(
            checkpoint.get("factor_state_sha256"),
            f"trainer checkpoint {index} factor_state_sha256",
        )
        _validate_candidate_metric_geometry(
            metrics,
            rollout_count=expected_rollout_count,
            horizon=expected_horizon,
        )
        checkpoint_steps.append(step)
    if (
        not checkpoint_steps
        or checkpoint_steps[0] != 0
        or checkpoint_steps != sorted(set(checkpoint_steps))
    ):
        raise ExactCandidateFrontierError(
            "Trainer checkpoints must be unique, ordered, and retain step zero."
        )
    step0 = checkpoints[0]
    if step0.get("factor_state_sha256") != initial_sha:
        raise ExactCandidateFrontierError(
            "Trainer step-zero checkpoint does not bind acquisition step zero."
        )
    selected = next(
        (
            value
            for value in checkpoints
            if isinstance(value, Mapping) and value.get("step") == selected_step
        ),
        None,
    )
    if (
        not isinstance(selected, Mapping)
        or selected.get("factor_state_sha256") != selected_sha
        or not isinstance(selected.get("metrics"), Mapping)
    ):
        raise ExactCandidateFrontierError(
            "Trainer checkpoints do not bind the selected factor state."
        )
    return (
        selected_sha,
        selected_step,
        dict(selected["metrics"]),
        dict(step0["metrics"]),
    )


def _comparison_and_target_contracts(
    report: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    authenticated_preregistration: Mapping[str, Any],
) -> tuple[str, str, int]:
    evaluation = require_sha256(
        report.get("evaluation_contract_sha256"), "evaluation_contract_sha256"
    )
    guided = report.get("schema") == GUIDED_RUN_SCHEMA
    retention = authenticated_preregistration.get("retention_authentication")
    authentication = (
        retention.get("artifact_authentication")
        if guided and isinstance(retention, Mapping)
        else report.get("artifact_authentication")
    )
    if not isinstance(authentication, Mapping):
        raise ExactCandidateFrontierError(
            "Terminal report lacks artifact authentication."
        )
    parent = _stable_artifact_authentication(authentication.get("parent"), "parent")
    blank = _stable_artifact_authentication(
        authentication.get("blank_target"), "blank_target"
    )
    parent_binding = (
        retention.get("parent")
        if guided and isinstance(retention, Mapping)
        else preregistration.get("parent")
    )
    blank_binding = (
        retention.get("blank_target")
        if guided and isinstance(retention, Mapping)
        else preregistration.get("blank_target")
    )
    if not isinstance(parent_binding, Mapping) or not isinstance(
        blank_binding, Mapping
    ):
        raise ExactCandidateFrontierError(
            "Preregistration lacks parent or blank-target bindings."
        )
    parent_parameters = _nonnegative_int(
        parent_binding.get("exact_unique_parameters"),
        "preregistration.parent.exact_unique_parameters",
    )
    target_parameters = _nonnegative_int(
        blank_binding.get("exact_unique_parameters"),
        "preregistration.blank_target.exact_unique_parameters",
    )
    target_contract = {
        "parent_exact_unique_parameters": parent_parameters,
        "target_exact_unique_parameters": target_parameters,
        "parent_artifact": parent,
        "blank_target_artifact": blank,
    }
    return evaluation, canonical_sha256(target_contract), target_parameters


def _protected_roots(
    report: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    prereg_path: Path,
    authenticated_preregistration: Mapping[str, Any],
) -> dict[str, str]:
    guided = report.get("schema") == GUIDED_RUN_SCHEMA
    retention = authenticated_preregistration.get("retention_authentication")
    blank = (
        retention.get("blank_target")
        if guided and isinstance(retention, Mapping)
        else preregistration.get("blank_target")
    )
    if not isinstance(blank, Mapping):
        raise ExactCandidateFrontierError("Preregistration lacks blank_target.")
    blank_path = _resolve_binding(
        prereg_path.parent,
        blank.get("path"),
        "preregistration.blank_target.path",
    )
    _assert_ordinary_directory(blank_path, "canonical blank target")
    result = {"canonical_blank": str(blank_path)}
    cache = report.get("compiler_cache")
    if guided:
        cache = (
            {"root": retention.get("compiler_cache_root")}
            if isinstance(retention, Mapping)
            else None
        )
    if cache is not None:
        if not isinstance(cache, Mapping):
            raise ExactCandidateFrontierError("compiler_cache report is malformed.")
        cache_path = _resolve_binding(
            prereg_path.parent, cache.get("root"), "compiler_cache.root"
        )
        _assert_ordinary_directory(cache_path, "compiler cache")
        result["compiler_cache"] = str(cache_path)
    if guided and isinstance(retention, Mapping):
        sources = retention.get("protected_sources")
        if not isinstance(sources, Mapping):
            raise ExactCandidateFrontierError(
                "Guided retention authentication lacks protected source paths."
            )
        for name, value in sources.items():
            if not isinstance(name, str) or not name:
                raise ExactCandidateFrontierError(
                    "Guided protected source names must be nonempty strings."
                )
            path = _resolve_binding(
                prereg_path.parent,
                value,
                f"retention_authentication.protected_sources.{name}",
            )
            _assert_ordinary_file(path, f"guided protected source {name}")
            identity = file_sha256(path)
            binding_identity = canonical_sha256({"path": str(path), "sha256": identity})
            result[f"guided_{name}_{binding_identity[:16]}"] = str(path)
    return result


def _validate_guided_retention_authentication(
    value: Any,
) -> Mapping[str, Any]:
    expected = {
        "artifact_authentication",
        "parent",
        "blank_target",
        "compiler_cache_root",
        "lexical_rank",
        "source_v2_tensor_sha256s",
        "protected_sources",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ExactCandidateFrontierError(
            "Guided canonical preflight lacks the exact retention authentication."
        )
    authentication = value.get("artifact_authentication")
    if not isinstance(authentication, Mapping) or set(authentication) < {
        "parent",
        "blank_target",
    }:
        raise ExactCandidateFrontierError(
            "Guided retention artifact authentication is incomplete."
        )
    _stable_artifact_authentication(authentication.get("parent"), "parent")
    _stable_artifact_authentication(authentication.get("blank_target"), "blank_target")
    for name in ("parent", "blank_target"):
        binding = value.get(name)
        if (
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("path"), str)
            or not binding.get("path")
            or _nonnegative_int(
                binding.get("exact_unique_parameters"),
                f"retention_authentication.{name}.exact_unique_parameters",
            )
            <= 0
        ):
            raise ExactCandidateFrontierError(
                f"Guided retention {name} binding is malformed."
            )
    cache = value.get("compiler_cache_root")
    if not isinstance(cache, str) or not cache:
        raise ExactCandidateFrontierError(
            "Guided retention compiler-cache root is malformed."
        )
    if (
        _nonnegative_int(
            value.get("lexical_rank"), "retention_authentication.lexical_rank"
        )
        <= 0
    ):
        raise ExactCandidateFrontierError("Guided retention lexical rank is malformed.")
    source_hashes = value.get("source_v2_tensor_sha256s")
    if (
        not isinstance(source_hashes, Mapping)
        or len(source_hashes) != 18
        or any(not isinstance(name, str) or not name for name in source_hashes)
    ):
        raise ExactCandidateFrontierError(
            "Guided retention source v2 tensor hashes are incomplete."
        )
    for name, digest in source_hashes.items():
        require_sha256(
            digest, f"retention_authentication.source_v2_tensor_sha256s.{name}"
        )
    protected = value.get("protected_sources")
    if not isinstance(protected, Mapping) or not protected:
        raise ExactCandidateFrontierError(
            "Guided retention protected source map is incomplete."
        )
    return value


def _authenticate_canonical_preregistration(
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
    preregistration_sha256: str,
    report: Mapping[str, Any],
    authenticator: PreregistrationAuthenticator,
) -> Mapping[str, Any]:
    """Require the canonical runner's complete model-free preregistration audit."""

    if not callable(authenticator):
        raise ExactCandidateFrontierError(
            "A canonical preregistration authenticator is required."
        )
    implementation = preregistration.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ExactCandidateFrontierError(
            "Preregistration lacks canonical implementation source pins."
        )
    runner_sha = require_sha256(
        implementation.get("runner_sha256"), "implementation.runner_sha256"
    )
    trainer_sha = require_sha256(
        implementation.get("trainer_sha256"), "implementation.trainer_sha256"
    )
    trainer_module = implementation.get("trainer_module")
    if not isinstance(trainer_module, str) or not trainer_module:
        raise ExactCandidateFrontierError(
            "Preregistration trainer module pin is invalid."
        )
    authenticated = authenticator(preregistration_path, preregistration_sha256)
    if not isinstance(authenticated, Mapping):
        raise ExactCandidateFrontierError(
            "Canonical preregistration authenticator returned a non-mapping."
        )
    common_disagreement = (
        authenticated.get("status") != "preflight-passed"
        or authenticated.get("preregistration") != str(preregistration_path)
        or authenticated.get("preregistration_sha256") != preregistration_sha256
        or authenticated.get("evaluation_contract_sha256")
        != report.get("evaluation_contract_sha256")
        or authenticated.get("implementation_authentication")
        != report.get("implementation_authentication")
        or authenticated.get("splits") != report.get("splits")
        or authenticated.get("compiler_split") != report.get("compiler_split")
        or authenticated.get("thresholds") != report.get("thresholds")
    )
    if common_disagreement:
        raise ExactCandidateFrontierError(
            "Canonical preregistration preflight disagrees with the terminal report."
        )
    source_authentication = authenticated.get("implementation_authentication")
    if not isinstance(source_authentication, Mapping):
        raise ExactCandidateFrontierError(
            "Canonical preflight lacks implementation authentication."
        )
    if report.get("schema") == GUIDED_RUN_SCHEMA:
        if preregistration.get("schema") != GUIDED_PREREGISTRATION_SCHEMA:
            raise ExactCandidateFrontierError(
                "Guided terminal report uses the wrong preregistration schema."
            )
        for field in (
            "parent",
            "target",
            "guided_qkvo",
            "source",
            "audit",
            "selection_domain",
            "method_gate",
            "internal_numeric_matching_can_promote",
            "sealed_evaluation_locked_until_candidate_passes",
        ):
            if authenticated.get(field) != report.get(field):
                raise ExactCandidateFrontierError(
                    "Guided canonical preflight disagrees with the terminal report."
                )
        if (
            source_authentication.get("runner_sha256") != runner_sha
            or source_authentication.get("trainer_sha256") != trainer_sha
            or source_authentication.get("trainer_module") != trainer_module
            or report.get("internal_numeric_matching_can_promote") is not False
            or report.get("sealed_evaluation_locked_until_candidate_passes") is not True
            or preregistration.get("method_gate") != report.get("method_gate")
        ):
            raise ExactCandidateFrontierError(
                "Guided runner/trainer/method source pins do not authenticate."
            )
        _validate_guided_retention_authentication(
            authenticated.get("retention_authentication")
        )
    else:
        if authenticated.get("artifact_authentication") != report.get(
            "artifact_authentication"
        ):
            raise ExactCandidateFrontierError(
                "Canonical preregistration preflight disagrees with the terminal report."
            )
        runner_auth = source_authentication.get("runner")
        trainer_auth = source_authentication.get("trainer")
        if (
            not isinstance(runner_auth, Mapping)
            or runner_auth.get("sha256") != runner_sha
            or not isinstance(trainer_auth, Mapping)
            or trainer_auth.get("sha256") != trainer_sha
            or trainer_auth.get("module") != trainer_module
        ):
            raise ExactCandidateFrontierError(
                "Canonical runner/trainer source pins do not authenticate."
            )
    return authenticated


@dataclass(frozen=True, slots=True)
class TerminalCandidate:
    terminal_run_schema: str
    run_root: Path
    run_report_path: Path
    run_report_sha256: str
    preregistration_path: Path
    preregistration_sha256: str
    comparison_contract_sha256: str
    target_contract_sha256: str
    exact_unique_parameters: int
    artifact_kind: ArtifactKind
    truth_label: TruthLabel
    terminal_status: str
    terminal_reason: str | None
    candidate_metrics: Mapping[str, Any]
    dynamic_method_gate_decision: Mapping[str, Any] | None
    rollout_first_rank_key: tuple[int, ...]
    selected_step: int
    selected_factor_state_sha256: str
    content_identity_sha256: str
    source_artifact: Path
    source_artifact_identity_sha256: str
    source_artifact_tree: Mapping[str, Any] | None
    fresh_reload_verified: bool
    fresh_reload_certificate_sha256: str | None
    independent_child_authentication: Mapping[str, Any] | None
    standalone_compaction_passed: bool
    deployment_accepted: bool
    protected_roots: Mapping[str, str]

    @property
    def submission_identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "target_contract_sha256": self.target_contract_sha256,
                "comparison_contract_sha256": self.comparison_contract_sha256,
                "artifact_kind": self.artifact_kind,
                "content_identity_sha256": self.content_identity_sha256,
                "run_report_sha256": self.run_report_sha256,
            }
        )


_WINNER_PUBLICATION_METHOD_CHECKS = {
    "winner_is_post_step_zero",
    "candidate_has_one_lcp_at_least_4_rollout",
    "tournament_later_state_required_pareto_safety_vs_step_zero",
}


def _valid_winner_publication_method_decision(
    value: Any,
    *,
    winner_step: int | None = None,
    candidate_metrics: Mapping[str, Any] | None = None,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "passed",
        "checks",
        "winner_step",
        "candidate_lcp_at_least_4_count",
        "internal_numeric_matching_consulted",
        "decision_sha256",
    }:
        return False
    checks = value.get("checks")
    decision_sha = value.get("decision_sha256")
    valid = bool(
        isinstance(checks, Mapping)
        and set(checks) == _WINNER_PUBLICATION_METHOD_CHECKS
        and all(checks.get(name) is True for name in _WINNER_PUBLICATION_METHOD_CHECKS)
        and value.get("passed") is True
        and value.get("internal_numeric_matching_consulted") is False
        and _nonnegative_int(value.get("winner_step"), "winner_step") > 0
        and _nonnegative_int(
            value.get("candidate_lcp_at_least_4_count"),
            "candidate_lcp_at_least_4_count",
        )
        > 0
        and isinstance(decision_sha, str)
        and decision_sha
        == canonical_sha256(
            {name: item for name, item in value.items() if name != "decision_sha256"}
        )
    )
    if not valid:
        return False
    if winner_step is not None and value.get("winner_step") != winner_step:
        return False
    if candidate_metrics is not None:
        try:
            lcp_count = _nonnegative_int(
                candidate_metrics.get("lcp_at_least_4_count"),
                "candidate_metrics.lcp_at_least_4_count",
            )
        except ExactCandidateFrontierError:
            return False
        if value.get("candidate_lcp_at_least_4_count") != lcp_count:
            return False
    return True


def _valid_winner_publication_pareto_admission(
    winner: Any,
    semantic_contract: Any,
) -> bool:
    if not isinstance(winner, Mapping) or not isinstance(semantic_contract, Mapping):
        return False
    admission = winner.get("tournament_pareto_admission")
    required = {
        "passed",
        "winner_arm_id",
        "winner_step",
        "winner_callback_ordinal",
        "step_zero_callback_ordinal",
        "declared_pareto_safe_vs_step0",
        "objectives_equal_to_step_zero",
        "dominates_step_zero",
        "winner_metrics_sha256",
        "step_zero_metrics_sha256",
        "recomputed_from_authenticated_output_metrics",
        "internal_numeric_matching_consulted",
        "decision_sha256",
    }
    if not isinstance(admission, Mapping) or set(admission) != required:
        return False
    try:
        decision_sha = require_sha256(
            admission.get("decision_sha256"),
            "winner.tournament_pareto_admission.decision_sha256",
        )
        winner_metrics_sha = require_sha256(
            admission.get("winner_metrics_sha256"),
            "winner.tournament_pareto_admission.winner_metrics_sha256",
        )
        require_sha256(
            admission.get("step_zero_metrics_sha256"),
            "winner.tournament_pareto_admission.step_zero_metrics_sha256",
        )
        winner_step = _nonnegative_int(winner.get("step"), "winner.step")
        admission_step = _nonnegative_int(
            admission.get("winner_step"), "pareto_admission.winner_step"
        )
        winner_ordinal = _nonnegative_int(
            admission.get("winner_callback_ordinal"),
            "pareto_admission.winner_callback_ordinal",
        )
        step_zero_ordinal = _nonnegative_int(
            admission.get("step_zero_callback_ordinal"),
            "pareto_admission.step_zero_callback_ordinal",
        )
    except ExactCandidateFrontierError:
        return False
    equal = admission.get("objectives_equal_to_step_zero") is True
    dominates = admission.get("dominates_step_zero") is True
    metrics = winner.get("metrics")
    return bool(
        winner_step > 0
        and admission_step == winner_step
        and admission.get("winner_arm_id") == winner.get("arm_id")
        and winner_ordinal != step_zero_ordinal
        and admission.get("passed") is True
        and admission.get("declared_pareto_safe_vs_step0") is True
        and (equal or dominates)
        and not (equal and dominates)
        and admission.get("recomputed_from_authenticated_output_metrics") is True
        and admission.get("internal_numeric_matching_consulted") is False
        and isinstance(metrics, Mapping)
        and winner_metrics_sha == canonical_sha256(metrics)
        and winner.get("metrics_sha256") == winner_metrics_sha
        and semantic_contract.get(
            "tournament_pareto_admission_decision_sha256"
        )
        == decision_sha
        and decision_sha
        == canonical_sha256(
            {name: item for name, item in admission.items() if name != "decision_sha256"}
        )
    )


_BEHAVIOR_GATE_CHECKS = {
    "minimum_action_count",
    "minimum_action_accuracy",
    "minimum_rollout_count",
    "minimum_exact_rollout_fraction",
    "minimum_token_match_fraction",
    "minimum_lcp_fraction",
    "maximum_repeat_fraction_excess",
    "maximum_adjacent_repeat_fraction_excess",
    "maximum_longest_run_excess",
}


def _valid_winner_publication_behavior_decision(
    value: Any,
    *,
    expected_metrics: Mapping[str, Any] | None,
    expected_thresholds: Mapping[str, Any],
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "passed",
        "checks",
        "metrics",
        "thresholds",
        "selection_inputs",
        "internal_numeric_matching_consulted",
        "decision_sha256",
    }:
        return False
    checks = value.get("checks")
    metrics = value.get("metrics")
    thresholds = value.get("thresholds")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != _BEHAVIOR_GATE_CHECKS
        or not isinstance(metrics, Mapping)
        or not isinstance(thresholds, Mapping)
        or canonical_json(thresholds) != canonical_json(expected_thresholds)
        or (
            expected_metrics is not None
            and canonical_json(metrics) != canonical_json(expected_metrics)
        )
        or value.get("internal_numeric_matching_consulted") is not False
        or value.get("decision_sha256")
        != canonical_sha256(
            {name: item for name, item in value.items() if name != "decision_sha256"}
        )
    ):
        return False
    required_metric_names = {
        "action_count",
        "action_accuracy",
        "rollout_count",
        "exact_rollout_fraction",
        "token_match_fraction",
        "lcp_fraction",
    }
    threshold_names = set(_BEHAVIOR_GATE_CHECKS)
    if not required_metric_names.issubset(metrics) or set(thresholds) != threshold_names:
        return False
    repetition_required = (
        float(thresholds["maximum_repeat_fraction_excess"]) < 1.0
        or float(thresholds["maximum_adjacent_repeat_fraction_excess"]) < 1.0
        or int(thresholds["maximum_longest_run_excess"]) < 2_147_483_647
    )
    selection_inputs = set(required_metric_names)
    if repetition_required:
        selection_inputs.update(
            {
                "repeat_fraction_excess",
                "adjacent_repeat_fraction_excess",
                "longest_run_excess",
            }
        )
    if value.get("selection_inputs") != sorted(selection_inputs):
        return False
    try:
        expected_checks = {
            "minimum_action_count": int(metrics["action_count"])
            >= int(thresholds["minimum_action_count"]),
            "minimum_action_accuracy": float(metrics["action_accuracy"])
            >= float(thresholds["minimum_action_accuracy"]),
            "minimum_rollout_count": int(metrics["rollout_count"])
            >= int(thresholds["minimum_rollout_count"]),
            "minimum_exact_rollout_fraction": float(
                metrics["exact_rollout_fraction"]
            )
            >= float(thresholds["minimum_exact_rollout_fraction"]),
            "minimum_token_match_fraction": float(metrics["token_match_fraction"])
            >= float(thresholds["minimum_token_match_fraction"]),
            "minimum_lcp_fraction": float(metrics["lcp_fraction"])
            >= float(thresholds["minimum_lcp_fraction"]),
            "maximum_repeat_fraction_excess": float(
                metrics.get("repeat_fraction_excess", 0.0)
            )
            <= float(thresholds["maximum_repeat_fraction_excess"]),
            "maximum_adjacent_repeat_fraction_excess": float(
                metrics.get("adjacent_repeat_fraction_excess", 0.0)
            )
            <= float(thresholds["maximum_adjacent_repeat_fraction_excess"]),
            "maximum_longest_run_excess": int(
                metrics.get("longest_run_excess", 0)
            )
            <= int(thresholds["maximum_longest_run_excess"]),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        dict(checks) == expected_checks
        and all(expected_checks.values())
        and value.get("passed") is True
    )


def _winner_publication_source_thresholds(
    preregistration: Mapping[str, Any], preregistration_path: Path
) -> tuple[dict[str, Mapping[str, Any]], Path, str]:
    source_guided = preregistration.get("source_guided")
    if not isinstance(source_guided, Mapping):
        raise ExactCandidateFrontierError(
            "Winner publication source-guided binding is absent."
        )
    source_path = _resolve_binding(
        preregistration_path.parent,
        source_guided.get("preregistration"),
        "source_guided.preregistration",
    )
    _assert_ordinary_file(source_path, "source guided preregistration")
    source_sha = require_sha256(
        source_guided.get("preregistration_sha256"),
        "source_guided.preregistration_sha256",
    )
    source_payload = strict_json(source_path)
    thresholds = source_payload.get("thresholds") if isinstance(source_payload, Mapping) else None
    if (
        file_sha256(source_path) != source_sha
        or not isinstance(source_payload, Mapping)
        or source_payload.get("evaluation_contract_sha256")
        != source_guided.get("evaluation_contract_sha256")
        or not isinstance(thresholds, Mapping)
        or set(thresholds) != {"fit", "candidate", "sealed"}
        or any(not isinstance(thresholds.get(name), Mapping) for name in thresholds)
    ):
        raise ExactCandidateFrontierError(
            "Winner publication source-guided thresholds failed authentication."
        )
    return (
        {name: dict(thresholds[name]) for name in ("fit", "candidate", "sealed")},
        source_path,
        source_sha,
    )


def _bound_publication_file(
    *,
    base: Path,
    path_value: Any,
    sha_value: Any,
    name: str,
) -> tuple[Path, str]:
    path = _resolve_binding(base, path_value, name)
    _assert_ordinary_file(path, name)
    expected = require_sha256(sha_value, f"{name}.sha256")
    if file_sha256(path) != expected:
        raise ExactCandidateFrontierError(f"{name} bytes changed.")
    return path, expected


def _bound_publication_mapping(
    value: Any,
    *,
    base: Path,
    name: str,
) -> tuple[Path, str]:
    if not isinstance(value, Mapping):
        raise ExactCandidateFrontierError(f"{name} binding is absent.")
    return _bound_publication_file(
        base=base,
        path_value=value.get("path"),
        sha_value=value.get("sha256"),
        name=name,
    )


def _bound_publication_splits(
    value: Any,
    *,
    base: Path,
    name: str,
) -> dict[str, tuple[Path, str]]:
    if not isinstance(value, Mapping) or set(value) != {
        "fit",
        "candidate",
        "sealed",
    }:
        raise ExactCandidateFrontierError(f"{name} split bindings are incomplete.")
    return {
        split: _bound_publication_mapping(
            value[split],
            base=base,
            name=f"{name} {split} split",
        )
        for split in ("fit", "candidate", "sealed")
    }


def _authenticate_self_hashed_mapping(
    value: Any,
    *,
    schema: str,
    hash_field: str,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != schema:
        raise ExactCandidateFrontierError(f"{name} schema is absent or changed.")
    expected = require_sha256(value.get(hash_field), f"{name}.{hash_field}")
    preimage = {key: item for key, item in value.items() if key != hash_field}
    if canonical_sha256(preimage) != expected:
        raise ExactCandidateFrontierError(f"{name} self-hash changed.")
    return value


def _authenticated_sha_mapping(
    value: Any,
    *,
    expected_names: set[str],
    name: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != expected_names:
        raise ExactCandidateFrontierError(f"{name} SHA mapping is incomplete.")
    return {
        key: require_sha256(value[key], f"{name}.{key}")
        for key in sorted(expected_names)
    }


def _guided_gud_winner_publication_source_thresholds(
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
) -> tuple[
    dict[str, Mapping[str, Any]],
    str,
    dict[str, tuple[Path, str]],
    dict[str, Any],
]:
    """Authenticate the publication -> v4 -> v3 -> guided threshold chain."""

    semantic = preregistration.get("semantic_contract")
    tournament_row = preregistration.get("tournament")
    source_v3 = preregistration.get("source_v3")
    winner = preregistration.get("winner")
    if (
        not isinstance(semantic, Mapping)
        or not isinstance(tournament_row, Mapping)
        or not isinstance(source_v3, Mapping)
        or not isinstance(winner, Mapping)
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication source bindings are incomplete."
        )
    comparison = require_sha256(
        preregistration.get("semantic_contract_sha256"),
        "semantic_contract_sha256",
    )
    if canonical_sha256(semantic) != comparison:
        raise ExactCandidateFrontierError(
            "Guided-GUD publication semantic contract self-hash changed."
        )
    target_contract = require_sha256(
        semantic.get("guided_gud_tournament_evaluation_contract_sha256"),
        "semantic_contract.guided_gud_tournament_evaluation_contract_sha256",
    )
    source_v3_contract = require_sha256(
        semantic.get("source_v3_tournament_evaluation_contract_sha256"),
        "semantic_contract.source_v3_tournament_evaluation_contract_sha256",
    )
    if (
        semantic.get("winner_portable_v4_factor_state_sha256")
        != winner.get("portable_factor_state_sha256")
        or semantic.get("winner_candidate_metrics_sha256")
        != winner.get("metrics_sha256")
        or semantic.get("candidate_frontier_sha256")
        != (
            winner.get("candidate_frontier", {}).get("sha256")
            if isinstance(winner.get("candidate_frontier"), Mapping)
            else None
        )
        or semantic.get("splits") != preregistration.get("splits")
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication semantic winner/split binding changed."
        )

    v4_prereg_path, v4_prereg_sha = _bound_publication_file(
        base=preregistration_path.parent,
        path_value=tournament_row.get("preregistration"),
        sha_value=tournament_row.get("preregistration_sha256"),
        name="guided-GUD tournament preregistration",
    )
    v4_run_path, v4_run_sha = _bound_publication_file(
        base=preregistration_path.parent,
        path_value=tournament_row.get("run_report"),
        sha_value=tournament_row.get("run_report_sha256"),
        name="guided-GUD tournament run report",
    )
    v4_winner_path, v4_winner_sha = _bound_publication_file(
        base=preregistration_path.parent,
        path_value=tournament_row.get("winner_selection"),
        sha_value=tournament_row.get("winner_selection_sha256"),
        name="guided-GUD tournament winner selection",
    )
    v4_frontier_path, v4_frontier_sha = _bound_publication_mapping(
        tournament_row.get("candidate_frontier"),
        base=preregistration_path.parent,
        name="guided-GUD candidate frontier",
    )
    if tournament_row.get("candidate_frontier") != winner.get("candidate_frontier"):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication winner/frontier bindings disagree."
        )

    v4_preregistration = strict_json(v4_prereg_path)
    if (
        not isinstance(v4_preregistration, Mapping)
        or v4_preregistration.get("schema")
        != GUIDED_GUD_TOURNAMENT_PREREGISTRATION_SCHEMA
        or v4_preregistration.get("sealed_before_model_access") is not True
        or v4_preregistration.get("evaluation_contract_sha256") != target_contract
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD tournament preregistration chain failed authentication."
        )
    source_v3_winner = v4_preregistration.get("source_v3_winner")
    if not isinstance(source_v3_winner, Mapping):
        raise ExactCandidateFrontierError(
            "Guided-GUD tournament lost its source-v3 winner binding."
        )
    publication_splits = _bound_publication_splits(
        preregistration.get("splits"),
        base=preregistration_path.parent,
        name="guided-GUD publication",
    )
    v4_splits = _bound_publication_splits(
        v4_preregistration.get("splits"),
        base=v4_prereg_path.parent,
        name="guided-GUD tournament",
    )
    if publication_splits != v4_splits:
        raise ExactCandidateFrontierError(
            "Guided-GUD publication and tournament split roots disagree."
        )
    v4_basis_path, v4_basis_sha = _bound_publication_mapping(
        v4_preregistration.get("basis_bundle"),
        base=v4_prereg_path.parent,
        name="guided-GUD basis bundle",
    )
    v4_site_mask_path, v4_site_mask_sha = _bound_publication_mapping(
        v4_preregistration.get("site_mask"),
        base=v4_prereg_path.parent,
        name="guided-GUD site mask",
    )

    v3_prereg_path, v3_prereg_sha = _bound_publication_file(
        base=preregistration_path.parent,
        path_value=source_v3.get("tournament_preregistration"),
        sha_value=source_v3.get("tournament_preregistration_sha256"),
        name="source-v3 tournament preregistration",
    )
    v4_source_v3_path, v4_source_v3_sha = _bound_publication_mapping(
        source_v3_winner.get("tournament_preregistration"),
        base=v4_prereg_path.parent,
        name="guided-GUD source-v3 tournament preregistration",
    )
    if v4_source_v3_path != v3_prereg_path or v4_source_v3_sha != v3_prereg_sha:
        raise ExactCandidateFrontierError(
            "Guided-GUD publication and tournament source-v3 roots disagree."
        )
    source_v3_run_path, source_v3_run_sha = _bound_publication_file(
        base=preregistration_path.parent,
        path_value=source_v3.get("tournament_run_report"),
        sha_value=source_v3.get("tournament_run_report_sha256"),
        name="source-v3 tournament run report",
    )
    v4_source_v3_run_path, v4_source_v3_run_sha = _bound_publication_mapping(
        source_v3_winner.get("tournament_run_report"),
        base=v4_prereg_path.parent,
        name="guided-GUD source-v3 tournament run report",
    )
    source_v3_winner_path, source_v3_winner_sha = _bound_publication_file(
        base=preregistration_path.parent,
        path_value=source_v3.get("winner_selection"),
        sha_value=source_v3.get("winner_selection_sha256"),
        name="source-v3 winner selection",
    )
    v4_source_v3_winner_path, v4_source_v3_winner_sha = (
        _bound_publication_mapping(
            source_v3_winner.get("winner_selection"),
            base=v4_prereg_path.parent,
            name="guided-GUD source-v3 winner selection",
        )
    )
    source_v3_frontier_path, source_v3_frontier_sha = (
        _bound_publication_mapping(
            source_v3.get("candidate_frontier"),
            base=preregistration_path.parent,
            name="source-v3 candidate frontier",
        )
    )
    if (
        (v4_source_v3_run_path, v4_source_v3_run_sha)
        != (source_v3_run_path, source_v3_run_sha)
        or (v4_source_v3_winner_path, v4_source_v3_winner_sha)
        != (source_v3_winner_path, source_v3_winner_sha)
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication and tournament source-v3 evidence disagree."
        )
    source_v3_factor_path, source_v3_factor_sha = _bound_publication_file(
        base=preregistration_path.parent,
        path_value=source_v3.get("factor_pack"),
        sha_value=source_v3.get("factor_pack_sha256"),
        name="source-v3 factor pack",
    )
    expected_source_v3 = source_v3_winner.get("expected_winner")
    if (
        not isinstance(expected_source_v3, Mapping)
        or source_v3.get("runtime_factor_state_sha256")
        != expected_source_v3.get("runtime_factor_state_sha256")
        or source_v3.get("portable_factor_state_sha256")
        != expected_source_v3.get("portable_factor_state_sha256")
        or source_v3_factor_sha
        != expected_source_v3.get("factor_pack_file_sha256")
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD source-v3 factor identities disagree."
        )
    require_sha256(
        source_v3.get("runtime_factor_state_sha256"),
        "source_v3.runtime_factor_state_sha256",
    )
    require_sha256(
        source_v3.get("portable_factor_state_sha256"),
        "source_v3.portable_factor_state_sha256",
    )
    if (
        semantic.get("source_v3_run_report_sha256") != source_v3_run_sha
        or semantic.get("source_v3_winner_selection_sha256")
        != source_v3_winner_sha
        or semantic.get("source_v3_candidate_frontier_sha256")
        != source_v3_frontier_sha
        or semantic.get("source_v3_factor_pack_file_sha256")
        != source_v3_factor_sha
        or semantic.get("winner_basis_bundle_file_sha256") != v4_basis_sha
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication transitive source semantics changed."
        )

    v3_preregistration = strict_json(v3_prereg_path)
    if (
        not isinstance(v3_preregistration, Mapping)
        or v3_preregistration.get("schema")
        != GUIDED_CANDIDATE_TOURNAMENT_PREREGISTRATION_SCHEMA
        or v3_preregistration.get("sealed_before_model_access") is not True
        or v3_preregistration.get("evaluation_contract_sha256")
        != source_v3_contract
    ):
        raise ExactCandidateFrontierError(
            "Source-v3 tournament preregistration chain failed authentication."
        )
    v3_splits = _bound_publication_splits(
        v3_preregistration.get("splits"),
        base=v3_prereg_path.parent,
        name="source-v3 tournament",
    )
    v3_site_mask_path, v3_site_mask_sha = _bound_publication_mapping(
        v3_preregistration.get("site_mask"),
        base=v3_prereg_path.parent,
        name="source-v3 site mask",
    )
    guided_prereg_path, guided_prereg_sha = _bound_publication_mapping(
        v3_preregistration.get("source_guided"),
        base=v3_prereg_path.parent,
        name="source guided preregistration",
    )
    guided_preregistration = strict_json(guided_prereg_path)
    thresholds = (
        guided_preregistration.get("thresholds")
        if isinstance(guided_preregistration, Mapping)
        else None
    )
    if (
        not isinstance(guided_preregistration, Mapping)
        or guided_preregistration.get("schema") != GUIDED_PREREGISTRATION_SCHEMA
        or not isinstance(
            guided_preregistration.get("evaluation_contract_sha256"), str
        )
        or require_sha256(
            guided_preregistration.get("evaluation_contract_sha256"),
            "source_guided.evaluation_contract_sha256",
        )
        != guided_preregistration.get("evaluation_contract_sha256")
        or not isinstance(thresholds, Mapping)
        or set(thresholds) != {"fit", "candidate", "sealed"}
        or any(not isinstance(thresholds.get(name), Mapping) for name in thresholds)
    ):
        raise ExactCandidateFrontierError(
            "Source guided behavior thresholds failed authentication."
        )
    guided_source = guided_preregistration.get("source")
    guided_audit = guided_preregistration.get("audit")
    if not isinstance(guided_source, Mapping) or not isinstance(
        guided_audit, Mapping
    ):
        raise ExactCandidateFrontierError(
            "Source guided provenance roots are incomplete."
        )
    guided_splits = _bound_publication_splits(
        guided_preregistration.get("splits"),
        base=guided_prereg_path.parent,
        name="source guided continuation",
    )
    guided_source_run_path, guided_source_run_sha = _bound_publication_mapping(
        guided_source.get("run_report"),
        base=guided_prereg_path.parent,
        name="source guided input run report",
    )
    guided_source_factor_path, guided_source_factor_sha = (
        _bound_publication_mapping(
            guided_source.get("factor_pack"),
            base=guided_prereg_path.parent,
            name="source guided input factor pack",
        )
    )
    guided_audit_prereg_path, guided_audit_prereg_sha = (
        _bound_publication_mapping(
            guided_audit.get("preregistration"),
            base=guided_prereg_path.parent,
            name="source guided audit preregistration",
        )
    )
    guided_audit_run_path, guided_audit_run_sha = _bound_publication_mapping(
        guided_audit.get("run_report"),
        base=guided_prereg_path.parent,
        name="source guided audit run report",
    )
    guided_basis_path, guided_basis_sha = _bound_publication_mapping(
        guided_audit.get("guided_basis_pack"),
        base=guided_prereg_path.parent,
        name="source guided basis pack",
    )

    source_v2_factor_state_sha = require_sha256(
        guided_source.get("factor_state_sha256"),
        "source_guided.source.factor_state_sha256",
    )
    compiler_report_sha = require_sha256(
        guided_source.get("compiler_report_sha256"),
        "source_guided.source.compiler_report_sha256",
    )
    guided_qkvo_audit_sha = require_sha256(
        guided_audit.get("audit_report_self_sha256"),
        "source_guided.audit.audit_report_self_sha256",
    )
    guided_qkvo_basis_file_sha = require_sha256(
        guided_audit.get("basis_envelope_file_sha256"),
        "source_guided.audit.basis_envelope_file_sha256",
    )
    guided_qkvo_payload_sha = require_sha256(
        guided_audit.get("basis_payload_sha256"),
        "source_guided.audit.basis_payload_sha256",
    )
    guided_qkvo_manifest_sha = require_sha256(
        guided_audit.get("basis_manifest_sha256"),
        "source_guided.audit.basis_manifest_sha256",
    )
    guided_qkvo_tensor_shas = _authenticated_sha_mapping(
        guided_audit.get("basis_tensor_sha256s"),
        expected_names={"q", "k", "v", "o"},
        name="source_guided.audit.basis_tensor_sha256s",
    )
    guided_gud_audit_sha = require_sha256(
        semantic.get("winner_basis_audit_report_sha256"),
        "semantic_contract.winner_basis_audit_report_sha256",
    )
    guided_gud_payload_sha = require_sha256(
        semantic.get("winner_basis_payload_sha256"),
        "semantic_contract.winner_basis_payload_sha256",
    )
    guided_gud_manifest_sha = require_sha256(
        semantic.get("winner_basis_manifest_sha256"),
        "semantic_contract.winner_basis_manifest_sha256",
    )
    guided_gud_tensor_shas = _authenticated_sha_mapping(
        semantic.get("winner_source_basis_tensor_sha256s"),
        expected_names={"gate", "up", "down"},
        name="semantic_contract.winner_source_basis_tensor_sha256s",
    )
    if guided_basis_sha != guided_qkvo_basis_file_sha:
        raise ExactCandidateFrontierError(
            "Source guided basis file provenance changed."
        )

    winner_factor_path, winner_factor_sha = _bound_publication_file(
        base=preregistration_path.parent,
        path_value=winner.get("factor_pack"),
        sha_value=winner.get("factor_pack_sha256"),
        name="guided-GUD winner factor pack",
    )
    protected = {
        "v4_tournament_preregistration": (v4_prereg_path, v4_prereg_sha),
        "v4_tournament_run_report": (v4_run_path, v4_run_sha),
        "v4_tournament_winner": (v4_winner_path, v4_winner_sha),
        "v4_candidate_frontier": (v4_frontier_path, v4_frontier_sha),
        "v4_winner_factor_pack": (winner_factor_path, winner_factor_sha),
        "source_v3_tournament_preregistration": (v3_prereg_path, v3_prereg_sha),
        "source_v3_tournament_run_report": (source_v3_run_path, source_v3_run_sha),
        "source_v3_winner_selection": (
            source_v3_winner_path,
            source_v3_winner_sha,
        ),
        "source_v3_candidate_frontier": (
            source_v3_frontier_path,
            source_v3_frontier_sha,
        ),
        "source_v3_factor_pack": (source_v3_factor_path, source_v3_factor_sha),
        "source_v3_site_mask": (v3_site_mask_path, v3_site_mask_sha),
        "source_guided_preregistration": (guided_prereg_path, guided_prereg_sha),
        "source_guided_input_run_report": (
            guided_source_run_path,
            guided_source_run_sha,
        ),
        "source_guided_input_factor_pack": (
            guided_source_factor_path,
            guided_source_factor_sha,
        ),
        "source_guided_audit_preregistration": (
            guided_audit_prereg_path,
            guided_audit_prereg_sha,
        ),
        "source_guided_audit_run_report": (
            guided_audit_run_path,
            guided_audit_run_sha,
        ),
        "source_guided_basis_pack": (guided_basis_path, guided_basis_sha),
        "v4_basis_bundle": (v4_basis_path, v4_basis_sha),
        "v4_site_mask": (v4_site_mask_path, v4_site_mask_sha),
    }
    for split, binding in v4_splits.items():
        protected[f"v4_split_{split}"] = binding
    for split, binding in v3_splits.items():
        protected[f"source_v3_split_{split}"] = binding
    for split, binding in guided_splits.items():
        protected[f"source_guided_split_{split}"] = binding
    provenance = {
        "compiler_report_sha256": compiler_report_sha,
        "source_v2_factor_state_sha256": source_v2_factor_state_sha,
        "guided_qkvo": {
            "audit_report_sha256": guided_qkvo_audit_sha,
            "basis_envelope_file_sha256": guided_qkvo_basis_file_sha,
            "basis_payload_sha256": guided_qkvo_payload_sha,
            "basis_manifest_sha256": guided_qkvo_manifest_sha,
            "source_basis_tensor_sha256s": guided_qkvo_tensor_shas,
        },
        "guided_gud": {
            "audit_report_sha256": guided_gud_audit_sha,
            "basis_envelope_file_sha256": v4_basis_sha,
            "basis_payload_sha256": guided_gud_payload_sha,
            "basis_manifest_sha256": guided_gud_manifest_sha,
            "source_basis_tensor_sha256s": guided_gud_tensor_shas,
        },
    }
    return (
        {name: dict(thresholds[name]) for name in ("fit", "candidate", "sealed")},
        target_contract,
        protected,
        provenance,
    )


def _authenticate_guided_gud_publication_provenance(
    report: Mapping[str, Any],
    *,
    comparison: str,
    provenance: Mapping[str, Any],
) -> None:
    """Authenticate the compiler-to-dual-bake receipt chain for v4 only."""

    compiler_sha = require_sha256(
        provenance.get("compiler_report_sha256"),
        "publication_provenance.compiler_report_sha256",
    )
    compiler_report = report.get("compiler_report")
    if (
        not isinstance(compiler_report, Mapping)
        or canonical_sha256(compiler_report) != compiler_sha
        or compiler_report.get("bootstrap_accepted") is not True
        or compiler_report.get("target_budget_exact") is not True
        or compiler_report.get("target_unique_parameter_count")
        != QWEN3_1P4B_EXACT_PARAMETERS
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD compiler report provenance is absent or changed."
        )

    inherited = _authenticate_self_hashed_mapping(
        report.get("inherited_installation"),
        schema=EXACT_TARGET_COMPILER_TRANSACTION_INSTALLATION_SCHEMA,
        hash_field="installation_transaction_sha256",
        name="guided-GUD inherited installation",
    )
    dense_qkvo = _authenticate_self_hashed_mapping(
        inherited.get("dense_qkvo_installation"),
        schema=DENSE_QKVO_INSTALLATION_SCHEMA,
        hash_field="installation_sha256",
        name="guided-GUD inherited QKVO installation",
    )
    dense_gud = _authenticate_self_hashed_mapping(
        inherited.get("dense_gate_up_down_installation"),
        schema=DENSE_GUD_INSTALLATION_SCHEMA,
        hash_field="installation_sha256",
        name="guided-GUD inherited GUD installation",
    )
    lexical_parameters = _nonnegative_int(
        inherited.get("temporary_lexical_factor_parameters"),
        "inherited_installation.temporary_lexical_factor_parameters",
    )
    if (
        inherited.get("exact_native_parameters")
        != QWEN3_1P4B_EXACT_PARAMETERS
        or inherited.get("published_parameters_if_promoted")
        != QWEN3_1P4B_EXACT_PARAMETERS
        or inherited.get("zero_step_identity") is not True
        or inherited.get("parent_compiler_report_sha256") != compiler_sha
        or lexical_parameters <= 0
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD inherited installation provenance is invalid."
        )

    source_v2_sha = require_sha256(
        provenance.get("source_v2_factor_state_sha256"),
        "publication_provenance.source_v2_factor_state_sha256",
    )
    source_restore = report.get("source_v2_restore")
    if (
        not isinstance(source_restore, Mapping)
        or source_restore.get("schema") != SOURCE_V2_FACTOR_SCHEMA
        or source_restore.get("factor_state_sha256") != source_v2_sha
        or source_restore.get("factor_count") != SOURCE_V2_FACTOR_COUNT
        or source_restore.get(
            "temporary_restore_clone_released_before_guided_install"
        )
        is not True
        or source_restore.get("full_tensor_or_model_clone_retained") is not False
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD source-v2 restore provenance is absent or changed."
        )

    qkvo_expected = provenance.get("guided_qkvo")
    gud_expected = provenance.get("guided_gud")
    if not isinstance(qkvo_expected, Mapping) or not isinstance(
        gud_expected, Mapping
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD guided source provenance is incomplete."
        )
    qkvo_install = _authenticate_self_hashed_mapping(
        report.get("guided_qkvo_installation"),
        schema=GUIDED_QKVO_INSTALLATION_SCHEMA,
        hash_field="installation_sha256",
        name="guided QKVO installation",
    )
    gud_install = _authenticate_self_hashed_mapping(
        report.get("guided_gud_installation"),
        schema=GUIDED_GUD_INSTALLATION_SCHEMA,
        hash_field="installation_sha256",
        name="guided GUD installation",
    )
    provenance_fields = (
        "audit_report_sha256",
        "basis_envelope_file_sha256",
        "basis_payload_sha256",
        "basis_manifest_sha256",
        "source_basis_tensor_sha256s",
    )
    if (
        qkvo_install.get("guided_adapter_parameter_count")
        != GUIDED_QKVO_FACTOR_PARAMETERS
        or qkvo_install.get("inherited_installation_sha256")
        != dense_qkvo.get("installation_sha256")
        or qkvo_install.get("factor_concatenation_used") is not False
        or qkvo_install.get("factor_regrouping_used") is not False
        or qkvo_install.get("zero_guided_left_preserves_inherited_function_exactly")
        is not True
        or any(qkvo_install.get(name) != qkvo_expected.get(name) for name in provenance_fields)
        or gud_install.get("guided_adapter_parameter_count")
        != GUIDED_GUD_FACTOR_PARAMETERS
        or gud_install.get("inherited_installation_sha256")
        != dense_gud.get("installation_sha256")
        or gud_install.get("factor_concatenation_used") is not False
        or gud_install.get("factor_regrouping_used") is not False
        or gud_install.get("zero_guided_left_preserves_inherited_function_exactly")
        is not True
        or any(gud_install.get(name) != gud_expected.get(name) for name in provenance_fields)
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD guided installation provenance is invalid."
        )

    publication = _authenticate_self_hashed_mapping(
        report.get("publication"),
        schema=EXACT_TARGET_COMPILER_TRANSACTION_PUBLICATION_SCHEMA,
        hash_field="publication_transaction_sha256",
        name="guided-GUD publication transaction",
    )
    external_gate = _authenticate_self_hashed_mapping(
        publication.get("external_behavior_gate"),
        schema=EXTERNAL_BEHAVIOR_GATE_SCHEMA,
        hash_field="receipt_sha256",
        name="guided-GUD external behavior gate",
    )
    qkvo_bake = _authenticate_self_hashed_mapping(
        publication.get("dense_qkvo_bake"),
        schema=GUIDED_QKVO_BAKE_SCHEMA,
        hash_field="bake_sha256",
        name="guided QKVO publication bake",
    )
    gud_bake = _authenticate_self_hashed_mapping(
        publication.get("dense_gate_up_down_bake"),
        schema=GUIDED_GUD_BAKE_SCHEMA,
        hash_field="bake_sha256",
        name="guided GUD publication bake",
    )
    lexical_bake = _authenticate_self_hashed_mapping(
        publication.get("dense_lexical_bake"),
        schema=LEXICAL_BAKE_SCHEMA,
        hash_field="bake_sha256",
        name="guided-GUD lexical publication bake",
    )
    qkvo_inherited = _nonnegative_int(
        qkvo_install.get("inherited_adapter_parameter_count"),
        "guided_qkvo_installation.inherited_adapter_parameter_count",
    )
    gud_inherited = _nonnegative_int(
        gud_install.get("inherited_adapter_parameter_count"),
        "guided_gud_installation.inherited_adapter_parameter_count",
    )
    if (
        qkvo_install.get("adapter_parameter_count")
        != qkvo_inherited + GUIDED_QKVO_FACTOR_PARAMETERS
        or gud_install.get("adapter_parameter_count")
        != gud_inherited + GUIDED_GUD_FACTOR_PARAMETERS
        or qkvo_bake.get("inherited_adapter_parameters_removed") != qkvo_inherited
        or qkvo_bake.get("guided_adapter_parameters_removed")
        != GUIDED_QKVO_FACTOR_PARAMETERS
        or qkvo_bake.get("adapter_parameters_removed")
        != qkvo_install.get("adapter_parameter_count")
        or gud_bake.get("inherited_adapter_parameters_removed") != gud_inherited
        or gud_bake.get("guided_adapter_parameters_removed")
        != GUIDED_GUD_FACTOR_PARAMETERS
        or gud_bake.get("adapter_parameters_removed")
        != gud_install.get("adapter_parameter_count")
        or lexical_bake.get("adapter_parameters_removed") != lexical_parameters
        or gud_bake.get("model_parameter_count_after_bake")
        != QWEN3_1P4B_EXACT_PARAMETERS
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD dual-bake parameter accounting is invalid."
        )
    bake_flags = (
        qkvo_bake.get("factor_concatenation_used") is False
        and qkvo_bake.get("factor_regrouping_used") is False
        and qkvo_bake.get("inherited_registry_removed") is True
        and qkvo_bake.get("guided_registry_removed") is True
        and qkvo_bake.get("projection_wrappers_removed") is True
        and gud_bake.get("factor_concatenation_used") is False
        and gud_bake.get("factor_regrouping_used") is False
        and gud_bake.get("inherited_registry_removed") is True
        and gud_bake.get("guided_registry_removed") is True
        and gud_bake.get("projection_wrappers_removed") is True
    )
    evidence_sha = require_sha256(
        report.get("external_behavior_gate_evidence_sha256"),
        "external_behavior_gate_evidence_sha256",
    )
    if (
        publication.get("exact_unique_parameters")
        != QWEN3_1P4B_EXACT_PARAMETERS
        or publication.get("exact_budget_match") is not True
        or publication.get("parent_compiler_report_sha256") != compiler_sha
        or publication.get("published_adapter_parameter_count") != 0
        or publication.get("published_parent_runtime_reference_count") != 0
        or publication.get("published_source_sized_sidecar_parameter_count") != 0
        or publication.get("nested_publication_bake_order")
        != [
            "lexical",
            "qkvo-inherited-then-guided",
            "gate-up-down-inherited-then-guided",
        ]
        or publication.get("guided_registry_suspended_for_lexical_bake") is not True
        or publication.get(
            "guided_gate_up_down_registry_suspended_for_lexical_bake"
        )
        is not True
        or publication.get("guided_gate_up_down_registry_suspended_for_qkvo_bake")
        is not True
        or external_gate.get("evidence_sha256") != evidence_sha
        or external_gate.get("evaluation_contract_sha256") != comparison
        or external_gate.get("passed") is not True
        or external_gate.get("internal_similarity_only") is not False
        or not bake_flags
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication transaction provenance is invalid."
        )


def _valid_retained_winner_publication_behavior(
    report: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
) -> bool:
    try:
        thresholds, _source_path, _source_sha = (
            _winner_publication_source_thresholds(
                preregistration,
                preregistration_path,
            )
        )
        winner = preregistration.get("winner")
        pre = report.get("pre_bake_behavior_decisions")
        post = report.get("post_bake_behavior_decisions")
        if (
            not isinstance(winner, Mapping)
            or not isinstance(winner.get("metrics"), Mapping)
            or not _valid_winner_publication_pareto_admission(
                winner,
                preregistration.get("semantic_contract"),
            )
            or not isinstance(pre, Mapping)
            or not isinstance(post, Mapping)
            or not isinstance(pre.get("candidate"), Mapping)
            or not isinstance(pre.get("sealed"), Mapping)
            or not isinstance(post.get("fit"), Mapping)
            or not isinstance(post.get("candidate"), Mapping)
        ):
            return False
        step = _nonnegative_int(winner.get("step"), "winner.step")
        return bool(
            step > 0
            and pre.get("fit") is None
            and post.get("sealed") is None
            and report.get("pre_sealed_candidate_decision") == pre["candidate"]
            and _valid_winner_publication_behavior_decision(
                pre["candidate"],
                expected_metrics=winner["metrics"],
                expected_thresholds=thresholds["candidate"],
            )
            and _valid_winner_publication_behavior_decision(
                pre["sealed"],
                expected_metrics=None,
                expected_thresholds=thresholds["sealed"],
            )
            and _valid_winner_publication_behavior_decision(
                post["fit"],
                expected_metrics=None,
                expected_thresholds=thresholds["fit"],
            )
            and _valid_winner_publication_behavior_decision(
                post["candidate"],
                expected_metrics=None,
                expected_thresholds=thresholds["candidate"],
            )
            and _valid_winner_publication_method_decision(
                report.get("guided_method_decision"),
                winner_step=step,
                candidate_metrics=pre["candidate"]["metrics"],
            )
            and _valid_winner_publication_method_decision(
                report.get("post_bake_guided_method_decision"),
                winner_step=step,
                candidate_metrics=post["candidate"]["metrics"],
            )
        )
    except (ExactCandidateFrontierError, KeyError, TypeError, ValueError):
        return False


def _valid_retained_guided_gud_winner_publication_behavior(
    report: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
) -> bool:
    try:
        thresholds, _target_contract, _protected, source_provenance = (
            _guided_gud_winner_publication_source_thresholds(
                preregistration,
                preregistration_path,
            )
        )
        winner = preregistration.get("winner")
        pre = report.get("pre_bake_behavior_decisions")
        post = report.get("post_bake_behavior_decisions")
        sealed = report.get("sealed_evaluation")
        if (
            not isinstance(winner, Mapping)
            or not isinstance(winner.get("metrics"), Mapping)
            or not _valid_winner_publication_pareto_admission(
                winner,
                preregistration.get("semantic_contract"),
            )
            or not isinstance(pre, Mapping)
            or not isinstance(post, Mapping)
            or not isinstance(pre.get("candidate"), Mapping)
            or not isinstance(pre.get("sealed"), Mapping)
            or not isinstance(post.get("fit"), Mapping)
            or not isinstance(post.get("candidate"), Mapping)
            or not isinstance(sealed, Mapping)
        ):
            return False
        step = _nonnegative_int(winner.get("step"), "winner.step")
        comparison = require_sha256(
            preregistration.get("semantic_contract_sha256"),
            "semantic_contract_sha256",
        )
        winner_portable = require_sha256(
            winner.get("portable_factor_state_sha256"),
            "winner.portable_factor_state_sha256",
        )
        _authenticate_guided_gud_publication_provenance(
            report,
            comparison=comparison,
            provenance=source_provenance,
        )
        sealed_binding = preregistration.get("sealed_consumption")
        marker_receipt = sealed.get("marker_receipt")
        split_rows = preregistration.get("splits")
        sealed_split = (
            split_rows.get("sealed") if isinstance(split_rows, Mapping) else None
        )
        if (
            not isinstance(sealed_binding, Mapping)
            or sealed_binding.get("schema")
            != GUIDED_GUD_SEALED_CONSUMPTION_SCHEMA
            or not isinstance(marker_receipt, Mapping)
            or not isinstance(sealed_split, Mapping)
        ):
            return False
        marker_path = _resolve_binding(
            preregistration_path.parent,
            marker_receipt.get("marker"),
            "retained sealed marker",
        )
        declared_marker = _resolve_binding(
            preregistration_path.parent,
            sealed_binding.get("marker"),
            "retained declared sealed marker",
        )
        if (
            marker_path != declared_marker
            or marker_path.name != "sealed_consumption.json"
        ):
            return False
        _assert_ordinary_file(marker_path, "retained guided-GUD sealed marker")
        marker_sha = require_sha256(
            marker_receipt.get("marker_file_sha256"),
            "retained marker_file_sha256",
        )
        marker_expected = {
            "schema": GUIDED_GUD_SEALED_CONSUMPTION_SCHEMA,
            "status": "consumed-before-forward",
            "preregistration_sha256": file_sha256(preregistration_path),
            "semantic_contract_sha256": comparison,
            "winner_portable_v4_factor_state_sha256": winner_portable,
            "sealed_split_sha256": sealed_split.get("sha256"),
            "sealed_forward_limit": 1,
            "output_root": report.get("output_root"),
        }
        marker_payload = strict_json(marker_path)
        return bool(
            step > 0
            and report.get("status") == "accepted"
            and report.get("standalone_compaction_passed") is True
            and report.get("exact_unique_parameters")
            == QWEN3_1P4B_EXACT_PARAMETERS
            and report.get("published_adapter_parameter_count") == 0
            and report.get("published_parent_runtime_reference_count") == 0
            and report.get("published_source_sized_sidecar_parameter_count") == 0
            and pre.get("fit") is None
            and post.get("sealed") is None
            and report.get("sealed_forward_count") == 1
            and report.get("post_bake_sealed_forward_count") == 0
            and report.get("sealed_used_for_reranking") is False
            and sealed.get("forward_count") == 1
            and sealed.get("one_shot_marker_consumed") is True
            and sealed.get("post_bake_sealed_forward_count") == 0
            and sealed.get("acceptance_only") is True
            and sealed.get("reranking_forbidden") is True
            and marker_receipt.get("exclusive_create_succeeded") is True
            and marker_receipt.get("sealed_forward_limit") == 1
            and marker_receipt.get("preregistration_sha256")
            == file_sha256(preregistration_path)
            and marker_receipt.get("semantic_contract_sha256") == comparison
            and marker_receipt.get("winner_portable_v4_factor_state_sha256")
            == winner_portable
            and marker_receipt.get("sealed_split_sha256")
            == sealed_split.get("sha256")
            and marker_receipt.get("output_root") == report.get("output_root")
            and file_sha256(marker_path) == marker_sha
            and marker_payload == marker_expected
            and all(
                marker_receipt.get(name) == value
                for name, value in marker_expected.items()
            )
            and report.get("pre_sealed_candidate_decision") == pre["candidate"]
            and _valid_winner_publication_behavior_decision(
                pre["candidate"],
                expected_metrics=winner["metrics"],
                expected_thresholds=thresholds["candidate"],
            )
            and _valid_winner_publication_behavior_decision(
                pre["sealed"],
                expected_metrics=None,
                expected_thresholds=thresholds["sealed"],
            )
            and _valid_winner_publication_behavior_decision(
                post["fit"],
                expected_metrics=None,
                expected_thresholds=thresholds["fit"],
            )
            and _valid_winner_publication_behavior_decision(
                post["candidate"],
                expected_metrics=None,
                expected_thresholds=thresholds["candidate"],
            )
            and _valid_winner_publication_method_decision(
                report.get("guided_candidate_method_decision"),
                winner_step=step,
                candidate_metrics=pre["candidate"]["metrics"],
            )
            and _valid_winner_publication_method_decision(
                report.get("post_bake_guided_candidate_method_decision"),
                winner_step=step,
                candidate_metrics=post["candidate"]["metrics"],
            )
        )
    except (ExactCandidateFrontierError, KeyError, TypeError, ValueError):
        return False


def _authenticate_guided_winner_publication_candidate(
    root: Path,
    report_path: Path,
    report: Mapping[str, Any],
    report_sha: str,
    *,
    expected_preregistration_sha256: str,
    preregistration_authenticator: PreregistrationAuthenticator,
    accepted_child_authenticator: AcceptedChildAuthenticator | None,
) -> TerminalCandidate:
    """Authenticate the one-shot-sealed winner schema without weakening legacy."""

    if (
        report.get("status") != "accepted"
        or report.get("standalone_compaction_passed") is not True
        or report.get("model_checkpoint_written") is not True
        or report.get("rollback_exact") is not None
        or report.get("sealed_forward_count") != 1
        or report.get("sealed_marker_consumed") is not True
        or report.get("post_bake_sealed_forward_count") != 0
    ):
        raise ExactCandidateFrontierError(
            "Winner publication lacks terminal one-shot standalone evidence."
        )
    prereg_path = _resolve_binding(
        report_path.parent,
        report.get("preregistration"),
        "run_report.preregistration",
    )
    _assert_ordinary_file(prereg_path, "winner publication preregistration")
    prereg_sha = file_sha256(prereg_path)
    if (
        prereg_sha
        != require_sha256(
            expected_preregistration_sha256,
            "expected_preregistration_sha256",
        )
        or prereg_sha
        != require_sha256(
            report.get("preregistration_sha256"),
            "preregistration_sha256",
        )
    ):
        raise ExactCandidateFrontierError(
            "Winner publication preregistration external roots disagree."
        )
    preregistration = strict_json(prereg_path)
    if (
        not isinstance(preregistration, Mapping)
        or preregistration.get("schema")
        != GUIDED_WINNER_PUBLICATION_PREREGISTRATION_SCHEMA
        or preregistration.get("model_free_preregistration") is not True
        or preregistration.get("publication_authorized_before_sealed_pass")
        is not False
    ):
        raise ExactCandidateFrontierError(
            "Winner publication preregistration is not model-free/sealed."
        )
    authenticated = preregistration_authenticator(prereg_path, prereg_sha)
    if not isinstance(authenticated, Mapping):
        raise ExactCandidateFrontierError(
            "Winner publication preregistration authenticator returned no mapping."
        )
    comparison = require_sha256(
        report.get("semantic_contract_sha256"),
        "semantic_contract_sha256",
    )
    if (
        preregistration.get("semantic_contract_sha256") != comparison
        or authenticated.get("semantic_contract_sha256") != comparison
    ):
        raise ExactCandidateFrontierError(
            "Winner publication semantic contract failed authentication."
        )
    winner = preregistration.get("winner")
    if not isinstance(winner, Mapping):
        raise ExactCandidateFrontierError("Winner publication lost winner provenance.")
    selected_step = _nonnegative_int(winner.get("step"), "winner.step")
    if selected_step <= 0:
        raise ExactCandidateFrontierError(
            "Winner publication may not promote a step-zero baseline."
        )
    winner_metrics = winner.get("metrics")
    if not isinstance(winner_metrics, Mapping):
        raise ExactCandidateFrontierError(
            "Winner publication preregistration lost candidate metrics."
        )
    if not _valid_winner_publication_pareto_admission(
        winner,
        preregistration.get("semantic_contract"),
    ):
        raise ExactCandidateFrontierError(
            "Winner publication Pareto admission is unauthenticated."
        )
    thresholds, source_guided_path, source_guided_sha = (
        _winner_publication_source_thresholds(preregistration, prereg_path)
    )
    selected_sha = require_sha256(
        report.get("runtime_rebound_factor_state_sha256"),
        "runtime_rebound_factor_state_sha256",
    )
    if (
        report.get("winner_portable_factor_state_sha256")
        != winner.get("portable_factor_state_sha256")
    ):
        raise ExactCandidateFrontierError(
            "Winner publication factor provenance is invalid."
        )
    pre = report.get("pre_bake_behavior_decisions")
    post = report.get("post_bake_behavior_decisions")
    sealed = report.get("sealed_evaluation")
    pre_method = report.get("guided_method_decision")
    post_method = report.get("post_bake_guided_method_decision")
    if (
        not isinstance(pre, Mapping)
        or pre.get("fit") is not None
        or not isinstance(pre.get("candidate"), Mapping)
        or not isinstance(pre.get("sealed"), Mapping)
        or not isinstance(post, Mapping)
        or not isinstance(post.get("fit"), Mapping)
        or not isinstance(post.get("candidate"), Mapping)
        or post.get("sealed") is not None
        or not _valid_winner_publication_behavior_decision(
            pre["candidate"],
            expected_metrics=winner_metrics,
            expected_thresholds=thresholds["candidate"],
        )
        or report.get("pre_sealed_candidate_decision") != pre["candidate"]
        or not _valid_winner_publication_behavior_decision(
            pre["sealed"],
            expected_metrics=None,
            expected_thresholds=thresholds["sealed"],
        )
        or not _valid_winner_publication_behavior_decision(
            post["fit"],
            expected_metrics=None,
            expected_thresholds=thresholds["fit"],
        )
        or not _valid_winner_publication_behavior_decision(
            post["candidate"],
            expected_metrics=None,
            expected_thresholds=thresholds["candidate"],
        )
        or not _valid_winner_publication_method_decision(
            pre_method,
            winner_step=selected_step,
            candidate_metrics=pre["candidate"]["metrics"],
        )
        or not _valid_winner_publication_method_decision(
            post_method,
            winner_step=selected_step,
            candidate_metrics=post["candidate"]["metrics"],
        )
        or not isinstance(sealed, Mapping)
        or sealed.get("forward_count") != 1
        or sealed.get("one_shot_marker_consumed") is not True
        or sealed.get("post_bake_sealed_forward_count") != 0
    ):
        raise ExactCandidateFrontierError(
            "Winner publication behavior gates violate one-shot sealed policy."
        )
    marker_receipt = sealed.get("marker_receipt")
    if (
        not isinstance(marker_receipt, Mapping)
        or marker_receipt.get("exclusive_create_succeeded") is not True
        or marker_receipt.get("sealed_forward_limit") != 1
        or marker_receipt.get("preregistration_sha256") != prereg_sha
        or marker_receipt.get("semantic_contract_sha256") != comparison
    ):
        raise ExactCandidateFrontierError(
            "Winner publication sealed marker receipt is unauthenticated."
        )
    marker_path = _resolve_binding(
        report_path.parent,
        marker_receipt.get("marker"),
        "sealed_evaluation.marker_receipt.marker",
    )
    if marker_path.parent != prereg_path.parent:
        raise ExactCandidateFrontierError(
            "Winner publication sealed marker escaped its preregistration root."
        )
    _assert_ordinary_file(marker_path, "winner publication sealed marker")
    marker_sha = require_sha256(
        marker_receipt.get("marker_file_sha256"),
        "sealed_evaluation.marker_receipt.marker_file_sha256",
    )
    marker_payload = strict_json(marker_path)
    if (
        file_sha256(marker_path) != marker_sha
        or not isinstance(marker_payload, Mapping)
        or marker_payload.get("schema")
        != "neural-forge-guided-sealed-consumption-v1"
        or marker_payload.get("status") != "consumed-before-forward"
        or marker_payload.get("preregistration_sha256") != prereg_sha
        or marker_payload.get("semantic_contract_sha256") != comparison
        or marker_payload.get("sealed_forward_limit") != 1
        or any(
            marker_payload.get(name) != marker_receipt.get(name)
            for name in marker_payload
        )
    ):
        raise ExactCandidateFrontierError(
            "Winner publication sealed marker bytes no longer authenticate."
        )
    post_metrics = post["candidate"].get("metrics")
    if not isinstance(post_metrics, Mapping):
        raise ExactCandidateFrontierError(
            "Winner publication post-bake candidate metrics are absent."
        )
    reload_gate = report.get("fresh_reload_guided_method_gate")
    if (
        not isinstance(reload_gate, Mapping)
        or reload_gate.get("passed") is not True
        or reload_gate.get("post_bake_decision_sha256")
        != post_method.get("decision_sha256")
        or reload_gate.get("state_bytes_match_pre_save") is not True
        or reload_gate.get("candidate_behavior_certified_before_save") is not True
        or reload_gate.get("serialized_bytes_equal_behavior_certified_bytes")
        is not True
        or reload_gate.get("internal_numeric_matching_consulted") is not False
    ):
        raise ExactCandidateFrontierError(
            "Winner publication reload behavior certificate is incomplete."
        )
    certificate = report.get("fresh_reload_certificate")
    independent_report = report.get("independent_reload_certificate")
    required_true = (
        "fresh_reload_performed",
        "exact_budget_match",
        "state_keys_match_pre_save",
        "state_bytes_match_pre_save",
    )
    if not isinstance(certificate, Mapping) or any(
        certificate.get(name) is not True for name in required_true
    ):
        raise ExactCandidateFrontierError(
            "Winner publication lacks its fresh-reload certificate."
        )
    exact_parameters = _nonnegative_int(
        certificate.get("exact_unique_parameters"),
        "fresh_reload_certificate.exact_unique_parameters",
    )
    if (
        exact_parameters <= 0
        or certificate.get("published_adapter_parameter_count") != 0
        or certificate.get("published_parent_runtime_reference_count") != 0
        or certificate.get("published_source_sized_sidecar_parameter_count") != 0
        or not isinstance(independent_report, Mapping)
    ):
        raise ExactCandidateFrontierError(
            "Winner publication certificate is not exact and parent-free."
        )
    child_path = _resolve_binding(
        report_path.parent,
        certificate.get("path"),
        "fresh_reload_certificate.path",
    )
    if child_path.parent != root or child_path.name != _CHILD_DIRECTORY:
        raise ExactCandidateFrontierError(
            "Winner publication child is not the canonical direct artifact."
        )
    _assert_ordinary_directory(child_path, "accepted child")
    tree = artifact_tree(child_path)
    declared_tree = certificate.get("artifact_tree")
    if not isinstance(declared_tree, Mapping) or any(
        declared_tree.get(name) != tree[name]
        for name in ("file_count", "total_bytes", "tree_sha256")
    ):
        raise ExactCandidateFrontierError(
            "Winner publication child tree differs from its certificate."
        )
    child_state_sha = require_sha256(
        certificate.get("state_dict_sha256"),
        "fresh_reload_certificate.state_dict_sha256",
    )
    if accepted_child_authenticator is None:
        raise ExactCandidateFrontierError(
            "Winner publication requires independent fresh-load authentication."
        )
    independent = accepted_child_authenticator(child_path, exact_parameters)
    if not isinstance(independent, Mapping):
        raise ExactCandidateFrontierError(
            "Independent child authenticator returned a non-mapping."
        )
    independent_tree = independent.get("artifact_tree")
    if (
        canonical_json(independent) != canonical_json(independent_report)
        or independent.get("fresh_reload_performed") is not True
        or independent.get("loaded_from_candidate_path") is not True
        or independent.get("exact_unique_parameters") != exact_parameters
        or independent.get("state_dict_sha256") != child_state_sha
        or independent.get("published_adapter_parameter_count") != 0
        or independent.get("parent_runtime_reference_count") != 0
        or independent.get("source_sized_sidecar_parameter_count") != 0
        or not isinstance(independent_tree, Mapping)
        or any(
            independent_tree.get(name) != tree[name]
            for name in ("file_count", "total_bytes", "tree_sha256")
        )
    ):
        raise ExactCandidateFrontierError(
            "Winner publication independent reload no longer verifies."
        )
    protected = {
        f"source_run_{report_sha[:16]}": str(root),
        f"source_child_{tree['tree_sha256'][:16]}": str(child_path),
        f"source_report_{report_sha[:16]}": str(report_path),
        f"source_prereg_{prereg_sha[:16]}": str(prereg_path),
        f"source_guided_prereg_{source_guided_sha[:16]}": str(source_guided_path),
        f"sealed_marker_{marker_sha[:16]}": str(marker_path),
    }
    source_guided = preregistration.get("source_guided")
    target_contract = require_sha256(
        source_guided.get("evaluation_contract_sha256")
        if isinstance(source_guided, Mapping)
        else None,
        "source_guided.evaluation_contract_sha256",
    )
    return TerminalCandidate(
        terminal_run_schema=GUIDED_WINNER_PUBLICATION_RUN_SCHEMA,
        run_root=root,
        run_report_path=report_path,
        run_report_sha256=report_sha,
        preregistration_path=prereg_path,
        preregistration_sha256=prereg_sha,
        comparison_contract_sha256=comparison,
        target_contract_sha256=target_contract,
        exact_unique_parameters=exact_parameters,
        artifact_kind="accepted-standalone-child",
        truth_label="accepted-standalone",
        terminal_status="accepted",
        terminal_reason=None,
        candidate_metrics=dict(post_metrics),
        # This schema's method gate is authenticated above, but its three
        # tournament/publication checks are not the legacy guided rank contract.
        # Keep frontier ordering purely output-behavior based.
        dynamic_method_gate_decision=None,
        rollout_first_rank_key=rollout_first_rank(post_metrics),
        selected_step=selected_step,
        selected_factor_state_sha256=selected_sha,
        content_identity_sha256=child_state_sha,
        source_artifact=child_path,
        source_artifact_identity_sha256=str(tree["tree_sha256"]),
        source_artifact_tree=tree,
        fresh_reload_verified=True,
        fresh_reload_certificate_sha256=canonical_sha256(
            {
                "run_certificate": dict(certificate),
                "independent_authentication": dict(independent),
            }
        ),
        independent_child_authentication=dict(independent),
        standalone_compaction_passed=True,
        deployment_accepted=True,
        protected_roots=protected,
    )


def _authenticate_guided_gud_winner_publication_candidate(
    root: Path,
    report_path: Path,
    report: Mapping[str, Any],
    report_sha: str,
    *,
    expected_preregistration_sha256: str,
    preregistration_authenticator: PreregistrationAuthenticator,
    accepted_child_authenticator: AcceptedChildAuthenticator | None,
) -> TerminalCandidate:
    """Authenticate a v4 QKVO+GUD one-shot publication and exact child."""

    if (
        report.get("status") != "accepted"
        or report.get("standalone_compaction_passed") is not True
        or report.get("model_checkpoint_written") is not True
        or report.get("model_save_count") != 1
        or report.get("rollback_exact") is not None
        or report.get("sealed_forward_count") != 1
        or report.get("sealed_marker_consumed") is not True
        or report.get("post_bake_sealed_forward_count") != 0
        or report.get("sealed_evaluation_accessed") is not True
        or report.get("winner_locked_before_sealed") is not True
        or report.get("sealed_used_for_acceptance_only") is not True
        or report.get("sealed_used_for_reranking") is not False
        or report.get("exact_unique_parameters") != QWEN3_1P4B_EXACT_PARAMETERS
        or report.get("published_adapter_parameter_count") != 0
        or report.get("published_parent_runtime_reference_count") != 0
        or report.get("published_source_sized_sidecar_parameter_count") != 0
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication lacks terminal one-shot exact-child evidence."
        )
    prereg_path = _resolve_binding(
        report_path.parent,
        report.get("preregistration"),
        "run_report.preregistration",
    )
    _assert_ordinary_file(prereg_path, "guided-GUD publication preregistration")
    prereg_sha = file_sha256(prereg_path)
    if (
        prereg_sha
        != require_sha256(
            expected_preregistration_sha256,
            "expected_preregistration_sha256",
        )
        or prereg_sha
        != require_sha256(
            report.get("preregistration_sha256"),
            "preregistration_sha256",
        )
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication preregistration external roots disagree."
        )
    preregistration = strict_json(prereg_path)
    if (
        not isinstance(preregistration, Mapping)
        or preregistration.get("schema")
        != GUIDED_GUD_WINNER_PUBLICATION_PREREGISTRATION_SCHEMA
        or preregistration.get("model_free_preregistration") is not True
        or preregistration.get("model_weights_loaded") is not False
        or preregistration.get("gpu_work_launched") is not False
        or preregistration.get("publication_authorized_before_sealed_pass")
        is not False
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication preregistration is not model-free/sealed."
        )
    authenticated = preregistration_authenticator(prereg_path, prereg_sha)
    if (
        not isinstance(authenticated, Mapping)
        or canonical_json(authenticated) != canonical_json(preregistration)
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD canonical preregistration authentication disagrees."
        )
    comparison = require_sha256(
        report.get("semantic_contract_sha256"),
        "semantic_contract_sha256",
    )
    if (
        preregistration.get("semantic_contract_sha256") != comparison
        or report.get("evaluation_contract_sha256") != comparison
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication semantic contract failed authentication."
        )
    winner = preregistration.get("winner")
    if not isinstance(winner, Mapping) or report.get("winner") != winner:
        raise ExactCandidateFrontierError(
            "Guided-GUD publication lost its locked winner provenance."
        )
    selected_step = _nonnegative_int(winner.get("step"), "winner.step")
    winner_metrics = winner.get("metrics")
    if (
        selected_step <= 0
        or not isinstance(winner_metrics, Mapping)
        or winner.get("metrics_sha256") != canonical_sha256(winner_metrics)
        or not _valid_winner_publication_pareto_admission(
            winner,
            preregistration.get("semantic_contract"),
        )
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication winner/Pareto admission is unauthenticated."
        )
    thresholds, target_contract, source_evidence, source_provenance = (
        _guided_gud_winner_publication_source_thresholds(
            preregistration,
            prereg_path,
        )
    )
    require_sha256(
        report.get("runtime_rebound_v4_factor_state_sha256"),
        "runtime_rebound_v4_factor_state_sha256",
    )
    require_sha256(
        report.get("runtime_rebound_v3_factor_state_sha256"),
        "runtime_rebound_v3_factor_state_sha256",
    )
    winner_portable_sha = require_sha256(
        winner.get("portable_factor_state_sha256"),
        "winner.portable_factor_state_sha256",
    )
    require_sha256(
        winner.get("runtime_factor_state_sha256"),
        "winner.runtime_factor_state_sha256",
    )
    if report.get("winner_portable_v4_factor_state_sha256") != winner_portable_sha:
        raise ExactCandidateFrontierError(
            "Guided-GUD publication v4 factor provenance is invalid."
        )
    selected_sha = winner_portable_sha

    pre = report.get("pre_bake_behavior_decisions")
    post = report.get("post_bake_behavior_decisions")
    sealed = report.get("sealed_evaluation")
    pre_method = report.get("guided_candidate_method_decision")
    post_method = report.get("post_bake_guided_candidate_method_decision")
    if (
        not isinstance(pre, Mapping)
        or pre.get("fit") is not None
        or not isinstance(pre.get("candidate"), Mapping)
        or not isinstance(pre.get("sealed"), Mapping)
        or not isinstance(post, Mapping)
        or not isinstance(post.get("fit"), Mapping)
        or not isinstance(post.get("candidate"), Mapping)
        or post.get("sealed") is not None
        or not _valid_winner_publication_behavior_decision(
            pre["candidate"],
            expected_metrics=winner_metrics,
            expected_thresholds=thresholds["candidate"],
        )
        or report.get("pre_sealed_candidate_decision") != pre["candidate"]
        or not _valid_winner_publication_behavior_decision(
            pre["sealed"],
            expected_metrics=None,
            expected_thresholds=thresholds["sealed"],
        )
        or not _valid_winner_publication_behavior_decision(
            post["fit"],
            expected_metrics=None,
            expected_thresholds=thresholds["fit"],
        )
        or not _valid_winner_publication_behavior_decision(
            post["candidate"],
            expected_metrics=None,
            expected_thresholds=thresholds["candidate"],
        )
        or not _valid_winner_publication_method_decision(
            pre_method,
            winner_step=selected_step,
            candidate_metrics=pre["candidate"]["metrics"],
        )
        or not _valid_winner_publication_method_decision(
            post_method,
            winner_step=selected_step,
            candidate_metrics=post["candidate"]["metrics"],
        )
        or not isinstance(sealed, Mapping)
        or sealed.get("forward_count") != 1
        or sealed.get("one_shot_marker_consumed") is not True
        or sealed.get("post_bake_sealed_forward_count") != 0
        or sealed.get("acceptance_only") is not True
        or sealed.get("reranking_forbidden") is not True
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication behavior gates violate one-shot sealed policy."
        )

    sealed_binding = preregistration.get("sealed_consumption")
    marker_receipt = sealed.get("marker_receipt")
    split_rows = preregistration.get("splits")
    sealed_split = (
        split_rows.get("sealed") if isinstance(split_rows, Mapping) else None
    )
    if (
        not isinstance(sealed_binding, Mapping)
        or sealed_binding.get("schema") != GUIDED_GUD_SEALED_CONSUMPTION_SCHEMA
        or sealed_binding.get("must_be_absent_at_preflight") is not True
        or sealed_binding.get("exclusive_create_immediately_before_forward")
        is not True
        or sealed_binding.get("maximum_forward_count") != 1
        or not isinstance(marker_receipt, Mapping)
        or marker_receipt.get("exclusive_create_succeeded") is not True
        or marker_receipt.get("sealed_forward_limit") != 1
        or marker_receipt.get("preregistration_sha256") != prereg_sha
        or marker_receipt.get("semantic_contract_sha256") != comparison
        or marker_receipt.get("winner_portable_v4_factor_state_sha256")
        != winner_portable_sha
        or not isinstance(sealed_split, Mapping)
        or marker_receipt.get("sealed_split_sha256") != sealed_split.get("sha256")
        or marker_receipt.get("output_root") != str(root)
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication sealed marker receipt is unauthenticated."
        )
    marker_path = _resolve_binding(
        report_path.parent,
        marker_receipt.get("marker"),
        "sealed_evaluation.marker_receipt.marker",
    )
    declared_marker_path = _resolve_binding(
        prereg_path.parent,
        sealed_binding.get("marker"),
        "sealed_consumption.marker",
    )
    if (
        marker_path != declared_marker_path
        or marker_path.parent != prereg_path.parent
        or marker_path.name != "sealed_consumption.json"
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD sealed marker escaped its preregistration root."
        )
    _assert_ordinary_file(marker_path, "guided-GUD publication sealed marker")
    marker_sha = require_sha256(
        marker_receipt.get("marker_file_sha256"),
        "sealed_evaluation.marker_receipt.marker_file_sha256",
    )
    marker_payload = strict_json(marker_path)
    marker_expected = {
        "schema": GUIDED_GUD_SEALED_CONSUMPTION_SCHEMA,
        "status": "consumed-before-forward",
        "preregistration_sha256": prereg_sha,
        "semantic_contract_sha256": comparison,
        "winner_portable_v4_factor_state_sha256": winner_portable_sha,
        "sealed_split_sha256": sealed_split.get("sha256"),
        "sealed_forward_limit": 1,
        "output_root": str(root),
    }
    if (
        file_sha256(marker_path) != marker_sha
        or marker_payload != marker_expected
        or any(
            marker_receipt.get(name) != value
            for name, value in marker_expected.items()
        )
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication sealed marker bytes no longer authenticate."
        )

    gate_evidence = report.get("external_behavior_gate_evidence")
    if (
        not isinstance(gate_evidence, Mapping)
        or report.get("external_behavior_gate_evidence_sha256")
        != canonical_sha256(gate_evidence)
        or gate_evidence.get("preregistration_sha256") != prereg_sha
        or gate_evidence.get("winner_portable_v4_factor_state_sha256")
        != winner_portable_sha
        or gate_evidence.get("candidate_decision_sha256")
        != pre["candidate"].get("decision_sha256")
        or gate_evidence.get("method_decision_sha256")
        != pre_method.get("decision_sha256")
        or gate_evidence.get("sealed_decision_sha256")
        != pre["sealed"].get("decision_sha256")
        or gate_evidence.get("sealed_marker_file_sha256") != marker_sha
        or gate_evidence.get("winner_locked_before_sealed") is not True
        or gate_evidence.get("sealed_used_for_reranking") is not False
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD external behavior-gate evidence is unauthenticated."
        )
    _authenticate_guided_gud_publication_provenance(
        report,
        comparison=comparison,
        provenance=source_provenance,
    )
    handoff = report.get("guided_v4_publication_handoff")
    if (
        not isinstance(handoff, Mapping)
        or handoff.get("handoff_complete") is not True
        or handoff.get("transaction_qkvo_handle_is_guided_composite") is not True
        or handoff.get("transaction_gud_handle_is_guided_composite") is not True
        or handoff.get("old_only_publication_paths_disabled") is not True
        or handoff.get("sequential_dense_bake")
        != [
            "lexical",
            "inherited-and-guided-qkvo",
            "inherited-and-guided-gud",
        ]
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD dual-composite publication handoff is incomplete."
        )

    post_metrics = post["candidate"].get("metrics")
    if not isinstance(post_metrics, Mapping):
        raise ExactCandidateFrontierError(
            "Guided-GUD post-bake candidate metrics are absent."
        )
    reload_gate = report.get("fresh_reload_guided_v4_method_gate")
    if (
        not isinstance(reload_gate, Mapping)
        or reload_gate.get("passed") is not True
        or reload_gate.get("post_bake_decision_sha256")
        != post_method.get("decision_sha256")
        or reload_gate.get("state_bytes_match_pre_save") is not True
        or reload_gate.get("candidate_behavior_certified_before_save") is not True
        or reload_gate.get("serialized_bytes_equal_behavior_certified_bytes")
        is not True
        or reload_gate.get("internal_numeric_matching_consulted") is not False
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD reload behavior certificate is incomplete."
        )
    certificate = report.get("fresh_reload_certificate")
    independent_report = report.get("independent_reload_certificate")
    required_true = (
        "fresh_reload_performed",
        "exact_budget_match",
        "state_keys_match_pre_save",
        "state_bytes_match_pre_save",
    )
    if not isinstance(certificate, Mapping) or any(
        certificate.get(name) is not True for name in required_true
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD publication lacks its fresh-reload certificate."
        )
    exact_parameters = _nonnegative_int(
        certificate.get("exact_unique_parameters"),
        "fresh_reload_certificate.exact_unique_parameters",
    )
    if (
        exact_parameters != QWEN3_1P4B_EXACT_PARAMETERS
        or certificate.get("published_adapter_parameter_count") != 0
        or certificate.get("published_parent_runtime_reference_count") != 0
        or certificate.get("published_source_sized_sidecar_parameter_count") != 0
        or not isinstance(independent_report, Mapping)
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD certificate is not exact and parent-free."
        )
    child_path = _resolve_binding(
        report_path.parent,
        certificate.get("path"),
        "fresh_reload_certificate.path",
    )
    if child_path.parent != root or child_path.name != _CHILD_DIRECTORY:
        raise ExactCandidateFrontierError(
            "Guided-GUD child is not the canonical direct artifact."
        )
    _assert_ordinary_directory(child_path, "guided-GUD accepted child")
    tree = artifact_tree(child_path)
    declared_tree = certificate.get("artifact_tree")
    if not isinstance(declared_tree, Mapping) or any(
        declared_tree.get(name) != tree[name]
        for name in ("file_count", "total_bytes", "tree_sha256")
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD child tree differs from its certificate."
        )
    child_state_sha = require_sha256(
        certificate.get("state_dict_sha256"),
        "fresh_reload_certificate.state_dict_sha256",
    )
    if accepted_child_authenticator is None:
        raise ExactCandidateFrontierError(
            "Guided-GUD publication requires independent fresh-load authentication."
        )
    independent = accepted_child_authenticator(child_path, exact_parameters)
    if not isinstance(independent, Mapping):
        raise ExactCandidateFrontierError(
            "Independent guided-GUD child authenticator returned a non-mapping."
        )
    independent_tree = independent.get("artifact_tree")
    if (
        canonical_json(independent) != canonical_json(independent_report)
        or independent.get("fresh_reload_performed") is not True
        or independent.get("loaded_from_candidate_path") is not True
        or independent.get("exact_unique_parameters") != exact_parameters
        or independent.get("state_dict_sha256") != child_state_sha
        or independent.get("published_adapter_parameter_count") != 0
        or independent.get("parent_runtime_reference_count") != 0
        or independent.get("source_sized_sidecar_parameter_count") != 0
        or not isinstance(independent_tree, Mapping)
        or any(
            independent_tree.get(name) != tree[name]
            for name in ("file_count", "total_bytes", "tree_sha256")
        )
    ):
        raise ExactCandidateFrontierError(
            "Guided-GUD independent reload no longer verifies."
        )

    protected = {
        f"source_run_{report_sha[:16]}": str(root),
        f"source_child_{tree['tree_sha256'][:16]}": str(child_path),
        f"source_report_{report_sha[:16]}": str(report_path),
        f"source_prereg_{prereg_sha[:16]}": str(prereg_path),
        f"sealed_marker_{marker_sha[:16]}": str(marker_path),
    }
    for label, (path, digest) in source_evidence.items():
        protected[f"{label}_{digest[:16]}"] = str(path)
    return TerminalCandidate(
        terminal_run_schema=GUIDED_GUD_WINNER_PUBLICATION_RUN_SCHEMA,
        run_root=root,
        run_report_path=report_path,
        run_report_sha256=report_sha,
        preregistration_path=prereg_path,
        preregistration_sha256=prereg_sha,
        comparison_contract_sha256=comparison,
        target_contract_sha256=target_contract,
        exact_unique_parameters=exact_parameters,
        artifact_kind="accepted-standalone-child",
        truth_label="accepted-standalone",
        terminal_status="accepted",
        terminal_reason=None,
        candidate_metrics=dict(post_metrics),
        dynamic_method_gate_decision=None,
        rollout_first_rank_key=rollout_first_rank(post_metrics),
        selected_step=selected_step,
        selected_factor_state_sha256=selected_sha,
        content_identity_sha256=child_state_sha,
        source_artifact=child_path,
        source_artifact_identity_sha256=str(tree["tree_sha256"]),
        source_artifact_tree=tree,
        fresh_reload_verified=True,
        fresh_reload_certificate_sha256=canonical_sha256(
            {
                "run_certificate": dict(certificate),
                "independent_authentication": dict(independent),
            }
        ),
        independent_child_authentication=dict(independent),
        standalone_compaction_passed=True,
        deployment_accepted=True,
        protected_roots=protected,
    )


def authenticate_terminal_candidate(
    run_root: Path | str,
    *,
    expected_run_report_sha256: str,
    expected_preregistration_sha256: str,
    preregistration_authenticator: PreregistrationAuthenticator,
    factor_authenticator: FactorAuthenticator | None = None,
    accepted_child_authenticator: AcceptedChildAuthenticator | None = None,
    method_gate_authenticator: MethodGateAuthenticator | None = None,
) -> TerminalCandidate:
    """Authenticate one run; accepted children require an independent reload."""

    root = Path(run_root).expanduser().resolve()
    _assert_ordinary_directory(root, "run root")
    report_path = root / "run_report.json"
    _assert_ordinary_file(report_path, "terminal run report")
    report_sha = file_sha256(report_path)
    if report_sha != require_sha256(
        expected_run_report_sha256, "expected_run_report_sha256"
    ):
        raise ExactCandidateFrontierError(
            "Terminal run report does not match the caller-provided external root."
        )
    report = strict_json(report_path)
    run_schema = report.get("schema") if isinstance(report, Mapping) else None
    if run_schema == GUIDED_GUD_WINNER_PUBLICATION_RUN_SCHEMA:
        return _authenticate_guided_gud_winner_publication_candidate(
            root,
            report_path,
            report,
            report_sha,
            expected_preregistration_sha256=expected_preregistration_sha256,
            preregistration_authenticator=preregistration_authenticator,
            accepted_child_authenticator=accepted_child_authenticator,
        )
    if run_schema == GUIDED_WINNER_PUBLICATION_RUN_SCHEMA:
        return _authenticate_guided_winner_publication_candidate(
            root,
            report_path,
            report,
            report_sha,
            expected_preregistration_sha256=expected_preregistration_sha256,
            preregistration_authenticator=preregistration_authenticator,
            accepted_child_authenticator=accepted_child_authenticator,
        )
    if run_schema not in {RUN_SCHEMA, GUIDED_RUN_SCHEMA}:
        raise ExactCandidateFrontierError("Unsupported terminal run report schema.")
    guided = run_schema == GUIDED_RUN_SCHEMA
    status = report.get("status")
    if status not in {"rejected", "accepted"}:
        raise ExactCandidateFrontierError(
            "Only terminal rejected or accepted training runs are candidates."
        )
    if guided:
        _validate_guided_terminal_report_schema(report, status)
    prereg_path = _resolve_binding(
        report_path.parent,
        report.get("preregistration"),
        "run_report.preregistration",
    )
    _assert_ordinary_file(prereg_path, "preregistration")
    prereg_sha = file_sha256(prereg_path)
    externally_expected_preregistration = require_sha256(
        expected_preregistration_sha256, "expected_preregistration_sha256"
    )
    if (
        prereg_sha != externally_expected_preregistration
        or prereg_sha
        != require_sha256(
            report.get("preregistration_sha256"), "preregistration_sha256"
        )
    ):
        raise ExactCandidateFrontierError(
            "Preregistration does not match both external and report roots."
        )
    preregistration = strict_json(prereg_path)
    expected_preregistration_schema = (
        GUIDED_PREREGISTRATION_SCHEMA if guided else PREREGISTRATION_SCHEMA
    )
    if (
        not isinstance(preregistration, Mapping)
        or preregistration.get("schema") != expected_preregistration_schema
        or preregistration.get("sealed_before_model_access") is not True
    ):
        raise ExactCandidateFrontierError(
            "Preregistration schema or sealing attestation is invalid."
        )
    authenticated_preregistration = _authenticate_canonical_preregistration(
        preregistration,
        prereg_path,
        prereg_sha,
        report,
        preregistration_authenticator,
    )
    expected_rollout_count, expected_horizon = _candidate_metric_geometry(
        report, preregistration
    )
    evidence = _trainer_evidence(report)
    if (
        evidence.get("selection_policy") != "rollout-first-acquisition"
        or evidence.get("internal_numeric_matching_used") is not False
        or evidence.get("selection_uses_internal_numeric_targets") is not False
    ):
        raise ExactCandidateFrontierError(
            "Trainer evidence does not prove output-only rollout-first selection."
        )
    (
        selected_sha,
        selected_step,
        selected_metrics,
        step0_metrics,
    ) = _selected_factor_state(
        report,
        evidence,
        expected_rollout_count=expected_rollout_count,
        expected_horizon=expected_horizon,
    )
    dynamic_method_decision: Mapping[str, Any] | None = None
    dynamic_method_passed: bool | None = None
    if guided:
        if method_gate_authenticator is None:
            raise ExactCandidateFrontierError(
                "A source-pinned guided method-gate authenticator is required."
            )
        authenticated_gate = method_gate_authenticator(evidence, None)
        if (
            not isinstance(authenticated_gate, tuple)
            or len(authenticated_gate) != 2
            or not isinstance(authenticated_gate[0], bool)
            or not isinstance(authenticated_gate[1], Mapping)
        ):
            raise ExactCandidateFrontierError(
                "Guided method-gate authenticator returned a malformed result."
            )
        dynamic_method_passed = authenticated_gate[0]
        dynamic_method_decision = dict(authenticated_gate[1])
        if (
            report.get("guided_method_decision") != dynamic_method_decision
            or dynamic_method_decision.get("passed") is not dynamic_method_passed
            or dynamic_method_decision.get("thresholds")
            != authenticated_preregistration.get("method_gate")
        ):
            raise ExactCandidateFrontierError(
                "Guided terminal method decision does not authenticate."
            )
        _dynamic_method_rank(dynamic_method_decision)
    selected_rank = behavior_rank(selected_metrics, dynamic_method_decision)
    step0_rank = rollout_first_rank(step0_metrics)
    if not rollout_first_rank(selected_metrics) < step0_rank:
        raise ExactCandidateFrontierError(
            "Selected candidate does not strictly improve the authenticated step-zero "
            "rollout-first behavior key."
        )
    pre_metrics, pre_decisions = _decision_metrics(report, post_bake=False)
    _validate_candidate_metric_geometry(
        pre_metrics,
        rollout_count=expected_rollout_count,
        horizon=expected_horizon,
    )
    evidence_metrics = evidence.get("candidate_metrics")
    if (
        not isinstance(evidence_metrics, Mapping)
        or canonical_sha256(pre_metrics) != canonical_sha256(evidence_metrics)
        or canonical_sha256(pre_metrics) != canonical_sha256(selected_metrics)
    ):
        raise ExactCandidateFrontierError(
            "Selected checkpoint, trainer evidence, and candidate decision disagree."
        )
    expected_candidate_gate = bool(pre_decisions["candidate"]["passed"])
    if guided:
        expected_candidate_gate = expected_candidate_gate and bool(
            dynamic_method_passed
        )
    if (
        not isinstance(evidence.get("candidate_gate_passed"), bool)
        or evidence.get("candidate_gate_passed") is not expected_candidate_gate
        or not isinstance(evidence.get("sealed_evaluation_unlocked"), bool)
        or evidence.get("sealed_evaluation_unlocked")
        is not bool(evidence.get("candidate_gate_passed"))
    ):
        raise ExactCandidateFrontierError(
            "Trainer gate/unlock evidence disagrees with candidate behavior."
        )
    comparison, target_contract, exact_parameters = _comparison_and_target_contracts(
        report, preregistration, authenticated_preregistration
    )
    protected = _protected_roots(
        report, preregistration, prereg_path, authenticated_preregistration
    )
    reason = report.get("reason")
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ExactCandidateFrontierError("Terminal reason must be nonempty when set.")

    def factor_only_candidate(
        *, fallback_reason: str | None = None
    ) -> TerminalCandidate:
        factor_field = (
            "selected_factor_pack"
            if guided
            else (
                "factor_pack"
                if isinstance(report.get("factor_pack"), Mapping)
                else "selected_factor_pack"
            )
        )
        factor = report.get(factor_field)
        if not isinstance(factor, Mapping):
            raise ExactCandidateFrontierError(
                "No independently authenticated child or retained factor-only "
                "fallback is available."
            )
        factor_path = _resolve_binding(
            report_path.parent,
            factor.get("path"),
            f"run_report.{factor_field}.path",
        )
        allowed_names = (
            {_GUIDED_SELECTED_FACTOR_FILENAME}
            if guided
            else (
                {_FACTOR_FILENAME}
                if status == "rejected"
                else {_FACTOR_FILENAME, _SELECTED_FACTOR_FILENAME}
            )
        )
        if factor_path.parent != root or factor_path.name not in allowed_names:
            raise ExactCandidateFrontierError(
                "Factor pack is not a canonical direct run artifact."
            )
        _assert_ordinary_file(factor_path, "selected factor pack")
        factor_file_sha = file_sha256(factor_path)
        if factor_file_sha != require_sha256(
            factor.get("sha256"), f"run_report.{factor_field}.sha256"
        ):
            raise ExactCandidateFrontierError("Factor pack file hash does not verify.")
        if factor_authenticator is None:
            raise ExactCandidateFrontierError(
                "A source-pinned exact factor authenticator is required."
            )
        lexical = preregistration.get("lexical_residual")
        lexical_rank = None
        if guided:
            retention = authenticated_preregistration.get("retention_authentication")
            if not isinstance(retention, Mapping):
                raise ExactCandidateFrontierError(
                    "Guided retention authentication is missing."
                )
            lexical_rank = _nonnegative_int(
                retention.get("lexical_rank"),
                "retention_authentication.lexical_rank",
            )
        if lexical is not None:
            if not isinstance(lexical, Mapping):
                raise ExactCandidateFrontierError("lexical_residual is malformed.")
            lexical_rank = _nonnegative_int(
                lexical.get("rank"), "lexical_residual.rank"
            )
            if lexical_rank <= 0:
                raise ExactCandidateFrontierError(
                    "lexical_residual.rank must be positive."
                )
        authenticated = factor_authenticator(factor_path, lexical_rank)
        if not isinstance(authenticated, Mapping):
            raise ExactCandidateFrontierError(
                "Factor authenticator returned a non-mapping result."
            )
        envelope = authenticated.get("envelope_provenance")
        envelope_reason = (
            envelope.get("reason") if isinstance(envelope, Mapping) else None
        )
        if (
            require_sha256(
                authenticated.get("factor_state_sha256"),
                "authenticated factor_state_sha256",
            )
            != selected_sha
            or not isinstance(envelope, Mapping)
            or not isinstance(envelope_reason, str)
            or not envelope_reason
            or (guided and envelope_reason != "selected")
            or (not guided and status == "rejected" and envelope_reason != reason)
            or envelope.get("preregistration_sha256") != prereg_sha
            or envelope.get("evaluation_contract_sha256") != comparison
        ):
            raise ExactCandidateFrontierError(
                "Factor topology/provenance does not match its terminal report."
            )
        if guided:
            retention = authenticated_preregistration.get("retention_authentication")
            source_hashes = (
                retention.get("source_v2_tensor_sha256s")
                if isinstance(retention, Mapping)
                else None
            )
            guided_binding = authenticated.get("guided_source_binding")
            audit = authenticated_preregistration.get("audit")
            source = authenticated_preregistration.get("source")
            guided_installation = report.get("guided_installation")
            installation_binding_fields = (
                "inherited_installation_sha256",
                "inherited_factor_state_sha256_at_install",
                "adapter_site_indices_sha256",
                "audit_site_indices_sha256",
            )
            if (
                authenticated.get("factor_pack_schema")
                != "neural-forge-rejected-target-native-guided-factor-pack-v1"
                or authenticated.get("trainer_factor_pack_schema")
                != "neural-forge-target-native-factor-pack-v3"
                or authenticated.get("factor_count") != _GUIDED_FACTOR_COUNT
                or authenticated.get("factor_parameters") != _GUIDED_FACTOR_PARAMETERS
                or authenticated.get("lexical_rank") != lexical_rank
                or not isinstance(guided_binding, Mapping)
                or guided_binding.get("source_v2_tensor_sha256s") != source_hashes
                or not isinstance(source, Mapping)
                or guided_binding.get("source_v2_factor_state_sha256")
                != source.get("factor_state_sha256")
                or envelope.get("source_factor_state_sha256")
                != source.get("factor_state_sha256")
                or not isinstance(audit, Mapping)
                or envelope.get("audit_report_self_sha256")
                != audit.get("audit_report_self_sha256")
                or envelope.get("audit_report_canonical_sha256")
                != audit.get("audit_report_canonical_sha256")
                or envelope.get("basis_envelope_file_sha256")
                != audit.get("basis_envelope_file_sha256")
                or envelope.get("basis_domain_artifact_sha256")
                != audit.get("basis_domain_artifact_sha256")
                or envelope.get("basis_payload_sha256")
                != audit.get("basis_payload_sha256")
                or guided_binding.get("audit_report_sha256")
                != audit.get("audit_report_self_sha256")
                or guided_binding.get("basis_envelope_file_sha256")
                != audit.get("basis_envelope_file_sha256")
                or guided_binding.get("basis_payload_sha256")
                != audit.get("basis_payload_sha256")
                or guided_binding.get("basis_manifest_sha256")
                != audit.get("basis_manifest_sha256")
                or guided_binding.get("source_basis_tensor_sha256s")
                != audit.get("basis_tensor_sha256s")
                or not isinstance(guided_installation, Mapping)
                or any(
                    guided_binding.get(name) != guided_installation.get(name)
                    for name in installation_binding_fields
                )
            ):
                raise ExactCandidateFrontierError(
                    "Guided v3 factor sources do not match the authenticated "
                    "preregistration chain."
                )
        factor_protected = {
            **protected,
            f"source_run_{report_sha[:16]}": str(root),
            f"source_factor_{factor_file_sha[:8]}_{report_sha[:8]}": str(factor_path),
            f"source_report_{report_sha[:16]}": str(report_path),
            f"source_prereg_{prereg_sha[:8]}_{report_sha[:8]}": str(prereg_path),
        }
        return TerminalCandidate(
            terminal_run_schema=str(run_schema),
            run_root=root,
            run_report_path=report_path,
            run_report_sha256=report_sha,
            preregistration_path=prereg_path,
            preregistration_sha256=prereg_sha,
            comparison_contract_sha256=comparison,
            target_contract_sha256=target_contract,
            exact_unique_parameters=exact_parameters,
            artifact_kind="research-factor-pack",
            truth_label="research-factor-only-candidate",
            terminal_status=status,
            terminal_reason=(reason if fallback_reason is None else fallback_reason),
            candidate_metrics=pre_metrics,
            dynamic_method_gate_decision=(
                None
                if dynamic_method_decision is None
                else dict(dynamic_method_decision)
            ),
            rollout_first_rank_key=selected_rank,
            selected_step=selected_step,
            selected_factor_state_sha256=selected_sha,
            content_identity_sha256=selected_sha,
            source_artifact=factor_path,
            source_artifact_identity_sha256=factor_file_sha,
            source_artifact_tree=None,
            fresh_reload_verified=False,
            fresh_reload_certificate_sha256=None,
            independent_child_authentication=None,
            standalone_compaction_passed=False,
            deployment_accepted=False,
            protected_roots=factor_protected,
        )

    if status == "rejected":
        if (
            report.get("standalone_compaction_passed") is not False
            or report.get("model_checkpoint_written") is not False
            or report.get("rollback_exact") is not True
            or not isinstance(reason, str)
        ):
            raise ExactCandidateFrontierError(
                "Rejected candidate lacks exact rollback/non-standalone evidence."
            )
        return factor_only_candidate()

    if (
        report.get("standalone_compaction_passed") is not True
        or report.get("model_checkpoint_written") is not True
        or report.get("rollback_exact") is not None
    ):
        raise ExactCandidateFrontierError(
            "Accepted candidate lacks standalone publication evidence."
        )
    for stage_name, decisions in (
        ("pre-bake", pre_decisions),
        (
            "post-bake",
            report.get("post_bake_behavior_decisions"),
        ),
    ):
        if not isinstance(decisions, Mapping) or any(
            not isinstance(decisions.get(name), Mapping)
            or decisions[name].get("passed") is not True
            for name in ("fit", "candidate", "sealed")
        ):
            raise ExactCandidateFrontierError(
                f"Accepted candidate did not pass every {stage_name} behavior gate."
            )
    post_metrics, _post_decisions = _decision_metrics(report, post_bake=True)
    _validate_candidate_metric_geometry(
        post_metrics,
        rollout_count=expected_rollout_count,
        horizon=expected_horizon,
    )
    post_dynamic_method_decision: Mapping[str, Any] | None = None
    if guided:
        assert method_gate_authenticator is not None
        authenticated_post_gate = method_gate_authenticator(evidence, post_metrics)
        if (
            not isinstance(authenticated_post_gate, tuple)
            or len(authenticated_post_gate) != 2
            or authenticated_post_gate[0] is not True
            or not isinstance(authenticated_post_gate[1], Mapping)
        ):
            raise ExactCandidateFrontierError(
                "Accepted guided child lacks a passed post-bake method gate."
            )
        post_dynamic_method_decision = dict(authenticated_post_gate[1])
        reload_gate = report.get("fresh_reload_guided_method_gate")
        if (
            report.get("guided_method_decision", {}).get("passed") is not True
            or report.get("post_bake_guided_method_decision")
            != post_dynamic_method_decision
            or post_dynamic_method_decision.get("passed") is not True
            or not isinstance(reload_gate, Mapping)
            or reload_gate.get("passed") is not True
            or reload_gate.get("internal_numeric_matching_consulted") is not False
            or reload_gate.get("post_bake_decision_sha256")
            != post_dynamic_method_decision.get("decision_sha256")
            or reload_gate.get("state_bytes_match_pre_save") is not True
        ):
            raise ExactCandidateFrontierError(
                "Accepted guided method/reload gates do not authenticate."
            )
        _dynamic_method_rank(post_dynamic_method_decision)

    def accepted_standalone_candidate() -> TerminalCandidate:
        certificate = report.get("fresh_reload_certificate")
        required_true = (
            "fresh_reload_performed",
            "exact_budget_match",
            "state_keys_match_pre_save",
            "state_bytes_match_pre_save",
        )
        if not isinstance(certificate, Mapping) or any(
            certificate.get(name) is not True for name in required_true
        ):
            raise ExactCandidateFrontierError(
                "Accepted child lacks a complete fresh-reload certificate."
            )
        if (
            certificate.get("exact_unique_parameters") != exact_parameters
            or certificate.get("published_adapter_parameter_count") != 0
            or certificate.get("published_parent_runtime_reference_count") != 0
            or certificate.get("published_source_sized_sidecar_parameter_count") != 0
        ):
            raise ExactCandidateFrontierError(
                "Accepted child certificate does not prove an exact parent-free "
                "artifact."
            )
        child_path = _resolve_binding(
            report_path.parent,
            certificate.get("path"),
            "fresh_reload_certificate.path",
        )
        if child_path.parent != root or child_path.name != _CHILD_DIRECTORY:
            raise ExactCandidateFrontierError(
                "Accepted child is not the canonical direct run artifact."
            )
        _assert_ordinary_directory(child_path, "accepted child")
        actual_tree = artifact_tree(child_path)
        declared_tree = certificate.get("artifact_tree")
        if not isinstance(declared_tree, Mapping) or any(
            declared_tree.get(name) != actual_tree[name]
            for name in ("file_count", "total_bytes", "tree_sha256")
        ):
            raise ExactCandidateFrontierError(
                "Accepted child tree differs from its fresh-reload certificate."
            )
        child_state_sha = require_sha256(
            certificate.get("state_dict_sha256"),
            "fresh_reload_certificate.state_dict_sha256",
        )
        if accepted_child_authenticator is None:
            raise ExactCandidateFrontierError(
                "Accepted child requires an independent fresh-load authenticator."
            )
        independently_authenticated = accepted_child_authenticator(
            child_path, exact_parameters
        )
        if not isinstance(independently_authenticated, Mapping):
            raise ExactCandidateFrontierError(
                "Independent child authenticator returned a non-mapping."
            )
        independent_tree = independently_authenticated.get("artifact_tree")
        if (
            independently_authenticated.get("fresh_reload_performed") is not True
            or independently_authenticated.get("loaded_from_candidate_path") is not True
            or independently_authenticated.get("exact_unique_parameters")
            != exact_parameters
            or independently_authenticated.get("state_dict_sha256") != child_state_sha
            or independently_authenticated.get("published_adapter_parameter_count") != 0
            or independently_authenticated.get("parent_runtime_reference_count") != 0
            or independently_authenticated.get("source_sized_sidecar_parameter_count")
            != 0
            or not isinstance(independent_tree, Mapping)
            or any(
                independent_tree.get(name) != actual_tree[name]
                for name in ("file_count", "total_bytes", "tree_sha256")
            )
        ):
            raise ExactCandidateFrontierError(
                "Independent fresh-load count/state/tree authentication failed."
            )
        child_protected = {
            **protected,
            f"source_run_{report_sha[:16]}": str(root),
            f"source_child_{actual_tree['tree_sha256'][:8]}_{report_sha[:8]}": str(
                child_path
            ),
            f"source_report_{report_sha[:16]}": str(report_path),
            f"source_prereg_{prereg_sha[:8]}_{report_sha[:8]}": str(prereg_path),
        }
        return TerminalCandidate(
            terminal_run_schema=str(run_schema),
            run_root=root,
            run_report_path=report_path,
            run_report_sha256=report_sha,
            preregistration_path=prereg_path,
            preregistration_sha256=prereg_sha,
            comparison_contract_sha256=comparison,
            target_contract_sha256=target_contract,
            exact_unique_parameters=exact_parameters,
            artifact_kind="accepted-standalone-child",
            truth_label="accepted-standalone",
            terminal_status=status,
            terminal_reason=None,
            candidate_metrics=post_metrics,
            dynamic_method_gate_decision=(
                None
                if post_dynamic_method_decision is None
                else dict(post_dynamic_method_decision)
            ),
            rollout_first_rank_key=behavior_rank(
                post_metrics, post_dynamic_method_decision
            ),
            selected_step=selected_step,
            selected_factor_state_sha256=selected_sha,
            content_identity_sha256=child_state_sha,
            source_artifact=child_path,
            source_artifact_identity_sha256=str(actual_tree["tree_sha256"]),
            source_artifact_tree=actual_tree,
            fresh_reload_verified=True,
            fresh_reload_certificate_sha256=canonical_sha256(
                {
                    "run_certificate": dict(certificate),
                    "independent_authentication": dict(independently_authenticated),
                }
            ),
            independent_child_authentication=dict(independently_authenticated),
            standalone_compaction_passed=True,
            deployment_accepted=True,
            protected_roots=child_protected,
        )

    try:
        return accepted_standalone_candidate()
    except Exception as error:
        try:
            return factor_only_candidate(
                fallback_reason="accepted-child-independent-authentication-failed"
            )
        except ExactCandidateFrontierError:
            raise ExactCandidateFrontierError(
                "Accepted child failed independent authentication and no authenticated "
                "factor-only fallback exists."
            ) from error


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _copy_ordinary_file(source: Path, destination: Path) -> None:
    _assert_ordinary_file(source, "source artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ExactCandidateFrontierError(
            f"Refusing to overwrite staged artifact {destination}."
        )
    shutil.copy2(source, destination, follow_symlinks=False)
    _assert_ordinary_file(destination, "staged artifact")


def _copy_ordinary_tree(source: Path, destination: Path) -> None:
    source_tree = artifact_tree(source)
    destination.mkdir(parents=True)
    for row in source_tree["files"]:
        relative = Path(str(row["path"]))
        _copy_ordinary_file(source / relative, destination / relative)
    if artifact_tree(destination) != source_tree:
        raise ExactCandidateFrontierError(
            "Staged child tree differs from its authenticated source."
        )


class ExactCandidateFrontier:
    """Content-addressed, bounded exact-stage candidate retention."""

    def __init__(
        self,
        root: Path | str,
        *,
        max_per_contract: int = MAX_RETAINED_PER_CONTRACT,
        protected_roots: Sequence[Path | str] = (),
    ) -> None:
        if (
            isinstance(max_per_contract, bool)
            or not isinstance(max_per_contract, int)
            or not 1 <= max_per_contract <= MAX_RETAINED_PER_CONTRACT
        ):
            raise ExactCandidateFrontierError(
                f"max_per_contract must be in [1, {MAX_RETAINED_PER_CONTRACT}]."
            )
        requested = Path(root).expanduser().absolute()
        explicit = tuple(
            Path(value).expanduser().resolve() for value in protected_roots
        )
        requested_comparison = requested.resolve(strict=False)
        for path in explicit:
            _assert_ordinary_directory(path, "protected root")
            if _paths_overlap(requested_comparison, path):
                raise ExactCandidateFrontierError(
                    "Frontier root overlaps an immutable protected root."
                )
        if requested.exists() or requested.is_symlink():
            _assert_ordinary_directory(requested, "frontier root")
            self.root = requested.resolve()
        else:
            self.root = requested_comparison
        self.artifacts_root = self.root / "artifacts"
        self.evidence_root = self.root / "evidence"
        self.manifest_path = self.root / "frontier_manifest.json"
        self.lock_path = self.root / ".frontier.lock"
        self.max_per_contract = max_per_contract
        self.extra_protected_roots = explicit
        for path in self.extra_protected_roots:
            _assert_ordinary_directory(path, "protected root")
            if _paths_overlap(self.root, path):
                raise ExactCandidateFrontierError(
                    "Frontier root overlaps an immutable protected root."
                )

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "schema": FRONTIER_SCHEMA,
            "max_per_comparison_contract": self.max_per_contract,
            "ranking_policy": "candidate-output-rollout-first-acquisition",
            "ranking_includes_authenticated_dynamic_method_gate": True,
            "ranking_uses_fit_or_sealed_metrics": False,
            "ranking_uses_internal_numeric_metrics": False,
            "best_accepted_standalone_fallback_slot": True,
            "prune_mode": "durable-plan-only-conservative-v1",
            "recursive_deletion_implemented": False,
            "apply_prune_supported": False,
            "source_run_deletion_allowed": False,
            "canonical_blank_or_cache_copy_allowed": False,
            "next_sequence": 1,
            "protected_roots": {},
            "records": [],
            "pending_prune": [],
            "evicted_total": 0,
        }

    def _root_entry_names(self) -> set[str]:
        if not self.root.exists():
            return set()
        _assert_ordinary_directory(self.root, "frontier root")
        return {entry.name for entry in os.scandir(self.root)}

    def _load_manifest(
        self, *, allow_new_empty_storage: bool = False
    ) -> dict[str, Any]:
        if not self.manifest_path.exists():
            entries = self._root_entry_names()
            allowed = (
                {"artifacts", "evidence", ".frontier.lock"}
                if allow_new_empty_storage
                else set()
            )
            if entries - allowed:
                raise ExactCandidateFrontierError(
                    "A manifestless frontier contains unowned preexisting entries."
                )
            if allow_new_empty_storage:
                for directory in (self.artifacts_root, self.evidence_root):
                    _assert_ordinary_directory(directory, "new frontier storage")
                    if any(os.scandir(directory)):
                        raise ExactCandidateFrontierError(
                            "Manifestless deterministic storage is not empty."
                        )
            return self._empty_manifest()
        _assert_ordinary_file(self.manifest_path, "frontier manifest")
        value = strict_json(self.manifest_path)
        if not isinstance(value, Mapping):
            raise ExactCandidateFrontierError("Frontier manifest is not an object.")
        payload = dict(value)
        declared = require_sha256(payload.get("manifest_sha256"), "manifest_sha256")
        if (
            payload.get("schema") != FRONTIER_SCHEMA
            or payload.get("max_per_comparison_contract") != self.max_per_contract
            or _record_hash(payload, "manifest_sha256") != declared
            or not isinstance(payload.get("records"), list)
            or not isinstance(payload.get("pending_prune"), list)
            or not isinstance(payload.get("protected_roots"), Mapping)
        ):
            raise ExactCandidateFrontierError(
                "Existing frontier manifest failed authentication."
            )
        next_sequence = _nonnegative_int(payload.get("next_sequence"), "next_sequence")
        if next_sequence <= 0:
            raise ExactCandidateFrontierError("next_sequence must be positive.")
        records = [self._authenticate_record(record) for record in payload["records"]]
        pending = [
            self._authenticate_record(record) for record in payload["pending_prune"]
        ]
        record_hashes = [value["record_sha256"] for value in records]
        pending_hashes = [value["record_sha256"] for value in pending]
        if (
            len(record_hashes) != len(set(record_hashes))
            or len(pending_hashes) != len(set(pending_hashes))
            or set(record_hashes) & set(pending_hashes)
        ):
            raise ExactCandidateFrontierError(
                "Frontier retained and pending record identities are not disjoint."
            )
        group_counts: dict[tuple[str, str], int] = {}
        for record in records:
            key = (
                str(record["target_contract_sha256"]),
                str(record["comparison_contract_sha256"]),
            )
            group_counts[key] = group_counts.get(key, 0) + 1
        if any(value > self.max_per_contract for value in group_counts.values()):
            raise ExactCandidateFrontierError(
                "Frontier exceeds its per-comparison-contract retention bound."
            )
        for record in [*records, *pending]:
            if int(record["sequence"]) >= next_sequence:
                raise ExactCandidateFrontierError(
                    "Frontier record sequence is outside the manifest history."
                )
        self._assert_manifest_owns_every_storage_entry(payload)
        return payload

    def _assert_manifest_owns_every_storage_entry(
        self, manifest: Mapping[str, Any]
    ) -> None:
        entries = self._root_entry_names()
        allowed_root = {
            "artifacts",
            "evidence",
            "frontier_manifest.json",
            ".frontier.lock",
        }
        if entries - allowed_root:
            raise ExactCandidateFrontierError(
                "Frontier root contains an unowned preexisting entry."
            )
        _assert_ordinary_directory(self.artifacts_root, "frontier artifacts root")
        _assert_ordinary_directory(self.evidence_root, "frontier evidence root")
        owned_records = [
            *list(manifest.get("records", [])),
            *list(manifest.get("pending_prune", [])),
        ]
        expected_artifacts = {
            Path(str(record["artifact_path"])).name for record in owned_records
        }
        expected_evidence = {
            Path(str(record["evidence_path"])).name for record in owned_records
        }
        actual_artifacts = {entry.name for entry in os.scandir(self.artifacts_root)}
        actual_evidence = {entry.name for entry in os.scandir(self.evidence_root)}
        if (
            actual_artifacts != expected_artifacts
            or actual_evidence != expected_evidence
        ):
            raise ExactCandidateFrontierError(
                "Frontier storage contains missing, extra, or unowned deterministic "
                "directories."
            )

    def _create_storage_after_authentication(self) -> None:
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=False)
        _assert_ordinary_directory(self.root, "frontier root")
        entries = self._root_entry_names()
        if entries:
            if entries != {"frontier_manifest.json", "artifacts", "evidence"}:
                raise ExactCandidateFrontierError(
                    "Refusing to adopt preexisting frontier storage."
                )
            return
        self.artifacts_root.mkdir()
        self.evidence_root.mkdir()

    def _write_manifest(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(manifest)
        payload["manifest_sha256"] = _record_hash(payload, "manifest_sha256")
        _atomic_replace_json(self.manifest_path, payload)
        return payload

    def _acquire_lock(self) -> int:
        try:
            return os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise ExactCandidateFrontierError(
                f"Frontier is already locked: {self.lock_path}"
            ) from error

    def _release_lock(self, descriptor: int) -> None:
        os.close(descriptor)
        self.lock_path.unlink(missing_ok=True)

    def _protected_mapping(
        self, manifest: Mapping[str, Any], candidate: TerminalCandidate | None
    ) -> dict[str, str]:
        current = {
            str(name): str(Path(path).expanduser().resolve())
            for name, path in dict(manifest.get("protected_roots", {})).items()
        }
        if candidate is not None:
            for name, path in candidate.protected_roots.items():
                normalized = str(Path(path).expanduser().resolve())
                prior = current.get(name)
                if prior is not None and prior != normalized:
                    raise ExactCandidateFrontierError(
                        f"Protected root {name!r} changed across frontier records."
                    )
                current[name] = normalized
        for index, path in enumerate(self.extra_protected_roots):
            current.setdefault(f"explicit_{index}", str(path))
        for name, value in current.items():
            path = Path(value)
            _assert_ordinary_protected_path(path, f"protected path {name}")
            if _paths_overlap(self.root, path):
                raise ExactCandidateFrontierError(
                    f"Frontier overlaps protected root {name!r}."
                )
        return dict(sorted(current.items()))

    def _bundle_path(self, record: Mapping[str, Any], field: str, root: Path) -> Path:
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise ExactCandidateFrontierError(f"Record {field} is invalid.")
        candidate = (self.root / value).resolve()
        if candidate.parent != root.resolve() or candidate == root.resolve():
            raise ExactCandidateFrontierError(
                f"Record {field} escapes its confined frontier root."
            )
        return candidate

    def _authenticate_record(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ExactCandidateFrontierError("Frontier record is not an object.")
        record = dict(value)
        record_sha = require_sha256(record.get("record_sha256"), "record_sha256")
        if (
            record.get("schema") != RECORD_SCHEMA
            or _record_hash(record, "record_sha256") != record_sha
            or record.get("truth_label")
            not in {"research-factor-only-candidate", "accepted-standalone"}
            or record.get("artifact_kind")
            not in {"research-factor-pack", "accepted-standalone-child"}
            or not isinstance(record.get("rollout_first_rank_key"), list)
        ):
            raise ExactCandidateFrontierError("Frontier record failed authentication.")
        rank = tuple(record["rollout_first_rank_key"])
        metrics = record.get("candidate_metrics")
        method_decision = record.get("dynamic_method_gate_decision")
        if method_decision is not None and not isinstance(method_decision, Mapping):
            raise ExactCandidateFrontierError(
                "Frontier record dynamic method decision is malformed."
            )
        if not isinstance(metrics, Mapping) or rank != behavior_rank(
            metrics, method_decision
        ):
            raise ExactCandidateFrontierError(
                "Frontier record behavior ranking does not verify."
            )
        for field in (
            "target_contract_sha256",
            "comparison_contract_sha256",
            "selected_factor_state_sha256",
            "content_identity_sha256",
            "source_artifact_identity_sha256",
            "run_report_sha256",
            "preregistration_sha256",
        ):
            require_sha256(record.get(field), field)
        sequence = _nonnegative_int(record.get("sequence"), "record.sequence")
        if (
            sequence <= 0
            or _nonnegative_int(
                record.get("exact_unique_parameters"), "record.exact_unique_parameters"
            )
            <= 0
        ):
            raise ExactCandidateFrontierError(
                "Frontier record sequence and exact parameter count must be positive."
            )
        research = record["truth_label"] == "research-factor-only-candidate"
        if research != (record["artifact_kind"] == "research-factor-pack"):
            raise ExactCandidateFrontierError(
                "Frontier record truth label and artifact kind disagree."
            )
        if research:
            if (
                record.get("fresh_reload_verified") is not False
                or record.get("fresh_reload_certificate_sha256") is not None
                or record.get("independent_child_authentication") is not None
                or record.get("standalone_compaction_passed") is not False
                or record.get("deployment_accepted") is not False
            ):
                raise ExactCandidateFrontierError(
                    "Research factor-only record overclaims standalone acceptance."
                )
        else:
            require_sha256(
                record.get("fresh_reload_certificate_sha256"),
                "fresh_reload_certificate_sha256",
            )
            if (
                record.get("fresh_reload_verified") is not True
                or not isinstance(
                    record.get("independent_child_authentication"), Mapping
                )
                or record.get("standalone_compaction_passed") is not True
                or record.get("deployment_accepted") is not True
            ):
                raise ExactCandidateFrontierError(
                    "Accepted standalone record lacks fresh-reload acceptance evidence."
                )
        artifact = self._bundle_path(record, "artifact_path", self.artifacts_root)
        evidence = self._bundle_path(record, "evidence_path", self.evidence_root)
        expected_artifact = require_sha256(
            record.get("retained_artifact_tree_sha256"),
            "retained_artifact_tree_sha256",
        )
        expected_evidence = require_sha256(
            record.get("retained_evidence_tree_sha256"),
            "retained_evidence_tree_sha256",
        )
        actual_artifact_tree = artifact_tree(artifact)
        if actual_artifact_tree["tree_sha256"] != expected_artifact:
            raise ExactCandidateFrontierError("Retained candidate artifact changed.")
        actual_evidence_tree = artifact_tree(evidence)
        if actual_evidence_tree["tree_sha256"] != expected_evidence:
            raise ExactCandidateFrontierError("Retained candidate evidence changed.")
        report_copy = evidence / "run_report.json"
        preregistration_copy = evidence / "preregistration.json"
        if (
            file_sha256(report_copy) != record["run_report_sha256"]
            or file_sha256(preregistration_copy) != record["preregistration_sha256"]
        ):
            raise ExactCandidateFrontierError(
                "Retained report/preregistration evidence hashes do not verify."
            )
        copied_report = strict_json(report_copy)
        copied_preregistration = strict_json(preregistration_copy)
        if (
            not isinstance(copied_report, Mapping)
            or not isinstance(copied_preregistration, Mapping)
            or copied_report.get("evaluation_contract_sha256")
            != record["comparison_contract_sha256"]
        ):
            raise ExactCandidateFrontierError(
                "Retained report comparison contract does not verify."
            )
        copied_schema = copied_report.get("schema")
        declared_schema = record.get("terminal_run_schema", copied_schema)
        if (
            copied_schema
            not in {
                RUN_SCHEMA,
                GUIDED_RUN_SCHEMA,
                GUIDED_WINNER_PUBLICATION_RUN_SCHEMA,
                GUIDED_GUD_WINNER_PUBLICATION_RUN_SCHEMA,
            }
            or declared_schema != copied_schema
            or (
                copied_schema == GUIDED_RUN_SCHEMA
                and copied_report.get(
                    "post_bake_guided_method_decision"
                    if record["truth_label"] == "accepted-standalone"
                    else "guided_method_decision"
                )
                != method_decision
            )
            or (
                copied_schema == GUIDED_WINNER_PUBLICATION_RUN_SCHEMA
                and (
                    method_decision is not None
                    or not _valid_retained_winner_publication_behavior(
                        copied_report,
                        copied_preregistration,
                        preregistration_copy,
                    )
                )
            )
            or (
                copied_schema == GUIDED_GUD_WINNER_PUBLICATION_RUN_SCHEMA
                and (
                    method_decision is not None
                    or not _valid_retained_guided_gud_winner_publication_behavior(
                        copied_report,
                        copied_preregistration,
                        preregistration_copy,
                    )
                )
            )
            or (copied_schema == RUN_SCHEMA and method_decision is not None)
        ):
            raise ExactCandidateFrontierError(
                "Retained report schema/dynamic method decision does not verify."
            )
        if research:
            factor_file = artifact / "factor_pack.pt"
            if file_sha256(factor_file) != record["source_artifact_identity_sha256"]:
                raise ExactCandidateFrontierError(
                    "Retained factor pack content identity changed."
                )
        else:
            if (
                actual_artifact_tree["tree_sha256"]
                != record["source_artifact_identity_sha256"]
            ):
                raise ExactCandidateFrontierError(
                    "Retained standalone child tree identity changed."
                )
            independent = record["independent_child_authentication"]
            independent_tree = independent.get("artifact_tree")
            if (
                independent.get("fresh_reload_performed") is not True
                or independent.get("loaded_from_candidate_path") is not True
                or independent.get("exact_unique_parameters")
                != record["exact_unique_parameters"]
                or independent.get("state_dict_sha256")
                != record["content_identity_sha256"]
                or independent.get("published_adapter_parameter_count") != 0
                or independent.get("parent_runtime_reference_count") != 0
                or independent.get("source_sized_sidecar_parameter_count") != 0
                or not isinstance(independent_tree, Mapping)
                or any(
                    independent_tree.get(name) != actual_artifact_tree[name]
                    for name in ("file_count", "total_bytes", "tree_sha256")
                )
            ):
                raise ExactCandidateFrontierError(
                    "Retained independent child authentication no longer verifies."
                )
            certificate = copied_report.get("fresh_reload_certificate")
            if (
                not isinstance(certificate, Mapping)
                or canonical_sha256(
                    {
                        "run_certificate": dict(certificate),
                        "independent_authentication": dict(
                            record["independent_child_authentication"]
                        ),
                    }
                )
                != record["fresh_reload_certificate_sha256"]
            ):
                raise ExactCandidateFrontierError(
                    "Retained fresh-reload certificate does not verify."
                )
        return record

    @staticmethod
    def _rank_records(
        records: Sequence[Mapping[str, Any]], maximum: int
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            (dict(value) for value in records),
            key=lambda value: (
                tuple(value["rollout_first_rank_key"]),
                int(value["sequence"]),
            ),
        )
        kept = ordered[:maximum]
        accepted = next(
            (
                value
                for value in ordered
                if value["truth_label"] == "accepted-standalone"
            ),
            None,
        )
        if accepted is not None and accepted not in kept:
            kept[-1] = accepted
            kept.sort(
                key=lambda value: (
                    tuple(value["rollout_first_rank_key"]),
                    int(value["sequence"]),
                )
            )
        return kept

    def _stage_candidate(
        self,
        candidate: TerminalCandidate,
        record_identity: str,
        manifest: Mapping[str, Any],
    ) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
        artifact_name = (
            "factor-file-"
            if candidate.artifact_kind == "research-factor-pack"
            else "child-tree-"
        ) + candidate.source_artifact_identity_sha256
        artifact_destination = self.artifacts_root / artifact_name
        evidence_destination = self.evidence_root / f"record-{record_identity}"
        owned_records = [
            *list(manifest.get("records", [])),
            *list(manifest.get("pending_prune", [])),
        ]
        owned_artifacts = {
            (self.root / str(record["artifact_path"])).resolve()
            for record in owned_records
        }
        owned_evidence = {
            (self.root / str(record["evidence_path"])).resolve()
            for record in owned_records
        }
        if artifact_destination.exists():
            if artifact_destination.resolve() not in owned_artifacts:
                raise ExactCandidateFrontierError(
                    "Refusing to adopt a preexisting deterministic artifact directory."
                )
            artifact_report = artifact_tree(artifact_destination)
            if candidate.artifact_kind == "research-factor-pack":
                retained_file = artifact_destination / "factor_pack.pt"
                if (
                    file_sha256(retained_file)
                    != candidate.source_artifact_identity_sha256
                ):
                    raise ExactCandidateFrontierError(
                        "Content-addressed factor artifact is corrupt."
                    )
            elif (
                artifact_report["tree_sha256"]
                != candidate.source_artifact_identity_sha256
            ):
                raise ExactCandidateFrontierError(
                    "Content-addressed child artifact is corrupt."
                )
        else:
            temporary = Path(
                tempfile.mkdtemp(prefix=".candidate-artifact.", dir=self.root)
            )
            staged = temporary / "bundle"
            try:
                if candidate.artifact_kind == "research-factor-pack":
                    staged.mkdir()
                    _copy_ordinary_file(
                        candidate.source_artifact, staged / "factor_pack.pt"
                    )
                    if (
                        file_sha256(staged / "factor_pack.pt")
                        != candidate.source_artifact_identity_sha256
                    ):
                        raise ExactCandidateFrontierError(
                            "Staged factor bytes failed authentication."
                        )
                else:
                    _copy_ordinary_tree(candidate.source_artifact, staged)
                    if artifact_tree(staged)["tree_sha256"] != (
                        candidate.source_artifact_identity_sha256
                    ):
                        raise ExactCandidateFrontierError(
                            "Staged child tree failed authentication."
                        )
                os.replace(staged, artifact_destination)
            finally:
                if temporary.exists():
                    try:
                        temporary.rmdir()
                    except OSError:
                        # Conservative v1 never recursively deletes partial trees.
                        pass
            artifact_report = artifact_tree(artifact_destination)

        if evidence_destination.exists():
            if evidence_destination.resolve() not in owned_evidence:
                raise ExactCandidateFrontierError(
                    "Refusing to adopt a preexisting deterministic evidence directory."
                )
            evidence_report = artifact_tree(evidence_destination)
            report_copy = evidence_destination / "run_report.json"
            prereg_copy = evidence_destination / "preregistration.json"
            if (
                file_sha256(report_copy) != candidate.run_report_sha256
                or file_sha256(prereg_copy) != candidate.preregistration_sha256
            ):
                raise ExactCandidateFrontierError(
                    "Content-addressed retained evidence is corrupt."
                )
        else:
            temporary = Path(
                tempfile.mkdtemp(prefix=".candidate-evidence.", dir=self.root)
            )
            staged = temporary / "bundle"
            try:
                staged.mkdir()
                _copy_ordinary_file(
                    candidate.run_report_path, staged / "run_report.json"
                )
                _copy_ordinary_file(
                    candidate.preregistration_path, staged / "preregistration.json"
                )
                if (
                    file_sha256(staged / "run_report.json")
                    != candidate.run_report_sha256
                    or file_sha256(staged / "preregistration.json")
                    != candidate.preregistration_sha256
                ):
                    raise ExactCandidateFrontierError(
                        "Staged evidence bytes failed authentication."
                    )
                os.replace(staged, evidence_destination)
            finally:
                if temporary.exists():
                    try:
                        temporary.rmdir()
                    except OSError:
                        # Conservative v1 never recursively deletes partial trees.
                        pass
            evidence_report = artifact_tree(evidence_destination)
        return (
            artifact_destination,
            artifact_report,
            evidence_destination,
            evidence_report,
        )

    def _record_payload(
        self,
        candidate: TerminalCandidate,
        *,
        sequence: int,
        artifact_path: Path,
        artifact_report: Mapping[str, Any],
        evidence_path: Path,
        evidence_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": RECORD_SCHEMA,
            "sequence": sequence,
            "target_contract_sha256": candidate.target_contract_sha256,
            "comparison_contract_sha256": candidate.comparison_contract_sha256,
            "exact_unique_parameters": candidate.exact_unique_parameters,
            "artifact_kind": candidate.artifact_kind,
            "truth_label": candidate.truth_label,
            "terminal_status": candidate.terminal_status,
            "terminal_reason": candidate.terminal_reason,
            "terminal_run_schema": candidate.terminal_run_schema,
            "candidate_metrics": dict(candidate.candidate_metrics),
            "dynamic_method_gate_decision": (
                None
                if candidate.dynamic_method_gate_decision is None
                else dict(candidate.dynamic_method_gate_decision)
            ),
            "rollout_first_rank_key": list(candidate.rollout_first_rank_key),
            "ranking_domain": "candidate-output-actions-and-free-rollouts-only",
            "selected_step": candidate.selected_step,
            "selected_factor_state_sha256": (candidate.selected_factor_state_sha256),
            "content_identity_sha256": candidate.content_identity_sha256,
            "source_artifact_identity_sha256": (
                candidate.source_artifact_identity_sha256
            ),
            "run_report_sha256": candidate.run_report_sha256,
            "preregistration_sha256": candidate.preregistration_sha256,
            "artifact_path": artifact_path.relative_to(self.root).as_posix(),
            "retained_artifact_tree_sha256": require_sha256(
                artifact_report.get("tree_sha256"), "retained artifact tree"
            ),
            "retained_artifact_total_bytes": int(artifact_report["total_bytes"]),
            "evidence_path": evidence_path.relative_to(self.root).as_posix(),
            "retained_evidence_tree_sha256": require_sha256(
                evidence_report.get("tree_sha256"), "retained evidence tree"
            ),
            "fresh_reload_verified": candidate.fresh_reload_verified,
            "fresh_reload_certificate_sha256": (
                candidate.fresh_reload_certificate_sha256
            ),
            "independent_child_authentication": (
                None
                if candidate.independent_child_authentication is None
                else dict(candidate.independent_child_authentication)
            ),
            "standalone_compaction_passed": candidate.standalone_compaction_passed,
            "deployment_accepted": candidate.deployment_accepted,
            "contains_parent_parameters": False,
            "optimizer_state_retained": False,
            "source_run_deleted": False,
        }
        payload["record_sha256"] = _record_hash(payload, "record_sha256")
        return payload

    def register(
        self,
        candidate: TerminalCandidate,
        *,
        apply_prune: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(candidate, TerminalCandidate):
            raise ExactCandidateFrontierError(
                "register requires an authenticated TerminalCandidate."
            )
        if apply_prune:
            raise ExactCandidateFrontierError(
                "--apply-prune is unsupported in conservative v1; only a durable "
                "pending-prune plan can be emitted."
            )
        # Authenticate every existing byte and every protected source before the
        # first frontier directory, lock, artifact, evidence, or manifest write.
        pre_manifest = self._load_manifest()
        pre_protected = self._protected_mapping(pre_manifest, candidate)
        pre_existing = [dict(value) for value in pre_manifest["records"]]
        pre_duplicate = next(
            (
                value
                for value in pre_existing
                if value["target_contract_sha256"] == candidate.target_contract_sha256
                and value["comparison_contract_sha256"]
                == candidate.comparison_contract_sha256
                and value["artifact_kind"] == candidate.artifact_kind
                and value["content_identity_sha256"]
                == candidate.content_identity_sha256
            ),
            None,
        )
        if pre_duplicate is not None:
            return {
                "candidate_retained": True,
                "deduplicated": True,
                "record_sha256": pre_duplicate["record_sha256"],
                "apply_prune": False,
            }
        pre_sequence = _nonnegative_int(
            pre_manifest.get("next_sequence"), "next_sequence"
        )
        pre_provisional = {
            "sequence": pre_sequence,
            "target_contract_sha256": candidate.target_contract_sha256,
            "comparison_contract_sha256": candidate.comparison_contract_sha256,
            "truth_label": candidate.truth_label,
            "artifact_kind": candidate.artifact_kind,
            "content_identity_sha256": candidate.content_identity_sha256,
            "rollout_first_rank_key": list(candidate.rollout_first_rank_key),
        }
        pre_group = [
            value
            for value in pre_existing
            if value["target_contract_sha256"] == candidate.target_contract_sha256
            and value["comparison_contract_sha256"]
            == candidate.comparison_contract_sha256
        ]
        if pre_provisional not in self._rank_records(
            [*pre_group, pre_provisional], self.max_per_contract
        ):
            return {
                "candidate_retained": False,
                "deduplicated": False,
                "reason": "behavior-dominated-outside-bounded-frontier",
                "apply_prune": False,
            }
        for value in pre_protected.values():
            if _paths_overlap(self.root, Path(value)):
                raise ExactCandidateFrontierError(
                    "Frontier overlaps an immutable protected source/blank/cache path."
                )
        self._create_storage_after_authentication()
        descriptor = self._acquire_lock()
        try:
            manifest = self._load_manifest(allow_new_empty_storage=True)
            if canonical_sha256(manifest) != canonical_sha256(pre_manifest):
                raise ExactCandidateFrontierError(
                    "Frontier manifest changed between authenticated read and lock."
                )
            protected = self._protected_mapping(manifest, candidate)
            for value in protected.values():
                if _paths_overlap(self.root, Path(value)):
                    raise ExactCandidateFrontierError(
                        "Frontier overlaps an immutable blank/cache root."
                    )
            existing_records = [dict(value) for value in manifest["records"]]
            duplicate = next(
                (
                    value
                    for value in existing_records
                    if value["target_contract_sha256"]
                    == candidate.target_contract_sha256
                    and value["comparison_contract_sha256"]
                    == candidate.comparison_contract_sha256
                    and value["artifact_kind"] == candidate.artifact_kind
                    and value["content_identity_sha256"]
                    == candidate.content_identity_sha256
                ),
                None,
            )
            if duplicate is not None:
                result = {
                    "candidate_retained": True,
                    "deduplicated": True,
                    "record_sha256": duplicate["record_sha256"],
                    "apply_prune": False,
                }
                return result

            sequence = _nonnegative_int(manifest.get("next_sequence"), "next_sequence")
            provisional = {
                "sequence": sequence,
                "target_contract_sha256": candidate.target_contract_sha256,
                "comparison_contract_sha256": candidate.comparison_contract_sha256,
                "truth_label": candidate.truth_label,
                "artifact_kind": candidate.artifact_kind,
                "content_identity_sha256": candidate.content_identity_sha256,
                "rollout_first_rank_key": list(candidate.rollout_first_rank_key),
            }
            group = [
                value
                for value in existing_records
                if value["target_contract_sha256"] == candidate.target_contract_sha256
                and value["comparison_contract_sha256"]
                == candidate.comparison_contract_sha256
            ]
            selected = self._rank_records([*group, provisional], self.max_per_contract)
            if provisional not in selected:
                result = {
                    "candidate_retained": False,
                    "deduplicated": False,
                    "reason": "behavior-dominated-outside-bounded-frontier",
                    "apply_prune": False,
                }
                return result

            record_identity = candidate.submission_identity_sha256
            (
                artifact_path,
                artifact_report,
                evidence_path,
                evidence_report,
            ) = self._stage_candidate(candidate, record_identity, manifest)
            record = self._record_payload(
                candidate,
                sequence=sequence,
                artifact_path=artifact_path,
                artifact_report=artifact_report,
                evidence_path=evidence_path,
                evidence_report=evidence_report,
            )
            without_group = [value for value in existing_records if value not in group]
            selected_records = self._rank_records(
                [*group, record], self.max_per_contract
            )
            selected_hashes = {value["record_sha256"] for value in selected_records}
            evicted = [
                value
                for value in group
                if value["record_sha256"] not in selected_hashes
            ]
            pending = [dict(value) for value in manifest["pending_prune"]]
            pending_hashes = {value["record_sha256"] for value in pending}
            pending.extend(
                value
                for value in evicted
                if value["record_sha256"] not in pending_hashes
            )
            updated = {
                **manifest,
                "protected_roots": protected,
                "next_sequence": sequence + 1,
                "records": sorted(
                    [*without_group, *selected_records],
                    key=lambda value: (
                        value["target_contract_sha256"],
                        value["comparison_contract_sha256"],
                        tuple(value["rollout_first_rank_key"]),
                        int(value["sequence"]),
                    ),
                ),
                "pending_prune": pending,
                "evicted_total": int(manifest.get("evicted_total", 0)) + len(evicted),
            }
            persisted = self._write_manifest(updated)
            result = {
                "candidate_retained": True,
                "deduplicated": False,
                "record_sha256": record["record_sha256"],
                "truth_label": record["truth_label"],
                "comparison_contract_sha256": record["comparison_contract_sha256"],
                "retained_in_contract": len(selected_records),
                "pending_prune_count": len(persisted["pending_prune"]),
                "apply_prune": False,
            }
            return result
        finally:
            self._release_lock(descriptor)

    def apply_pending_prune(self) -> dict[str, Any]:
        raise ExactCandidateFrontierError(
            "Actual deletion is unsupported in conservative v1. The authenticated "
            "manifest retains a durable pending-prune plan for manual review."
        )

    def inspect(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        self._protected_mapping(manifest, None)
        groups: dict[str, int] = {}
        for record in manifest["records"]:
            key = (
                f"{record['target_contract_sha256']}:"
                f"{record['comparison_contract_sha256']}"
            )
            groups[key] = groups.get(key, 0) + 1
        return {
            "schema": FRONTIER_SCHEMA,
            "manifest_path": str(self.manifest_path),
            "retained_record_count": len(manifest["records"]),
            "pending_prune_count": len(manifest["pending_prune"]),
            "retained_per_contract": groups,
            "max_per_comparison_contract": self.max_per_contract,
            "prune_mode": "durable-plan-only-conservative-v1",
            "recursive_deletion_implemented": False,
        }


__all__ = [
    "AcceptedChildAuthenticator",
    "ExactCandidateFrontier",
    "ExactCandidateFrontierError",
    "FactorAuthenticator",
    "GUIDED_PREREGISTRATION_SCHEMA",
    "GUIDED_RUN_SCHEMA",
    "GUIDED_GUD_WINNER_PUBLICATION_PREREGISTRATION_SCHEMA",
    "GUIDED_GUD_WINNER_PUBLICATION_RUN_SCHEMA",
    "GUIDED_WINNER_PUBLICATION_PREREGISTRATION_SCHEMA",
    "GUIDED_WINNER_PUBLICATION_RUN_SCHEMA",
    "MAX_RETAINED_PER_CONTRACT",
    "MethodGateAuthenticator",
    "PreregistrationAuthenticator",
    "TerminalCandidate",
    "artifact_tree",
    "authenticate_terminal_candidate",
    "behavior_rank",
    "canonical_sha256",
    "file_sha256",
    "rollout_first_rank",
    "strict_json",
]
