#!/usr/bin/env python3
"""E1: the D3 hybrid module -- qwen3_5 text layers + the Chimera charted units at their depth +
the C15/C11 interface rows + the native media (flow) head, with the four D3 sub-stages.

WHY THIS FILE EXISTS.  DISTILL_DESIGN_2026-10-12 section 4 (D3) and section 5.2-5.6: the five
media units of OMNI1 are Chimera-only tensors (`blocks.32-36.charted.*`), the interface rows live
in Chimera's `native_media_bridge` / `native_media_head`, and the C18 music band is a slice of the
Chimera vocabulary -- none of them has a slot in the qwen3_5 layout the streamed engines train in.
So D3 trains a HYBRID: the text trunk runs as stock `Qwen3_5DecoderLayer`s fed by the S0-P1
zero-copy reader (`p1_qwen35_reader.py` >= 1.0.5, which serves OMNI1's 64 text layers out of its
69 blocks and every Chimera-only tensor as an extra), and the media parts run as the Chimera
tree's OWN modules (`ChartedSourceBlock`, `NativeContinuousMediaBridge`,
`UnifiedNativeMediaEmissionHead`) loaded from those extras, glued with the tree's own
`ChimeraOmniModel._add_native_media` / `_charted_layout` (called on a shim, never re-typed here).

WHERE THE UNITS SIT.  Read from the checkpoint, never assumed: a unit at Chimera block j reads the
output of the last text layer below j.  On OMNI1 (composite_manifest `placements`, insert arm,
"depth-fraction -> nearest gated_gqa refinement layer") ALL FIVE units are inserted after text
layer 31, so d = 32 for every unit and D3b backprops through text layers 32..63.  The names the
brief uses (ACE L12, H3 L25, QIMG L30, HY3D L10, QVL32 L13) are the SOURCE models' block indices
(the containers ace_dit_L12, h3_dit_L25_curve6, qimg_mmdit_L30_img, hy3d_dit_L10, qvl32_vit_L13),
not trunk depths.  When the manifest is present its `insert_after` must equal what the block
schedule says, or the module refuses.

THE SUB-STAGES (design section 4 D3; order `D3_STAGE_ORDER`, reconciled plan 2026-10-14 section 4):
  D3a  per-unit refit on host rows: chart_in(host row) -> the parent block's input row
       (relative MSE), plus, where the teacher cache has block in/out pairs (H3 block 25,
       QVL32 ViT block 13), the block on the parent's own input -> the parent's output delta.
       QVL32 is weighted 0.1 (design section 5.6).  No trunk layer runs.  1.1.0: chart_out is
       trained only against a DECLARED host-write target (`check_d3a_declaration`, A3 draft):
       delta_scale * (parent_out - parent_in) @ chart_out^T -> the host write recaptured on the
       D2 child.  Without a declaration chart_out is not in the D3a census (1.0.2 owned it with no
       loss term reaching it -- Codex's D3a finding).
  D3b  guided-velocity KD (prereg section 3 D3b: "mean squared error of the guided velocity").
       x_d (the hidden state entering depth d, BEFORE the units) is recaptured once with layers
       0..d-1 frozen and stored in <= 20 GB chunks (`XdChunkWriter`); a step applies the units to
       x_d, streams layers d..63 forward (no grad, boundaries on the host), runs the final norm and
       the flow head, and walks d..63 back with the weights reloaded (per-layer checkpointing, the
       T(ii) algorithm of streamed_kd_trainer.py, reused).  Trainable: the unit, the emission rows
       of its types (native_media_head.out band rows + the type_affine rows of its types), and the
       shared LoRA r16 on layers d..63.  Lane order ACE -> QIMG -> HY3D -> H3.
  D3c  the C18 music band of embed and lm_head by KL to the ACE LM's recorded top-64. OMNI1
       allocates 65,536 band rows from 252,416; the ACE LM writes 64,000 codes (rows [252,416,
       316,416)); the other 1,536 are reserved.  The KL softmax and every update run over the
       64,000 written rows only; the reserved rows (embed and head) and the ledger rows below the
       band are bit-frozen and checked after every step.  Host fp32 masters, `LazyRowAdam` (embed:
       rows of the codes in the batch; head: all 64,000 written rows) + the shared LoRA r16.
  D3d  vision-in / video-in answer KD through the QVL32 unit + the C11 rows (bridge
       feature_proj columns [640, 2176) and type_embed row 6) + the shared LoRA r16, full depth,
       over text rows [0, 248,320) with the T(ii) objective (`chunked_objective`, Q-text cache
       format) on assistant answer positions; plus the QVL32 feature KD at weight 0.1 (the unit's
       block output on its packed rows vs the ViT block-13 output, attached during the streamed
       backward, `UnitFeatureKD`).  `check_vision_alignment` / `check_answer_targets` bind the C11
       pixels, positions, feature rows and native-vocabulary answers before a step.
  D3e  speech on the H3 audio channel (SPEECH_LANE_PROPOSAL_2026-10-14 section 4, option B), after
       D3d: D3e_in (= Codex's D3d_audio_ce: clean type-3 observation rows inside the prompt, CE on
       the assistant answer, the H3 unit reads prompt rows only) then D3e_out (= Codex's
       D3d_speech_flow: x_t = (1 - sigma) x0 + sigma eps in the type-3 feature band, x0 on the
       emission band, velocity-form loss, sigma in {0.05, 0.3, 0.7, 1.0}, 20% H3 AV replay).
Every text loss is over vocab rows [0, 248,320) only (the head is sliced before the matmul, by
the trainer's engine); the D3c loss is over the written band rows only.

FLOW HEAD PARAMETRISATION x0, OBJECTIVE GUIDED VELOCITY.  The 2026-08-28 proof: pure velocity
regression pins at the target variance at small scale and does not train, so the emission band is
read as the clean endpoint x0 (`--flow-param x0`, default; Chimera's own
`native_media_target_kind="x0"`).  The OBJECTIVE is the preregistered guided-velocity MSE: the x0
prediction is mapped to velocity v = (x_t - x0_hat) / sigma (canonical v = eps - x0 for
x_t = (1 - sigma) x0 + sigma eps) with x_t read from each type's audited noisy-channel
sub-band of the FEATURE layout (`OMNI1_NOISY_FEATURE_BANDS`; the bridge and the head have different
allocations, so x_t is never assumed to sit at the emission columns), and compared with the
teacher's velocity (parent raw: v = -SIGN * raw, s8_media_smoke.SIGN; or canonical velocity; or
x0) -- `flow_velocity_loss`, loss-parity-tested against Codex's `training.flow_velocity_loss`.
1.0.2's endpoint MSE `flow_x0_loss` stays as objective `x0_endpoint` (a different objective; a
diagnostic, never the D3b lane loss without an amendment).

WHAT IS PROVEN WHERE.
  parity     OMNI1, CPU fp32, a few tokens per unit: every unit's held S5 pair segment
             (s7_omni1_check.media_segment, trimmed) through the hybrid AND through the Chimera
             reference (s7_omni1_check.Checkpoint / build_block / build_top: strict, sha-verified,
             layer-major), compared per text layer.  Functional gates: hybrid media modules
             bit-identical to the reference's; every unit's write bit-identical on the
             reference's own input; exact zeros on rows a unit does not own (every unit, every
             segment) and a non-zero write on its own segment; all finite; end-to-end relative L2
             (x_d, final hidden, own emission band, text logits rows [0, 248,320)) <= --tol-rel.
  synthetic  a tiny Chimera child built with the REAL importer + real ChartedSourceBlocks: the
             hybrid vs `ChimeraOmniModel.forward` on it, streamed grads vs full autograd, and
             D3a / D3b / D3c / D3d losses that decrease.  test_d3_hybrid_module.py beside this
             file holds the same checks as unit tests.
  rate       real OMNI1 weights on one card, lr = 0 (nothing changes, asserted): x_d recapture
             tok/s, D3b (or D3d-shaped) fwd+bwd tok/s, peak VRAM, host RSS.

STATE ACROSS SUB-STAGES (1.1.0, ported from Codex's native_state_inheritance / stage_inheritance).
Each stage ends with `export_native_state`: units ever trained (fp32 replacements), the bridge and
emission interface rows as CUMULATIVE fp32 deltas against the immutable base (inherited + this
stage's, cast once into the forward), the music tables (fp32 masters, all allocated rows, the
reserved ones bit-equal to the base) and the shared r16 LoRA bank.  The next stage imports it
with `import_native_state` before any optimiser exists (fresh Adam per stage); only the
immediate predecessor in the run's stage order is accepted.  Inside a stage,
`save_stage_checkpoint` / `load_stage_checkpoint` give exact resume: trainables, optimiser and
lazy-row states, sampler counter, RNG, and `total_steps` pinned (a checkpoint without it, or with
another value, is refused).

D4 DRIVER PREP (1.1.0).  `cache_identity` joins receipts in their AUTHENTIC producer schemas
(Codex native_flow_cache.v1: `cache_sha256` + a nested `source` identity; Codex stage caches:
`tensors_sha256`; Claude teacher-cache shards: `file_sha256`), `sigma_grid_of` reads the producer's
`sigma_grid` (or `sigmas`, never both disagreeing); `bind_native_flow_batch` (1.1.1: Codex ReadyObjectiveAdapter
guards, caller signs cross-checked against the receipt); `d4_check_optimizer_union` (one optimiser
owner per physical tensor, lr exactly 0.3 x the producer's), `d4_check_base` (a base that already
holds D2 is refused: D2 would apply twice), `d4_select_replay` (each distinct item at most once,
a short lane reports its deficit instead of duplicating), `d4_check_stage_chain` (D2 -> D3 order
with speech-in before speech-out, one shared-r16 lineage).

WHAT IS NOT HERE.  The per-teacher cache adapters (cache tensors -> native feature rows / emission
targets) for Claude's own production caches; the D3 and D4 run drivers (rungs, launch, delta
export).  The functions they call are here and tested.

Usage (server; env: CHIMERA_OMNI_SRC=<tree ef1ffb9>, PYTHONPATH=<tree>/src:<this dir>):
    $PY d3_hybrid_module.py parity --omni1 DIR --parent-config PDIR --tap-rows DIR --receipt OUT
    $PY d3_hybrid_module.py synthetic --receipt OUT
    $PY d3_hybrid_module.py rate --omni1 DIR --parent-config PDIR --tap-rows DIR --device cuda:0 \\
        --stage d3b --units ace_dit_L12,hy3d_dit_L10 --receipt OUT
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

TOOL_NAME = "d3_hybrid_module"
#: 1.0.0 (2026-10-13): first version, lane d3 (wave 2, E1).
#: 1.0.1 (2026-10-13): the rate cell is a function (`rate_cell`), CPU-tested on the tiny child before
#: it runs on a card; every cell starts from a trainable LoRA bank (1.0.0's D3d branch relied on
#: running in a fresh process); a trainable tensor that gets no gradient is a refusal.
#: 1.0.2 (2026-10-13, review fixes): BandTrainer.head_rows returns a copy (on CPU fp32 `.to` aliased
#: the master and D3c stepped on the running sum of every earlier grad); the D3d rate cell puts its KD
#: positions in answer text appended AFTER the media rows (1.0.1 scored prompt rows only, so causal
#: attention gave the unit and the bridge rows exactly zero gradient); a unit / delta / LoRA group
#: whose gradient norm is exactly 0 is a refusal, and every cell records its group grad norms.
#: 1.1.0 (2026-10-14, port of Codex's verified D3/D4 corrections, Codex branch HEAD 81f54988, reconciled
#: plan 1.3 / 4): D3b guided-velocity objective on the x0 head (`flow_velocity_loss`, rate cell
#: `--flow-objective`, default guided_velocity); D3c 64,000 written vs 65,536 allocated rows, reserved
#: and ledger rows bit-frozen; D3d QVL feature KD at 0.1 + C11 pixel / feature / answer alignment
#: checks; D3a declared host-write target for chart_out; cumulative native-state export / import and
#: exact stage checkpoints with total_steps pinned; D3e speech-in CE / speech-out flow hooks with the
#: prompt-only H3 read scope and the 20% H3 replay schedule; one type-3 position chart for D3b and
#: D3e; authentic receipt joins and D4 guards.  A second delta on one interface parameter is now
#: refused (1.0.2 kept only the last one in its functional_call dict).
#: 1.1.1 (2026-10-14, adversarial review of the 1.1.0 port): `bind_native_flow_batch` carries the Codex
#: ReadyObjectiveAdapter.bind + validate_native_batch guards 1.1.0 dropped -- a required `route` equal to
#: the receipt route, route / type dispatch, the corpus / teacher / amendment sha pins and the declared
#: head parametrisation, no weight on padding rows, sigma in (0, 1] on valid rows, no target aliasing the
#: inputs, H3 source positions, joint-AV same-origin alignment, ACE authentic code conditioning (old S8
#: fixtures put the clean x0 in source64: refused) -- and parent signs the CALLER supplies (flow_signs)
#: that must equal the receipt signs (1.1.0 took the receipt signs alone when the caller gave none).
#: `d4_select_replay` carries the Codex select_replay row rules (sealed target, source pins, pair_id =
#: digest of input / target / conditioning shas, input != target).
TOOL_VERSION = "1.1.1"
RECEIPT_SCHEMA = "neural-forge-omni-d3-hybrid-v1"
XD_SCHEMA = "neural-forge-omni-d3-xd-chunks-v1"
HERE = Path(__file__).resolve().parent
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
COMPOSITE_TOOLS = Path(os.environ.get("D3_COMPOSITE_TOOLS", str(OC / "composite" / "tools")))
MANIFEST_FILENAME = "composite_manifest.json"
#: the reader must know charted blocks (1.0.4 refuses OMNI1 at block 32)
P1_MIN_VERSION = (1, 0, 5)
#: ACE-Step 1.5 LM code ids (media_b_teacher_cache.py ACE_AUDIO_START/END: codes are stored as
#: indices into [151669, 215669)); the C18 band writes code k at band row k (c18_music_band.py)
ACE_CODES = 64_000
#: OMNI1's C18 allocation (rows [252,416, 317,952)); rows past the 64,000 written codes are reserved
OMNI1_MUSIC_ROWS = 65_536
DEFAULT_XD_CHUNK_BYTES = 20 * 10**9
#: D3 sub-stages in run order (reconciled plan 2026-10-14 section 4; speech proposal section 4.2).  The
#: optional A3 lanes run after D3e_out when they run at all.  Codex names its speech stages
#: D3d_audio_ce / D3d_speech_flow; they are the same stages.
D3_STAGE_ORDER = ("D3a", "D3b.ace", "D3b.qimg", "D3b.hy3d", "D3b.h3", "D3c", "D3d", "D3e_in", "D3e_out")
D3_OPTIONAL_STAGES = ("D3f.image_edit", "D3f.temporal_video")
STAGE_ALIASES = {"D3d_audio_ce": "D3e_in", "D3d_speech_flow": "D3e_out"}
D3B_LANE_SOURCE = {"D3b.ace": "ace", "D3b.qimg": "qimg", "D3b.hy3d": "hy3d", "D3b.h3": "h3"}
#: per-type noisy-latent sub-band of the FEATURE (bridge) layout, Codex's audited allocation
#: (artifacts/chimera_recovery/2026-10-13/native_noisy_feature_bands.json, checked against OMNI1's
#: composite_manifest interface bands on 2026-10-14: ACE's feature band is [source 128 | chunk mask 128 |
#: noisy 128]; every other generated type's whole feature band is its noisy latent)
OMNI1_NOISY_FEATURE_BANDS = {4: (256, 384), 2: (384, 480), 3: (480, 512), 1: (512, 576), 7: (576, 640)}
#: the tiny child's analogue (the last emission-width columns of each generated type's feature band)
TINY_NOISY_FEATURE_BANDS = {4: (6, 12), 2: (16, 20), 3: (22, 24), 7: (28, 32)}
FLOW_OBJECTIVES = ("guided_velocity", "x0_endpoint")
#: D3d: QVL32 feature KD weight against ViT block 13 (prereg section 3 D3d)
QVL_FEATURE_WEIGHT = 0.1
#: D3e_out noise levels (Codex speech amendment cf676e8d, speech proposal section 2.4)
SPEECH_SIGMAS = (0.05, 0.3, 0.7, 1.0)
#: D3e_out mixes this fraction of H3 AV replay from the D3b cache (speech proposal section 4.2)
H3_REPLAY_FRACTION = 0.2
#: D4: every D2 / D3 trainable at producer lr x 0.3 (prereg section 3 D4); per-modality pair quota
D4_LR_MULTIPLIER = "0.3"
D4_PAIR_QUOTA = 300
D4_PRIOR_ORDER = ("D2",) + D3_STAGE_ORDER
#: D4 optimiser families (Codex consolidation.OPTIMIZER_FAMILIES): each physical tensor has one owner
D4_OPTIMIZER_FAMILIES = ("D2.lora", "D2.k3_lora", "D2.gain", "D3.unit", "D3.emission", "D3.bridge", "D3.music_embed",
                         "D3.music_head", "D3.shared_lora")
NATIVE_STATE_SCHEMA = "neural-forge-omni-d3-native-state-v1"
STAGE_CHECKPOINT_SCHEMA = "neural-forge-omni-d3-stage-checkpoint-v1"


class D3Error(RuntimeError):
    """A structural refusal: wrong layout, missing tensor, non-finite value, failed gate."""


def log(message: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tool_identity() -> dict[str, Any]:
    return {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": file_sha256(Path(__file__).resolve())}


def tensor_sha256(tensor: torch.Tensor) -> str:
    """sha256 over dtype, shape and the contiguous CPU bytes (bit-identity of a tensor)."""

    value = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256(f"{value.dtype}|{tuple(value.shape)}|".encode())
    digest.update(value.view(torch.uint8).numpy().tobytes() if value.numel() else b"")
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """sha256 of canonical JSON (sorted keys, compact, UTF-8, no NaN) -- the digest Codex's
    `training.digest` computes, so receipt digests such as noisy_feature_bands_sha256 re-check."""

    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def canonical_stage(name: str) -> str:
    return STAGE_ALIASES.get(str(name), str(name))


def stage_order(optional: tuple[str, ...] | list[str] = ()) -> tuple[str, ...]:
    """The run's D3 stage order: `D3_STAGE_ORDER` then the optional A3 lanes that run, in their order."""

    extra = [canonical_stage(s) for s in optional]
    unknown = [s for s in extra if s not in D3_OPTIONAL_STAGES]
    if unknown or len(set(extra)) != len(extra):
        raise D3Error(f"optional D3 stages {list(optional)} are not distinct members of {D3_OPTIONAL_STAGES}")
    return D3_STAGE_ORDER + tuple(s for s in D3_OPTIONAL_STAGES if s in extra)


def predecessor(stage: str, order: tuple[str, ...] = D3_STAGE_ORDER) -> str | None:
    stage = canonical_stage(stage)
    if stage not in order:
        raise D3Error(f"stage {stage!r} is not in the run order {list(order)}")
    index = order.index(stage)
    return order[index - 1] if index else None


def audited_noisy_bands(bands: dict[int, dict[str, Any]]) -> dict[int, tuple[int, int]]:
    """The audited per-type noisy-latent sub-band of the FEATURE layout for these interface bands:
    OMNI1's (`OMNI1_NOISY_FEATURE_BANDS`) or the tiny child's, chosen by which table fits.  Every
    entry must lie inside its type's feature band and be exactly as wide as its emission band (x_t
    and x0 then share a channel order).  No table fits -> refusal (velocity conversion would read the
    wrong channels)."""

    for table in (OMNI1_NOISY_FEATURE_BANDS, TINY_NOISY_FEATURE_BANDS):
        if not set(table) <= set(bands):
            continue
        ok = True
        for t, (lo, hi) in table.items():
            flo, fhi = bands[t]["feature"]
            elo, ehi = bands[t]["emission"]
            if not (flo <= lo < hi <= fhi and hi - lo == ehi - elo):
                ok = False
                break
        if ok:
            return dict(table)
    raise D3Error("no audited noisy-channel suballocation matches these interface bands")


# --------------------------------------------------------------------------
# Peers.  Every contract this file consumes is read through its own tool:
# the streamed engine + LoRA bank + objective from streamed_kd_trainer.py, the
# qwen3_5 map from p1_qwen35_reader.py (through the trainer's loader, so both
# see one module object), the Chimera reference and the S5 pair segments from
# composite/tools/s7_omni1_check.py, the flow signs from s8_media_smoke.py.
# --------------------------------------------------------------------------


def _load_by_path(name: str, path: Path) -> Any:
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(path):
        return sys.modules[name]
    if not path.is_file():
        raise D3Error(f"peer tool {path} is missing")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def trainer_module() -> Any:
    path = Path(os.environ.get("STREAMED_KD_TRAINER_PATH", str(HERE / "streamed_kd_trainer.py")))
    return _load_by_path("streamed_kd_trainer", path)


def p1_module() -> Any:
    module = trainer_module().load_peer_module("p1_qwen35_reader")
    version = str(getattr(module, "TOOL_VERSION", "")).split()[-1]
    try:
        parts = tuple(int(p) for p in version.split("."))
    except ValueError as error:
        raise D3Error(f"p1_qwen35_reader version {version!r} is not numeric") from error
    if parts < P1_MIN_VERSION:
        raise D3Error(
            f"p1_qwen35_reader {version} has no charted-block support (needs >= "
            f"{'.'.join(map(str, P1_MIN_VERSION))}: OMNI1's text layer t is not blocks.t)"
        )
    return module


def composite_tool(name: str) -> Any:
    return _load_by_path(f"{name}_d3", COMPOSITE_TOOLS / f"{name}.py")


def peer_record(module: Any) -> dict[str, Any]:
    path = Path(module.__file__).resolve()
    return {"path": str(path), "sha256": file_sha256(path), "version": getattr(module, "TOOL_VERSION", None)}


def flow_signs() -> dict[int, float]:
    """x0 = x_t + SIGN * sigma * raw per media type id, from s8_media_smoke.SIGN (its comments
    say which parent output convention each sign comes from)."""

    s8 = composite_tool("s8_media_smoke")
    by_name = dict(s8.SIGN)
    return {4: by_name["ace"], 1: by_name["qimg"], 7: by_name["hy3d"], 2: by_name["h3_video"], 3: by_name["h3_audio"]}


# --------------------------------------------------------------------------
# Layout: where each unit sits, which bands each type owns.
# --------------------------------------------------------------------------


@dataclass
class UnitPlacement:
    block: int                  # Chimera block index (OMNI1: 32..36)
    container: str              # manifest container (ace_dit_L12, ...) or the spec's component
    source: str                 # spec.source (ace, h3, hy3d, qimg, qvl32)
    type_ids: tuple[int, ...]
    after_text_layer: int       # the text layer whose output the unit reads; -1 = the embedding
    spec: Any = field(repr=False, default=None)

    @property
    def depth(self) -> int:
        """d: the first text layer ABOVE the unit (D3b backprops through d .. L-1)."""

        return self.after_text_layer + 1


def read_manifest(directory: Path) -> dict[str, Any] | None:
    path = Path(directory) / MANIFEST_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def unit_layout(reader: Any, cfg: Any, manifest: dict[str, Any] | None) -> list[UnitPlacement]:
    """Placements in Chimera block order, cross-checked against the manifest when present."""

    text_blocks = list(reader.text_blocks)
    placements = {int(p["layer_index"]): p for p in ((manifest or {}).get("blocks") or {}).get("placements", [])}
    trunk_map = {int(k): int(v) for k, v in ((manifest or {}).get("blocks") or {}).get("trunk_layer_map", {}).items()}
    if trunk_map and [trunk_map[t] for t in range(len(text_blocks))] != text_blocks:
        raise D3Error("composite_manifest trunk_layer_map disagrees with the reader's text_blocks")
    out: list[UnitPlacement] = []
    for block in reader.charted_blocks:
        below = [t for t, b in enumerate(text_blocks) if b < block]
        after = below[-1] if below else -1
        spec = cfg.charted_spec(block)
        container = str(spec.component)
        if placements:
            if block not in placements:
                raise D3Error(f"charted block {block} has no manifest placement")
            placed = placements[block]
            container = str(placed["container"])
            declared = int((placed.get("pairing") or {}).get("insert_after", after))
            if declared != after:
                raise D3Error(f"{container}: manifest insert_after {declared}, block schedule says {after}")
            if sorted(int(t) for t in placed["type_ids"]) != sorted(int(t) for t in spec.type_ids):
                raise D3Error(f"{container}: manifest type_ids {placed['type_ids']} != spec {list(spec.type_ids)}")
        out.append(UnitPlacement(block=int(block), container=container, source=str(spec.source),
                                 type_ids=tuple(int(t) for t in spec.type_ids), after_text_layer=int(after), spec=spec))
    owners: dict[int, str] = {}
    for unit in out:
        for t in unit.type_ids:
            if t in owners:
                raise D3Error(f"type {t} is owned by both {owners[t]} and {unit.container}")
            owners[t] = unit.container
    return out


def interface_bands(manifest: dict[str, Any] | None) -> dict[int, dict[str, tuple[int, int]]]:
    """type id -> {"feature": (lo, hi), "emission": (lo, hi)} from the manifest's interface entries."""

    bands: dict[int, dict[str, tuple[int, int]]] = {}
    for entry in ((manifest or {}).get("interface") or {}).get("entries", []) or []:
        t = int(entry["type_id"])
        bands[t] = {"feature": (int(entry["feature_band"][0]), int(entry["feature_band"][1])),
                    "emission": (int(entry["emission_band"][0]), int(entry["emission_band"][1])),
                    "name": entry.get("name"), "set": entry.get("set")}
    return bands


# --------------------------------------------------------------------------
# Module construction from P1 extras (Chimera names).
# --------------------------------------------------------------------------


def load_strict_from(module: nn.Module, fetch: Callable[[str], torch.Tensor], prefix: str) -> nn.Module:
    """`load_state_dict(strict=True)` with every tensor fetched by `prefix + name` and cast to the
    dtype its reference parameter already has (so the H3 fp32 islands stay fp32)."""

    state = {}
    for name, reference in module.state_dict().items():
        value = fetch(prefix + name)
        if tuple(value.shape) != tuple(reference.shape):
            raise D3Error(f"{prefix}{name}: stored {tuple(value.shape)} != module {tuple(reference.shape)}")
        state[name] = value.to(device=reference.device, dtype=reference.dtype if reference.is_floating_point() else value.dtype)
    module.load_state_dict(state, strict=True)
    return module


def build_unit(reader: Any, placement: UnitPlacement, hidden: int, *, device: str, dtype: torch.dtype) -> nn.Module:
    from chimera_omni.modeling.charted import ChartedSourceBlock

    unit = ChartedSourceBlock(placement.spec, hidden).to(device=device, dtype=dtype)
    return load_strict_from(unit, reader.extra, f"blocks.{placement.block}.charted.").eval()


def build_bridge(reader: Any, cfg: Any, *, device: str, dtype: torch.dtype) -> nn.Module:
    from chimera_omni.modeling.chimera import NativeContinuousMediaBridge

    bridge = NativeContinuousMediaBridge(cfg).to(device=device, dtype=dtype)
    return load_strict_from(bridge, reader.extra, "native_media_bridge.").eval().requires_grad_(False)


def build_media_head(reader: Any, cfg: Any, *, device: str, dtype: torch.dtype) -> nn.Module:
    from chimera_omni.modeling.chimera import UnifiedNativeMediaEmissionHead

    head = UnifiedNativeMediaEmissionHead(cfg).to(device=device, dtype=dtype)
    return load_strict_from(head, reader.extra, "native_media_head.").eval().requires_grad_(False)


class _BridgeCall(nn.Module):
    """`functional_call` target: the bridge's own `embed`, so a delta never re-types its code."""

    def __init__(self, bridge: nn.Module) -> None:
        super().__init__()
        self.bridge = bridge

    def forward(self, *args: Any) -> torch.Tensor:
        return self.bridge.embed(*args)


class _HeadCall(nn.Module):
    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.head = head

    def forward(self, hidden: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        return self.head(hidden, type_ids)


class SliceDelta(nn.Module):
    """A trainable additive delta on row (axis 0) or column (axis 1) ranges of a frozen weight.

    The master is fp32 and starts at exactly zero, so a fresh delta changes nothing; `full(base)`
    returns base + delta scattered into its ranges, differentiable with respect to the delta.
    Exported later as the trained subset of the named tensor (oc_checkpoint --delta-base)."""

    def __init__(self, name: str, shape: tuple[int, int], ranges: list[tuple[int, int]], axis: int,
                 device: str | torch.device) -> None:
        super().__init__()
        if axis not in (0, 1):
            raise D3Error("SliceDelta axis must be 0 (rows) or 1 (columns)")
        self.name = name
        self.shape = tuple(int(s) for s in shape)
        self.ranges = [(int(a), int(b)) for a, b in ranges]
        self.axis = axis
        for a, b in self.ranges:
            if not 0 <= a < b <= self.shape[axis]:
                raise D3Error(f"{name}: range [{a}, {b}) outside axis {axis} of {self.shape}")
        width = sum(b - a for a, b in self.ranges)
        other = self.shape[1 - axis]
        delta_shape = (width, other) if axis == 0 else (other, width)
        self.delta = nn.Parameter(torch.zeros(delta_shape, dtype=torch.float32, device=device))

    def scatter(self, device: str | torch.device | None = None) -> torch.Tensor:
        """The delta as an fp32 tensor of the full weight's shape (zeros off its ranges)."""

        scattered = torch.zeros(self.shape, dtype=torch.float32, device=self.delta.device if device is None else device)
        offset = 0
        for a, b in self.ranges:
            n = b - a
            if self.axis == 0:
                scattered[a:b] = self.delta[offset:offset + n]
            else:
                scattered[:, a:b] = self.delta[:, offset:offset + n]
            offset += n
        return scattered

    def full(self, base: torch.Tensor) -> torch.Tensor:
        return base + self.scatter(base.device).to(base.dtype)

    def descriptor(self, kind: str) -> dict[str, Any]:
        return {"kind": kind, "name": self.name, "shape": list(self.shape), "axis": self.axis, "ranges": [list(r) for r in self.ranges]}


def compose_interface(base: torch.Tensor, inherited: torch.Tensor | None, delta: SliceDelta | None) -> torch.Tensor:
    """base + (inherited fp32 delta + this stage's delta), cast ONCE to the base dtype (Codex's
    native_state_inheritance rule: earlier stages are never folded into a low-precision base)."""

    if inherited is None:
        return base if delta is None else delta.full(base)
    total = inherited.to(base.device, torch.float32)
    if delta is not None:
        total = total + delta.scatter(base.device)
    return base + total.to(base.dtype)


# --------------------------------------------------------------------------
# Lazy row Adam (host fp32 masters of the C18 band rows).
# --------------------------------------------------------------------------


class LazyRowAdam:
    """Adam over the rows of a host fp32 table, updating only the rows a step names.

    Each row keeps its own step count, so its bias correction is the one a dense Adam would have
    used had the row been updated only on the steps it was (TF LazyAdam / torch SparseAdam
    semantics).  When every row is named every step this is exactly dense Adam (tested).  The
    caller passes unique row ids and their summed gradients."""

    def __init__(self, table: torch.Tensor, *, lr: float, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        if table.dtype != torch.float32 or table.device.type != "cpu":
            raise D3Error("LazyRowAdam keeps an fp32 host master")
        self.table = table
        self.lr = float(lr)
        self.beta1, self.beta2 = (float(b) for b in betas)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.m = torch.zeros_like(table)
        self.v = torch.zeros_like(table)
        self.steps = torch.zeros(table.shape[0], dtype=torch.int64)
        self.rows_updated = 0

    @torch.no_grad()
    def step(self, rows: torch.Tensor, grads: torch.Tensor) -> None:
        rows = rows.to(device="cpu", dtype=torch.long)
        grads = grads.to(device="cpu", dtype=torch.float32)
        if rows.ndim != 1 or grads.shape != (rows.numel(), self.table.shape[1]):
            raise D3Error(f"LazyRowAdam.step wants rows [n] and grads [n, {self.table.shape[1]}]")
        if rows.numel() != torch.unique(rows).numel():
            raise D3Error("LazyRowAdam.step rows must be unique (sum duplicate gradients first)")
        if not bool(torch.isfinite(grads).all()):
            raise D3Error("LazyRowAdam.step got a non-finite gradient")
        t = self.steps[rows] + 1
        self.steps[rows] = t
        m = self.m[rows].mul_(self.beta1).add_(grads, alpha=1.0 - self.beta1)
        v = self.v[rows].mul_(self.beta2).addcmul_(grads, grads, value=1.0 - self.beta2)
        self.m[rows] = m
        self.v[rows] = v
        tf = t.to(torch.float64)
        c1 = (1.0 - self.beta1 ** tf).to(torch.float32).unsqueeze(1)
        c2 = (1.0 - self.beta2 ** tf).to(torch.float32).unsqueeze(1)
        update = (m / c1) / ((v / c2).sqrt() + self.eps)
        current = self.table[rows]
        if self.weight_decay:
            current.mul_(1.0 - self.lr * self.weight_decay)
        self.table[rows] = current - self.lr * update
        self.rows_updated += int(rows.numel())

    def state(self) -> dict[str, torch.Tensor]:
        return {"m": self.m, "v": self.v, "steps": self.steps}


def unique_row_grads(rows: torch.Tensor, grads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum gradients of repeated rows: (unique rows, [n_unique, H] summed grads)."""

    unique, inverse = torch.unique(rows, return_inverse=True)
    summed = torch.zeros(unique.numel(), grads.shape[-1], dtype=torch.float32, device=grads.device)
    summed.index_add_(0, inverse.to(grads.device), grads.float())
    return unique, summed


# --------------------------------------------------------------------------
# x_d chunk store (D3b).
# --------------------------------------------------------------------------


class XdChunkWriter:
    """x_d samples ([L, H] each, stored in `dtype`) packed into safetensors chunks of at most
    `max_chunk_bytes`, with a sealed index (sha256 per chunk, the identity the x_d depends on:
    base checkpoint, LoRA state of the layers below d, inputs).  A sample bigger than the cap
    gets a chunk of its own (recorded).  The directory must be new or empty."""

    def __init__(self, directory: str | Path, *, identity: dict[str, Any], max_chunk_bytes: int = DEFAULT_XD_CHUNK_BYTES,
                 dtype: torch.dtype = torch.bfloat16) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise D3Error(f"{self.directory} is not empty; an x_d store is written once")
        self.identity = identity
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.dtype = dtype
        self.buffer: dict[str, torch.Tensor] = {}
        self.buffer_bytes = 0
        self.chunks: list[dict[str, Any]] = []
        self.entries: dict[str, dict[str, Any]] = {}
        self.oversize: list[str] = []

    def add(self, key: str, xd: torch.Tensor, meta: dict[str, Any] | None = None) -> None:
        if key in self.entries or key in self.buffer:
            raise D3Error(f"x_d key {key!r} written twice")
        if xd.ndim != 2:
            raise D3Error("x_d samples are [L, H]")
        tensor = xd.detach().to(device="cpu", dtype=self.dtype).contiguous()
        if not bool(torch.isfinite(tensor.float()).all()):
            raise D3Error(f"x_d {key}: non-finite")
        nbytes = tensor.numel() * tensor.element_size()
        if self.buffer and self.buffer_bytes + nbytes > self.max_chunk_bytes:
            self.flush()
        if nbytes > self.max_chunk_bytes:
            self.oversize.append(key)
        self.buffer[key] = tensor
        self.buffer_bytes += nbytes
        self.entries[key] = {"chunk": None, "shape": list(tensor.shape), "meta": meta or {}}

    def flush(self) -> None:
        if not self.buffer:
            return
        from safetensors.torch import save_file

        name = f"chunk-{len(self.chunks):05d}.safetensors"
        path = self.directory / name
        save_file(self.buffer, str(path), metadata={"schema": XD_SCHEMA})
        sha = file_sha256(path)
        self.chunks.append({"file": name, "sha256": sha, "bytes": path.stat().st_size, "keys": sorted(self.buffer)})
        for key in self.buffer:
            self.entries[key]["chunk"] = name
        self.buffer = {}
        self.buffer_bytes = 0

    def close(self) -> dict[str, Any]:
        self.flush()
        index = {"schema": XD_SCHEMA, "tool": tool_identity(), "identity": self.identity, "dtype": str(self.dtype).replace("torch.", ""),
                 "max_chunk_bytes": self.max_chunk_bytes, "chunks": self.chunks, "entries": self.entries,
                 "oversize_samples": self.oversize, "samples": len(self.entries)}
        trainer_module().write_json_atomic(self.directory / "index.json", index)
        return index


class XdChunkReader:
    def __init__(self, directory: str | Path, *, expect_identity: dict[str, Any] | None = None, verify: bool = True) -> None:
        self.directory = Path(directory)
        self.index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        if self.index.get("schema") != XD_SCHEMA:
            raise D3Error(f"{self.directory} is not an x_d store")
        if expect_identity is not None and self.index["identity"] != expect_identity:
            raise D3Error("x_d store identity differs from this run's (base / lower-layer LoRA / inputs changed: recapture)")
        if verify:
            for chunk in self.index["chunks"]:
                if file_sha256(self.directory / chunk["file"]) != chunk["sha256"]:
                    raise D3Error(f"{chunk['file']} does not hash like its index")
        self._handles: dict[str, Any] = {}

    def keys(self) -> list[str]:
        return sorted(self.index["entries"])

    def get(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        entry = self.index["entries"].get(key)
        if entry is None:
            raise D3Error(f"x_d key {key!r} is not in the store")
        name = entry["chunk"]
        if name not in self._handles:
            self._handles[name] = safe_open(str(self.directory / name), framework="pt")
        return self._handles[name].get_tensor(key)


# --------------------------------------------------------------------------
# The hybrid.
# --------------------------------------------------------------------------


@dataclass
class HybridOptions:
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    head_device: str | None = None       # text head rows [0, base) (LayerwiseEngine); default = device
    mode: str = "streamed"                # LayerwiseEngine: streamed | resident
    attention: str = "sdpa"
    kernels: str = "torch"                # torch (CPU / Pascal / parity) or auto (fla on the 3090s)
    unit_dtype: torch.dtype = torch.float32  # units keep fp32 masters (trainable in D3a/b/d)
    lora: Any = None                      # streamed_kd_trainer.SitePlanConfig or None
    lora_seed: int = 0
    plan: str | None = None               # sealed oc_checkpoint delta-compose plan (P1 --plan)
    pin_boundaries: bool = True
    prefetch: bool = True


class HybridD3:
    """OMNI1 (or a delta child) as qwen3_5 layers + Chimera units / bridge / flow head.

    All rows must be valid (no padding): one sample per call, as the parity arm and the D3 rungs
    use it.  Chimera's per-block `torch.where(token_valid, ...)` is then the identity."""

    def __init__(self, omni_dir: str | Path, parent_config_dir: str | Path, options: HybridOptions | None = None) -> None:
        self.options = options or HybridOptions()
        o = self.options
        self.T = trainer_module()
        self.T.select_linear_attention_kernels(o.kernels)
        p1_module()  # version gate before the trainer opens it
        self.omni_dir = Path(omni_dir)
        self.source = self.T.P1Source(self.omni_dir, Path(parent_config_dir), Path(o.plan) if o.plan else None)
        self.reader = self.source.reader
        if not hasattr(self.reader, "text_blocks"):
            raise D3Error("the P1 reader has no text_blocks (needs >= 1.0.5)")
        from chimera_omni.config import ChimeraConfig
        from chimera_omni.modeling.chimera import ChimeraOmniModel

        self._chimera_model = ChimeraOmniModel
        self.cfg = ChimeraConfig.from_manifest(self.reader.manifest["config"])
        self.manifest = read_manifest(self.omni_dir)
        self.layout = unit_layout(self.reader, self.cfg, self.manifest)
        self.bands = interface_bands(self.manifest)
        self.base_vocab = int(self.reader.base_vocab)
        self.hidden = int(self.reader.hidden)
        self.device = torch.device(o.device)
        self.dtype = o.dtype
        self.text_config = self.T.text_config_from_source(self.source, o.dtype, o.attention)
        self.num_layers = int(self.text_config.num_hidden_layers)
        self.bank = None
        if o.lora is not None:
            plan = self.T.build_site_plan(self.text_config, o.lora)
            self.bank = self.T.AdapterBank(plan, [str(t) for t in self.text_config.layer_types[: self.num_layers]],
                                           o.device, o.lora_seed)
        self.engine = self.T.LayerwiseEngine(
            self.source, self.text_config, mode=o.mode, dtype=o.dtype, layer_devices=[o.device],
            head_device=o.head_device or o.device, bank=self.bank, text_vocab_rows=self.base_vocab,
            pin_boundaries=o.pin_boundaries, prefetch=o.prefetch,
        )
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        self.final_norm = Qwen3_5RMSNorm(self.text_config.hidden_size, eps=self.text_config.rms_norm_eps).to(self.device, o.dtype)
        with torch.no_grad():
            self.final_norm.weight.copy_(self.source.get(f"{self.T.DEFAULT_TEXT_PREFIX}.norm.weight"))
        self.final_norm.requires_grad_(False)
        self.units: dict[str, nn.Module] = {
            p.container: build_unit(self.reader, p, self.hidden, device=o.device, dtype=o.unit_dtype).requires_grad_(False)
            for p in self.layout
        }
        self.bridge = build_bridge(self.reader, self.cfg, device=o.device, dtype=o.dtype)
        self.media_head = build_media_head(self.reader, self.cfg, device=o.device, dtype=o.dtype)
        self._bridge_call = _BridgeCall(self.bridge)
        self._head_call = _HeadCall(self.media_head)
        self.bridge_deltas: list[SliceDelta] = []
        self.head_deltas: list[SliceDelta] = []
        #: cumulative fp32 interface deltas inherited from earlier D3 stages (`import_native_state`),
        #: keyed by parameter name ("feature_proj.weight", ...); full weight shape, zeros off their ranges
        self.inherited_bridge: dict[str, torch.Tensor] = {}
        self.inherited_head: dict[str, torch.Tensor] = {}
        self.inherited_descriptors: dict[str, dict[str, Any]] = {}
        self.inherited_units: set[str] = set()
        #: the band rows (host views of the extras; D3c swaps in trainable masters)
        self.band_embed = self.reader.extra("embed.weight[base:]")
        self.band_head = self.reader.extra("lm_head.weight[base:]") if self.reader.has("lm_head.weight") else None
        self.band_embed_override: torch.Tensor | None = None
        #: inherited fp32 music head rows (all allocated rows from music begin), set by `import_native_state`
        self.inherited_band_head: torch.Tensor | None = None
        self.native_parent: str | None = None
        self.media_placeholder = int(self.cfg.ledger().range("media").begin)
        self.music_band_begin = int(self.cfg.ledger().range("music_codes").begin)
        self.music_band_end = int(self.cfg.ledger().range("music_codes").end)

    # -- music band contract (D3c) -------------------------------------------

    @property
    def music_offset(self) -> int:
        """Band-table row of code 0 (the band tables start at base_vocab)."""

        return self.music_band_begin - self.base_vocab

    @property
    def music_rows(self) -> int:
        """Allocated music rows (OMNI1: 65,536), of which the first `ACE_CODES` are written."""

        return self.music_band_end - self.music_band_begin

    def music_contract(self, code_rows: int) -> dict[str, Any]:
        """Row bookkeeping + digests of everything D3c must NOT move (ledger rows below the band, the
        reserved rows past the written codes, in embed and head), on the BASE tables."""

        if not 1 <= int(code_rows) <= self.music_rows:
            raise D3Error(f"written code rows {code_rows} outside the {self.music_rows} allocated music rows")
        if self.band_head is None:
            raise D3Error("D3c needs an untied lm_head (the band head rows)")
        o, c = self.music_offset, int(code_rows)
        if self.band_embed.shape[0] < o + self.music_rows:
            raise D3Error("band tables end before the music band's ledger end")
        embed, head = self.band_embed.float(), self.band_head.float()
        return {"begin": self.music_band_begin, "end": self.music_band_end, "offset": o, "allocated_rows": self.music_rows,
                "written_rows": c, "reserved_rows": self.music_rows - c,
                "ledger_embed_sha256": tensor_sha256(embed[:o]), "ledger_head_sha256": tensor_sha256(head[:o]),
                "reserved_embed_sha256": tensor_sha256(embed[o + c:o + self.music_rows]),
                "reserved_head_sha256": tensor_sha256(head[o + c:o + self.music_rows])}

    def noisy_feature_bands(self) -> dict[int, tuple[int, int]]:
        """The audited noisy sub-band of each generated type's FEATURE band (`audited_noisy_bands`)."""

        return audited_noisy_bands(self.bands)

    # -- identity ------------------------------------------------------------

    def identity(self) -> dict[str, Any]:
        return {"source": self.source.identity(), "layout": [dataclasses.asdict(dataclasses.replace(p, spec=None)) for p in self.layout],
                "bands": {str(k): v for k, v in self.bands.items()}, "trainer": peer_record(self.T),
                "p1": peer_record(sys.modules["p1_qwen35_reader"]), "options": {
                    "dtype": str(self.dtype), "device": str(self.device), "mode": self.options.mode,
                    "kernels": self.options.kernels, "unit_dtype": str(self.options.unit_dtype),
                    "lora": None if self.options.lora is None else dataclasses.asdict(self.options.lora)}}

    def units_after(self, layer: int) -> list[UnitPlacement]:
        return [p for p in self.layout if p.after_text_layer == layer]

    # -- inputs --------------------------------------------------------------

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Token rows: [0, base) from the P1 text table, >= base from the band rows."""

        ids = ids.to("cpu", torch.long)
        text = F.embedding(ids.clamp(max=self.base_vocab - 1), self.engine.embed_weight)
        band = ids >= self.base_vocab
        if bool(band.any()):
            table = self.band_embed_override if self.band_embed_override is not None else self.band_embed
            text[band] = table[ids[band] - self.base_vocab].to(text.dtype)
        return text.to(device=self.device, dtype=self.dtype)

    def interface_params(self, kind: str) -> dict[str, torch.Tensor]:
        """`functional_call` overrides for the bridge ("bridge") or the flow head ("head"): every
        parameter with an inherited cumulative delta and / or a current-stage delta, composed by
        `compose_interface`.  Empty when neither exists (the module runs on its own weights)."""

        call = self._bridge_call if kind == "bridge" else self._head_call
        inherited = self.inherited_bridge if kind == "bridge" else self.inherited_head
        deltas = {d.name: d for d in (self.bridge_deltas if kind == "bridge" else self.head_deltas)}
        if not deltas and not inherited:
            return {}
        named = dict(call.named_parameters())
        return {f"{kind}.{name}": compose_interface(named[f"{kind}.{name}"], inherited.get(name), deltas.get(name))
                for name in sorted(set(deltas) | set(inherited))}

    def _shim(self) -> SimpleNamespace:
        bridge = self._bridge_call
        hybrid = self

        class _Embedder:
            def embed(_self, *args: Any) -> torch.Tensor:
                params = hybrid.interface_params("bridge")
                if not params:
                    return bridge(*args)
                return torch.func.functional_call(bridge, params, args)

        return SimpleNamespace(has_charted_blocks=bool(self.layout), native_media_bridge=_Embedder(), cfg=self.cfg)

    def prepare(self, segment: dict[str, Any], *, media_grad: bool = False) -> dict[str, Any]:
        """Embedding + native media (Chimera's own `_add_native_media` / `_charted_layout`)."""

        ids = segment["ids"].to(self.device)
        if ids.shape[0] != 1:
            raise D3Error("the hybrid runs one sample per call")
        x = self.embed(ids)
        shim = self._shim()
        model = self._chimera_model
        features = segment.get("features")
        context: dict[str, Any] = {"ids": ids, "valid": torch.ones(ids.shape, dtype=torch.bool, device=self.device),
                                   "row_types": None, "levels": None, "positions": None, "indices": None, "types": None,
                                   "media_rows": 0}
        if features is not None:
            types = segment["types"].to(self.device)
            valid_media = torch.ones(types.shape, dtype=torch.bool, device=self.device)
            sigma = segment.get("sigma")
            noise = None if sigma is None else sigma.to(self.device)
            indices = segment["indices"].to(self.device)
            with torch.set_grad_enabled(media_grad):
                x, count = model._add_native_media(shim, x, features.to(self.device), types,
                                                   segment["positions"].to(self.device), indices, valid_media, noise)
            row_types, levels = model._charted_layout(shim, ids, count, indices, types, valid_media, noise)
            context.update({"row_types": row_types, "levels": levels, "indices": indices, "types": types, "media_rows": int(count)})
            positions = segment.get("source_positions")
            context["positions"] = None if positions is None else {k: v.to(self.device) for k, v in positions.items()}
        context["x"] = x
        return context

    # -- units ---------------------------------------------------------------

    def unit_write(self, placement: UnitPlacement, x: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        unit = self.units[placement.container]
        positions = None if context["positions"] is None else context["positions"].get(placement.spec.source)
        x_unit = x.to(self.options.unit_dtype)
        write = unit(x_unit, context["row_types"], context["levels"], context["valid"], source_positions=positions)
        return write.to(x.dtype)

    def apply_units(self, x: torch.Tensor, after_layer: int, context: dict[str, Any],
                    writes: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        for placement in self.units_after(after_layer):
            write = self.unit_write(placement, x, context)
            if writes is not None:
                writes[placement.container] = write.detach()
            x = x + write
        return x

    # -- layers --------------------------------------------------------------

    def contexts(self, x: torch.Tensor) -> dict[str, dict[str, Any]]:
        return self.engine._context(int(x.shape[0]), int(x.shape[1]))

    def run_layers(self, x: torch.Tensor, start: int, stop: int, context: dict[str, Any], *,
                   bounds: list[torch.Tensor] | None = None, pre_units: dict[int, torch.Tensor] | None = None,
                   writes: dict[str, torch.Tensor] | None = None, per_layer: Callable[[int, torch.Tensor], None] | None = None,
                   units_at_stop: bool = True) -> torch.Tensor:
        """Text layers [start, stop) under no_grad, the units placed after each of them applied.

        `bounds[i - start]` = the input of layer i (host copy); `pre_units[i]` = layer i's output
        before the units that follow it (host copy), for the streamed backward.  With
        `units_at_stop=False` the units after layer stop-1 are NOT applied: that is how x_d (the
        hidden entering depth d, before the units) is captured."""

        contexts = self.contexts(x)
        with torch.no_grad():
            for i in range(start, stop):
                if bounds is not None:
                    bounds.append(x.detach().to("cpu", copy=True))
                layer = self.engine.layer(i, i + 1 if i + 1 < stop else None)
                x = self.engine._run(layer, i, x, contexts)
                if self.units_after(i) and (units_at_stop or i != stop - 1):
                    if pre_units is not None:
                        pre_units[i] = x.detach().to("cpu", copy=True)
                    x = self.apply_units(x, i, context, writes)
                if per_layer is not None:
                    per_layer(i, x)
        return x

    def backward_layers(self, grad: torch.Tensor, bounds: list[torch.Tensor], start: int, stop: int,
                        context: dict[str, Any], pre_units: dict[int, torch.Tensor],
                        feature: "UnitFeatureKD | None" = None) -> torch.Tensor:
        """Walk layers stop-1 .. start back: units (recomputed with grad) then the layer (weights
        reloaded, boundary from the host).  Returns d loss / d (input of layer `start`).

        `feature` (D3d): while the named unit is recomputed, its block output on the packed owned
        rows is captured and its feature loss joins the backward of that recompute."""

        contexts = self.contexts(grad)
        for i in reversed(range(start, stop)):
            if i in pre_units:
                y = pre_units[i].to(self.device, copy=True).requires_grad_(True)
                here = feature is not None and any(p.container == feature.container for p in self.units_after(i))
                captured: list[torch.Tensor] = []
                with (capture_block_output(self.units[feature.container], captured) if here else _no_capture()), torch.enable_grad():
                    z = self.apply_units(y, i, context)
                if here:
                    feature_loss = feature.loss(captured)
                    torch.autograd.backward((z, feature_loss), (grad, torch.full_like(feature_loss, feature.scale)))
                else:
                    z.backward(grad)
                grad = y.grad
                del y, z
            layer = self.engine.layer(i, i - 1 if i - 1 >= start else None)
            x = bounds[i - start].to(self.device, copy=True).requires_grad_(True)
            with torch.enable_grad():
                y = self.engine._run(layer, i, x, contexts)
            y.backward(grad)
            grad = x.grad
            del x, y
        return grad

    # -- heads ---------------------------------------------------------------

    def media_hidden(self, normed: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        index = context["indices"].unsqueeze(-1).expand(-1, -1, normed.shape[-1])
        return normed.gather(1, index)

    def emission(self, normed: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        """The flow head (`native_media_head`) on the media rows, with any trainable row deltas."""

        rows = self.media_hidden(normed, context)
        params = self.interface_params("head")
        if not params:
            return self._head_call(rows, context["types"])
        return torch.func.functional_call(self._head_call, params, (rows, context["types"]))

    def text_logits(self, normed: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Logits over the text rows [0, base) only, at the given positions (fp32)."""

        head = self.engine.head_weight
        rows = normed[0, positions.to(normed.device)].to(head.device, head.dtype)
        return F.linear(rows, head).float()

    # -- whole forward -------------------------------------------------------

    def forward(self, segment: dict[str, Any]) -> dict[str, Any]:
        """The whole hybrid, no grad: `pre_units[i]` is the hidden entering the units placed after
        text layer i (x_d for d = i + 1), `writes` every unit's write, `emission` the flow head."""

        context = self.prepare(segment)
        x = context["x"]
        out: dict[str, Any] = {"writes": {}}
        with torch.no_grad():
            x = self.apply_units(x, -1, context, out["writes"])
            pre: dict[int, torch.Tensor] = {}
            x = self.run_layers(x, 0, self.num_layers, context, pre_units=pre, writes=out["writes"])
            normed = self.final_norm(x)
        out["pre_units"] = pre
        out["normed"] = normed
        out["context"] = context
        if context["media_rows"]:
            with torch.no_grad():
                out["emission"] = self.emission(normed, context)
        return out

    # -- trainable sets ------------------------------------------------------

    def unit_parameters(self, containers: list[str]) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for name in containers:
            unit = self.units[name]
            for pname, p in unit.named_parameters():
                # source-frozen coefficients (H3 curve table, H3 RoPE frequencies) stay frozen
                if pname in ("h3_time.curve_table", "attn.rope_inv_freq"):
                    continue
                p.requires_grad_(True)
                params.append(p)
        return params

    def add_emission_deltas(self, type_ids: list[int]) -> list[SliceDelta]:
        """Trainable emission rows of these types (out.weight band rows) + their type_affine rows."""

        out_w = self.media_head.out.weight
        ta = self.media_head.type_affine.weight
        if self.head_deltas:
            raise D3Error("emission deltas exist already: one delta per interface parameter per stage (name all types in one call)")
        ranges = []
        for t in sorted(set(type_ids)):
            if t not in self.bands:
                raise D3Error(f"type {t} has no interface band in the manifest")
            ranges.append(self.bands[t]["emission"])
        deltas = [SliceDelta("out.weight", tuple(out_w.shape), ranges, 0, self.device),
                  SliceDelta("type_affine.weight", tuple(ta.shape), [(t, t + 1) for t in sorted(set(type_ids))], 0, self.device)]
        self.head_deltas.extend(deltas)
        return deltas

    def add_bridge_deltas(self, type_ids: list[int]) -> list[SliceDelta]:
        """Trainable interface-in rows: feature_proj columns of these types' feature bands + type_embed rows."""

        fp = self.bridge.feature_proj.weight
        te = self.bridge.type_embed.weight
        if self.bridge_deltas:
            raise D3Error("bridge deltas exist already: one delta per interface parameter per stage (name all types in one call)")
        for t in sorted(set(type_ids)):
            if t not in self.bands:
                raise D3Error(f"type {t} has no interface band in the manifest")
        cols = [self.bands[t]["feature"] for t in sorted(set(type_ids))]
        deltas = [SliceDelta("feature_proj.weight", tuple(fp.shape), cols, 1, self.device),
                  SliceDelta("type_embed.weight", tuple(te.shape), [(t, t + 1) for t in sorted(set(type_ids))], 0, self.device)]
        self.bridge_deltas.extend(deltas)
        return deltas

    def lora_parameters(self, layers: range | None = None) -> list[torch.Tensor]:
        if self.bank is None:
            return []
        keep = set(range(self.num_layers) if layers is None else layers)
        out = []
        for name in self.bank.names:
            tensor = self.bank.tensors[name]
            index = int(name.split(".")[1])
            if index in keep and tensor.requires_grad:
                out.append(tensor)
        return out

    def freeze_lora_outside(self, layers: range) -> None:
        """D3b: layers below d keep their LoRA but get no gradient (x_d is cached past them)."""

        if self.bank is None:
            return
        keep = set(layers)
        for name in self.bank.names:
            if int(name.split(".")[1]) not in keep:
                self.bank.tensors[name].requires_grad_(False)

    def lora_digest(self, layers: range) -> str:
        """sha256 over the bank tensors of `layers` (x_d validity: the layers below d)."""

        digest = hashlib.sha256()
        if self.bank is not None:
            for name in self.bank.names:
                if int(name.split(".")[1]) in set(layers):
                    digest.update(name.encode())
                    digest.update(self.bank.tensors[name].detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


# --------------------------------------------------------------------------
# Objectives.
# --------------------------------------------------------------------------


class _no_capture:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False


class capture_block_output:
    """Observe a ChartedSourceBlock's own `block` method (an instance attribute wrapper, removed on
    exit): its computation and weights are unchanged; every output is appended to `sink`."""

    def __init__(self, unit: nn.Module, sink: list[torch.Tensor]) -> None:
        self.unit, self.sink = unit, sink

    def __enter__(self) -> None:
        if "block" in self.unit.__dict__:
            raise D3Error("the unit's block method is already wrapped")
        original = self.unit.block

        def observed(*args: Any, **kwargs: Any) -> torch.Tensor:
            out = original(*args, **kwargs)
            self.sink.append(out)
            return out

        self.unit.block = observed

    def __exit__(self, *exc: Any) -> bool:
        del self.unit.block
        return False


@dataclass
class UnitFeatureKD:
    """D3d's QVL32 feature term: weight x mean((unit block output - teacher)^2) over the unit's packed
    owned rows [1, N, width] (absolute MSE, ViT block 13, weight 0.1 per prereg section 3 D3d)."""

    container: str
    teacher: torch.Tensor          # [1, N, width] or [N, width], detached
    weight: float = QVL_FEATURE_WEIGHT
    scale: float = 1.0             # the whole-objective loss scale (as `loss_scale`)
    value: float | None = None
    captures: int = 0

    def loss(self, captured: list[torch.Tensor]) -> torch.Tensor:
        self.captures += len(captured)
        if len(captured) != 1:
            raise D3Error(f"{self.container}: expected one block output in the recompute, saw {len(captured)}")
        actual = captured[0]
        teacher = self.teacher if self.teacher.ndim == 3 else self.teacher.unsqueeze(0)
        if teacher.requires_grad or tuple(teacher.shape) != tuple(actual.shape) or not bool(torch.isfinite(teacher).all()):
            raise D3Error(f"{self.container}: teacher features {tuple(teacher.shape)} do not match the packed owned rows "
                          f"{tuple(actual.shape)} (or are not detached / finite)")
        out = float(self.weight) * (actual.float() - teacher.to(actual.device).float()).square().mean()
        if not bool(torch.isfinite(out)):
            raise D3Error(f"{self.container}: feature loss is not finite")
        self.value = float(out.detach())
        return out


def relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction, target = prediction.float(), target.float()
    return (prediction - target).pow(2).sum() / target.pow(2).sum().clamp_min(1e-30)


def teacher_x0(x_t: torch.Tensor, raw: torch.Tensor, sigma: torch.Tensor, sign: float) -> torch.Tensor:
    """x0 from a parent's raw output in its own convention: x0 = x_t + SIGN * sigma * raw."""

    s = sigma.float()
    while s.ndim < x_t.ndim:
        s = s.unsqueeze(-1)
    return x_t.float() + float(sign) * s * raw.float()


def flow_x0_loss(prediction: torch.Tensor, x0_target: torch.Tensor, types: torch.Tensor, mask: torch.Tensor,
                 bands: dict[int, dict[str, Any]], *, param: str = "x0", x_t: torch.Tensor | None = None,
                 sigma: torch.Tensor | None = None, signs: dict[int, float] | None = None) -> torch.Tensor:
    """Per row: mean over the row type's EMISSION band of (x0_pred - x0_target)^2; type-balanced
    mean over supervised rows (Chimera's `native_loss_reduction = type_balanced`).

    prediction / x0_target / x_t are [B, M, F] in the emission layout; types [B, M]; mask [B, M]."""

    if param not in ("x0", "raw"):
        raise D3Error("flow param is x0 or raw")
    per_type: list[torch.Tensor] = []
    for t in sorted(int(v) for v in torch.unique(types[mask > 0]).tolist()):
        if t not in bands:
            raise D3Error(f"type {t} has no emission band")
        lo, hi = bands[t]["emission"]
        rows = (types == t) & (mask > 0)
        pred = prediction[..., lo:hi][rows].float()
        if param == "raw":
            if x_t is None or sigma is None or signs is None:
                raise D3Error("flow param raw needs x_t, sigma and the parent signs")
            pred = teacher_x0(x_t[..., lo:hi][rows], pred, sigma[rows], signs[t])
        target = x0_target[..., lo:hi][rows].float()
        if not bool(torch.isfinite(target).all()):
            raise D3Error(f"type {t}: non-finite x0 target")
        weight = mask[rows].float()
        err = (pred - target).pow(2).mean(dim=-1)
        per_type.append((err * weight).sum() / weight.sum().clamp_min(1e-12))
    if not per_type:
        raise D3Error("flow loss with no supervised media row")
    return torch.stack(per_type).mean()


def to_velocity(value: torch.Tensor, kind: str, *, x_t: torch.Tensor | None = None, sigma: torch.Tensor | None = None,
                sign: float | None = None) -> torch.Tensor:
    """Rows [n, f] of one type's emission band -> canonical velocity v = (x_t - x0) / sigma.

    kind "canonical_velocity": already v.  "parent_raw": a parent's raw output, x0 = x_t + SIGN sigma
    raw, so v = -SIGN raw.  "x0": needs the row's own x_t (emission-aligned) and sigma in (0, 1]."""

    if kind == "canonical_velocity":
        return value.float()
    if kind == "parent_raw":
        if sign not in (-1.0, 1.0, -1, 1):
            raise D3Error("a parent-raw velocity needs the audited per-type sign (+-1)")
        return -float(sign) * value.float()
    if kind == "x0":
        if x_t is None or sigma is None or x_t.shape != value.shape or sigma.shape != value.shape[:1]:
            raise D3Error("x0 -> velocity needs the exact noisy rows and one sigma per row")
        s = sigma.float()
        if not bool(torch.isfinite(s).all()) or bool(((s <= 0) | (s > 1)).any()):
            raise D3Error("x0 -> velocity needs sigma in (0, 1]")
        return (x_t.float() - value.float()) / s[:, None]
    raise D3Error(f"velocity parametrisation {kind!r}: one of canonical_velocity / parent_raw / x0")


def flow_velocity_loss(prediction: torch.Tensor, target: torch.Tensor, types: torch.Tensor, mask: torch.Tensor,
                       bands: dict[int, dict[str, Any]], *, prediction_kind: str = "x0", target_kind: str = "parent_raw",
                       x_t: torch.Tensor | None = None, sigma: torch.Tensor | None = None,
                       signs: dict[int, float] | None = None, noisy_feature_bands: dict[int, tuple[int, int]] | None = None,
                       x_t_layout: str = "feature", channel_mask: torch.Tensor | None = None) -> torch.Tensor:
    """The D3b / D3e_out guided-velocity objective (prereg section 3 D3b), on the x0 head.

    Per supervised type: velocity MSE over the type's EMISSION band (mean over its channels, or over
    `channel_mask`'s true channels), weighted by `mask` per row; type-balanced mean over types (as
    `flow_x0_loss`).  `prediction` [B, M, F] is the head output in `prediction_kind` ("x0", or "raw"
    = parent-raw velocity rows, or "canonical_velocity"); `target` [B, M, F] is in `target_kind`
    (emission layout).  `x_t` is in `x_t_layout`: "feature" = the bridge input rows, whose type-t
    noisy latent sits at `noisy_feature_bands[t]` (realigned onto the emission columns here; the two
    allocations differ), or "emission".  Semantics equal Codex's `training.flow_velocity_loss`
    (`param`/`target_kind`), checked by the parity arm in the tests."""

    if prediction.ndim != 3 or prediction.shape != target.shape or types.shape != prediction.shape[:2] or mask.shape != types.shape:
        raise D3Error("velocity objective: prediction / target [B, M, F], types and mask [B, M]")
    if target.requires_grad or not bool(torch.isfinite(mask).all()) or bool((mask < 0).any()):
        raise D3Error("velocity objective: the teacher target is detached and the row weights finite, >= 0")
    if channel_mask is not None and (channel_mask.shape != prediction.shape or channel_mask.dtype != torch.bool):
        raise D3Error("velocity channel mask must be bool [B, M, F]")
    head_kind = {"x0": "x0", "raw": "parent_raw", "canonical_velocity": "canonical_velocity"}.get(prediction_kind)
    if head_kind is None:
        raise D3Error(f"flow head parametrisation {prediction_kind!r}: x0 / raw / canonical_velocity")
    active = mask > 0
    if not bool(active.any()):
        raise D3Error("velocity objective with no supervised media row")
    if x_t is not None:
        if x_t_layout == "feature":
            aligned = torch.zeros_like(prediction, dtype=x_t.dtype)
            for t in sorted(int(v) for v in torch.unique(types[active]).tolist()):
                if noisy_feature_bands is None or t not in noisy_feature_bands or t not in bands:
                    raise D3Error(f"type {t}: velocity conversion needs the audited noisy-channel sub-band of its feature band")
                flo, fhi = (int(v) for v in noisy_feature_bands[t])
                blo, bhi = bands[t]["feature"]
                elo, ehi = bands[t]["emission"]
                if not (blo <= flo < fhi <= bhi <= x_t.shape[-1]) or fhi - flo != ehi - elo:
                    raise D3Error(f"type {t}: noisy sub-band [{flo}, {fhi}) outside its feature band or not emission-wide")
                rows = types == t
                aligned[..., elo:ehi][rows] = x_t[..., flo:fhi][rows]
            x_t = aligned
        elif x_t_layout != "emission":
            raise D3Error("x_t layout is 'feature' or 'emission'")
    if x_t is not None and x_t.shape != prediction.shape:
        raise D3Error("x_t must match the prediction's shape in the emission layout")
    per_type: list[torch.Tensor] = []
    for t in sorted(int(v) for v in torch.unique(types[active]).tolist()):
        if t not in bands:
            raise D3Error(f"type {t} has no emission band")
        lo, hi = bands[t]["emission"]
        if not 0 <= lo < hi <= prediction.shape[-1]:
            raise D3Error(f"type {t}: emission band outside the feature width")
        rows = (types == t) & active
        sign = None if signs is None else signs.get(t)
        xt_rows = None if x_t is None else x_t[..., lo:hi][rows]
        sg_rows = None if sigma is None else sigma[rows]
        pred = to_velocity(prediction[..., lo:hi][rows], head_kind, x_t=xt_rows, sigma=sg_rows, sign=sign)
        truth = to_velocity(target[..., lo:hi][rows], target_kind, x_t=xt_rows, sigma=sg_rows, sign=sign)
        if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(truth).all()):
            raise D3Error(f"type {t}: non-finite velocity prediction or teacher")
        sq = (pred - truth).square()
        if channel_mask is not None:
            channels = channel_mask[..., lo:hi][rows]
            counts = channels.sum(-1)
            if not bool((counts > 0).all()):
                raise D3Error(f"type {t}: a supervised row has no teacher channel")
            per_row = (sq * channels).sum(-1) / counts
        else:
            per_row = sq.mean(-1)
        weight = mask[rows].float()
        per_type.append((per_row * weight).sum() / weight.sum())
    return torch.stack(per_type).mean()


def make_flow_loss(objective: str, context: dict[str, Any], segment: dict[str, Any], target: torch.Tensor, mask: torch.Tensor,
                   bands: dict[int, dict[str, Any]], *, target_kind: str = "x0", prediction_kind: str = "x0",
                   noisy_feature_bands: dict[int, tuple[int, int]] | None = None, signs: dict[int, float] | None = None,
                   channel_mask: torch.Tensor | None = None) -> Callable[[torch.Tensor], torch.Tensor]:
    """The D3b `loss_fn(prediction)`: "guided_velocity" (the lane loss) or "x0_endpoint" (1.0.2's
    endpoint MSE, a diagnostic objective).  x_t is the segment's own feature rows and sigma its own
    per-row noise levels: the same tensors the bridge consumed, never a recomputed clean input."""

    if objective == "x0_endpoint":
        if target_kind != "x0" or prediction_kind != "x0":
            raise D3Error("x0_endpoint compares x0 predictions with x0 targets only")
        return lambda pred: flow_x0_loss(pred, target, context["types"], mask, bands)
    if objective != "guided_velocity":
        raise D3Error(f"flow objective {objective!r}: one of {FLOW_OBJECTIVES}")
    features, sigma = segment.get("features"), segment.get("sigma")
    if features is None or sigma is None:
        raise D3Error("guided velocity needs the segment's noisy feature rows and per-row sigma")
    x_t = features.to(target.device)
    sg = sigma.to(target.device)
    return lambda pred: flow_velocity_loss(pred, target, context["types"], mask, bands, prediction_kind=prediction_kind,
                                           target_kind=target_kind, x_t=x_t, sigma=sg, signs=signs,
                                           noisy_feature_bands=noisy_feature_bands, channel_mask=channel_mask)


def band_teacher_logprobs(topk_logprobs: torch.Tensor, lse_full: torch.Tensor, lse_band: torch.Tensor) -> torch.Tensor:
    """ACE LM top-k log-probs (log-softmax over the FULL 217,204 vocab) renormalised onto the
    64,000 code rows: log p_band = log p_full + lse_full - lse_band."""

    return topk_logprobs.float() + (lse_full.float() - lse_band.float()).unsqueeze(-1)


def band_kl(normed_rows: torch.Tensor, band_head_rows: torch.Tensor, topk_codes: torch.Tensor,
            topk_logprobs_band: torch.Tensor) -> torch.Tensor:
    """Mean over positions of KL(teacher top-k + tail || student) with the student's softmax over
    the code rows only (the trainer's bucket KL, so the tail is the exact logsumexp of the rest)."""

    T = trainer_module()
    logits = F.linear(normed_rows.float(), band_head_rows.float())
    logq = torch.log_softmax(logits, dim=-1)
    kl = T.bucket_forward_kl(logq, topk_codes.long(), topk_logprobs_band)
    return kl.mean()


#: the one declared D3a host-write objective (Codex train_stages STAGES["D3a"], A3 draft item 4)
D3A_WRITE_OBJECTIVE = "teacher_delta_host_write"


def check_d3a_declaration(declaration: dict[str, Any] | None, *, fixture: bool = False) -> dict[str, Any] | None:
    """The declared D3a host-write target, or None (chart_out then stays OUT of the D3a census).

    A production declaration pins, by sha256, the D2 child the host rows were recaptured on, the
    recaptured host cache, the independent host-write target capture and the teacher cache, names
    the objective `teacher_delta_host_write` and states the target is independent (captured on this
    child, not inferred from an older cache: Codex's D3a finding).  The weights of the three terms
    are explicit and positive."""

    if declaration is None:
        return None
    if declaration.get("objective") != D3A_WRITE_OBJECTIVE:
        raise D3Error(f"D3a host-write objective must be {D3A_WRITE_OBJECTIVE!r}")
    weights = declaration.get("weights")
    if not isinstance(weights, dict) or set(weights) != {"chart_in", "block", "chart_out"} or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise D3Error("D3a declaration weights {chart_in, block, chart_out} must be explicit, finite and positive")
    if not fixture:
        for key in ("d2_child_sha256", "host_cache_sha256", "host_write_target_sha256", "teacher_cache_sha256"):
            if not is_sha256(declaration.get(key)):
                raise D3Error(f"D3a declaration {key} must be a sha256 pin")
        if declaration.get("host_write_target_independent") is not True or declaration.get("inferred_from_old_cache") is not False:
            raise D3Error("D3a host-write targets must be recaptured on this D2 child, never inferred from an older cache")
    return dict(declaration)


def d3a_parameters(unit: nn.Module, *, write_target_declared: bool) -> dict[str, nn.Parameter]:
    """The D3a census of one unit: every parameter the stated loss reaches.  chart_out is in it
    only with a declared host-write target; the source-frozen H3 coefficients never are."""

    out: dict[str, nn.Parameter] = {}
    for name, p in unit.named_parameters():
        if name in ("h3_time.curve_table", "attn.rope_inv_freq"):
            continue
        if name.startswith("chart_out.") and not write_target_declared:
            continue
        out[name] = p
    return out


def d3a_loss(unit: nn.Module, host_rows: torch.Tensor, parent_in: torch.Tensor, *, levels: torch.Tensor | None = None,
             row_types: torch.Tensor | None = None, positions: torch.Tensor | None = None,
             parent_out: torch.Tensor | None = None, w_in: float = 1.0, w_block: float = 1.0, weight: float = 1.0,
             host_write_target: torch.Tensor | None = None, w_write: float = 1.0
             ) -> tuple[torch.Tensor, dict[str, float]]:
    """D3a for one unit on packed rows [n, *]: chart_in(host) -> parent block input; and, when
    the cache has the parent's block output, block(parent_in) - parent_in -> parent_out - parent_in
    (the block teacher-forced on the parent's own input).  `weight` is the unit weight (QVL32 0.1).

    With a declared `host_write_target` [n, hidden] (1.1.0): the outward chart on the TEACHER delta,
    delta_scale * (parent_out - parent_in) @ chart_out^T, -> the host write recaptured on the D2
    child (relative MSE over the rows the unit owns; the teacher delta is detached, so this term
    trains chart_out only).  Codex's version omits delta_scale; it is 1.0 on every OMNI1 unit."""

    n = int(host_rows.shape[0])
    dtype = unit.chart_in.weight.dtype
    projected = unit.chart_in(host_rows.to(dtype).view(1, n, -1))
    loss_in = relative_mse(projected[0], parent_in)
    parts = {"chart_in_rel_mse": float(loss_in.detach())}
    loss = w_in * loss_in
    if parent_out is not None:
        key_valid = torch.ones(1, n, dtype=torch.bool, device=host_rows.device)
        lv = torch.zeros(1, n, device=host_rows.device) if levels is None else levels.view(1, n).float()
        rt = None if row_types is None else row_types.view(1, n).long()
        sp = None if positions is None else positions.view(1, n, 3)
        p_in = parent_in.to(dtype).view(1, n, -1)
        out = unit.block(p_in, lv, key_valid, row_types=rt, source_positions=sp)
        loss_block = relative_mse((out - p_in)[0], parent_out.float() - parent_in.float())
        parts["block_rel_mse"] = float(loss_block.detach())
        loss = loss + w_block * loss_block
    if host_write_target is not None:
        if parent_out is None:
            raise D3Error("the D3a host-write term needs the parent's block output (the teacher delta)")
        if host_write_target.shape != host_rows.shape or host_write_target.requires_grad or not bool(torch.isfinite(host_write_target).all()):
            raise D3Error("D3a host-write target must be a detached finite [n, hidden] tensor")
        owned = torch.ones(n, dtype=torch.bool, device=host_rows.device)
        if row_types is not None:
            owned = torch.isin(row_types.view(n).long().to(host_rows.device),
                               torch.tensor(sorted(int(t) for t in unit.spec.type_ids), device=host_rows.device))
        if not bool(owned.any()):
            raise D3Error("D3a host-write rows hold no row the unit owns")
        out_w = unit.chart_out.weight
        teacher_delta = (parent_out.float() - parent_in.float()).detach() * float(unit.delta_scale)
        predicted = teacher_delta.to(out_w.dtype) @ out_w.transpose(0, 1)
        loss_write = relative_mse(predicted[owned], host_write_target.to(predicted.device)[owned])
        parts["chart_out_rel_mse"] = float(loss_write.detach())
        loss = loss + w_write * loss_write
    return weight * loss, parts


# --------------------------------------------------------------------------
# Streamed steps.
# --------------------------------------------------------------------------


def d3b_forward_backward(model: HybridD3, context: dict[str, Any], x_d: torch.Tensor, depth: int,
                         loss_fn: Callable[[torch.Tensor], torch.Tensor], *, loss_scale: float = 1.0) -> float:
    """One D3b micro-step from a cached x_d (the hidden entering depth d, before the units).

    Units (with graph) -> layers d..L-1 streamed (no grad, host boundaries) -> final norm + flow
    head (graph) -> loss -> layers walked back with reloaded weights -> units' backward."""

    x_in = x_d.to(device=model.device, dtype=model.dtype)
    with torch.enable_grad():
        z = model.apply_units(x_in, depth - 1, context)
    bounds: list[torch.Tensor] = []
    pre: dict[int, torch.Tensor] = {}
    h = model.run_layers(z.detach(), depth, model.num_layers, context, bounds=bounds, pre_units=pre)
    last = h.detach().requires_grad_(True)
    with torch.enable_grad():
        normed = model.final_norm(last)
        loss = loss_fn(model.emission(normed, context))
    if not bool(torch.isfinite(loss)):
        raise D3Error(f"D3b loss is not finite: {float(loss)}")
    (loss * float(loss_scale)).backward()
    grad = model.backward_layers(last.grad, bounds, depth, model.num_layers, context, pre)
    if z.requires_grad:
        z.backward(grad)
    return float(loss.detach())


def full_depth_forward_backward(model: HybridD3, segment: dict[str, Any],
                                head_loss: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor], *,
                                loss_scale: float = 1.0, input_grad: bool = False,
                                feature: UnitFeatureKD | None = None) -> tuple[float, torch.Tensor | None, dict[str, Any]]:
    """D3c / D3d / D3e: the whole stack.  The input (embedding + bridge, with any bridge deltas)
    keeps its graph; returns (loss, d loss / d x0 when `input_grad`, context).  With `feature`
    (D3d's QVL term) the returned loss is the head loss + the weighted feature loss, and
    `feature.value` holds the latter."""

    if feature is not None:
        placed = next((p for p in model.layout if p.container == feature.container), None)
        if placed is None or placed.after_text_layer < 0:
            raise D3Error(f"feature KD unit {feature.container!r} must sit after a text layer (it is recomputed in the walk back)")
        feature.scale = float(loss_scale)
        feature.captures, feature.value = 0, None
    context = model.prepare(segment, media_grad=bool(model.bridge_deltas))
    x0 = context["x"]
    x_start = x0.detach()
    with torch.enable_grad():
        x_start = model.apply_units(x_start, -1, context) if model.units_after(-1) else x_start
    bounds: list[torch.Tensor] = []
    pre: dict[int, torch.Tensor] = {}
    h = model.run_layers(x_start.detach(), 0, model.num_layers, context, bounds=bounds, pre_units=pre)
    last = h.detach().requires_grad_(True)
    with torch.enable_grad():
        normed = model.final_norm(last)
        loss = head_loss(normed, context)
    if not bool(torch.isfinite(loss)):
        raise D3Error(f"loss is not finite: {float(loss)}")
    (loss * float(loss_scale)).backward()
    grad = model.backward_layers(last.grad, bounds, 0, model.num_layers, context, pre, feature=feature)
    if x0.requires_grad:
        x0.backward(grad, retain_graph=input_grad)
    if feature is not None:
        if feature.captures != 1 or feature.value is None:
            raise D3Error(f"{feature.container}: the feature KD term was not attached (captures {feature.captures})")
        return float(loss.detach()) + feature.value, (grad.detach() if input_grad else None), context
    return float(loss.detach()), (grad.detach() if input_grad else None), context


class BandTrainer:
    """D3c: host fp32 masters of the C18 code rows of embed and lm_head, LazyRowAdam on each.

    The music band has `model.music_rows` allocated rows (OMNI1 65,536); the ACE LM writes `codes`
    of them (64,000).  The head master holds the written rows only, so the band softmax, the KL and
    every head update are over the written codes; the embed master is the whole band table but
    only rows of written codes that occur in a batch are ever stepped.  `frozen` digests the ledger
    rows below the band and the reserved rows (embed and head) at construction; every step re-checks
    them and refuses if one moved (Codex's D3c finding: 64,000 written vs 65,536 allocated)."""

    def __init__(self, model: HybridD3, *, lr: float, codes: int = ACE_CODES) -> None:
        self.model = model
        self.offset = model.music_band_begin - model.base_vocab   # band row index of code 0
        self.codes = int(codes)
        if model.band_head is None:
            raise D3Error("D3c needs an untied lm_head (the band head rows)")
        self.contract = model.music_contract(self.codes)
        # inherited music tables (an earlier D3c run's export) start the masters; else the base rows
        embed_source = model.band_embed_override if model.band_embed_override is not None else model.band_embed
        head_source = model.inherited_band_head
        full_embed = embed_source.float().clone()
        self.embed_master = full_embed
        if head_source is None:
            self.head_master = model.band_head[self.offset:self.offset + self.codes].float().clone()
            self.head_reserved = model.band_head[self.offset + self.codes:self.offset + model.music_rows].float().clone()
        else:
            self.head_master = head_source[:self.codes].float().clone()
            self.head_reserved = head_source[self.codes:].float().clone()
        self.embed_opt = LazyRowAdam(self.embed_master, lr=lr)
        self.head_opt = LazyRowAdam(self.head_master, lr=lr)
        model.band_embed_override = self.embed_master
        self.frozen = self._frozen_digests()
        if (self.frozen["ledger_embed_sha256"] != self.contract["ledger_embed_sha256"]
                or self.frozen["reserved_embed_sha256"] != self.contract["reserved_embed_sha256"]
                or self.frozen["reserved_head_sha256"] != self.contract["reserved_head_sha256"]):
            raise D3Error("inherited music tables differ from the base on ledger or reserved rows")
        self.max_teacher_mass = 0.0

    def _frozen_digests(self) -> dict[str, str]:
        o, c, r = self.offset, self.codes, self.model.music_rows
        return {"ledger_embed_sha256": tensor_sha256(self.embed_master[:o]),
                "reserved_embed_sha256": tensor_sha256(self.embed_master[o + c:o + r]),
                "reserved_head_sha256": tensor_sha256(self.head_reserved)}

    def check_frozen(self) -> None:
        if self._frozen_digests() != self.frozen:
            raise D3Error("D3c moved a ledger or reserved music row (only the written codes may change)")

    def music_tables(self) -> dict[str, torch.Tensor]:
        """The fp32 music tables over ALL allocated rows (export): written rows trained, reserved rows
        bit-equal to the base."""

        o, r = self.offset, self.model.music_rows
        return {"music.embed": self.embed_master[o:o + r].clone(),
                "music.head": torch.cat([self.head_master, self.head_reserved], dim=0)}

    def check_step_inputs(self, ids: torch.Tensor, prompt_len: int, topk_codes: torch.Tensor, topk_lp_band: torch.Tensor) -> None:
        """Teacher forcing over the written codes only: the prompt is text, the band ids are written
        codes, the teacher's top-k ids are distinct written codes with finite log-probs."""

        model = self.model
        n = int(topk_codes.shape[0])
        if ids.ndim != 1 or prompt_len < 1 or n < 1 or ids.numel() < prompt_len + n - 1:
            raise D3Error("D3c needs [prompt | codes] ids with one teacher row per next-code position")
        if topk_codes.ndim != 2 or topk_lp_band.shape != topk_codes.shape or not bool(torch.isfinite(topk_lp_band).all()):
            raise D3Error("D3c teacher top-k ids / log-probs must be finite [n, k]")
        prompt, codes = ids[:prompt_len], ids[prompt_len:]
        if bool(((prompt < 0) | (prompt >= model.base_vocab)).any()):
            raise D3Error("D3c prompt ids must be text rows")
        if bool(((codes < model.music_band_begin) | (codes >= model.music_band_begin + self.codes)).any()):
            raise D3Error(f"D3c band ids outside the {self.codes} written code rows (reserved / ledger rows are frozen)")
        if bool(((topk_codes < 0) | (topk_codes >= self.codes)).any()):
            raise D3Error(f"D3c teacher code ids outside [0, {self.codes})")
        if any(row.unique().numel() != row.numel() for row in topk_codes):
            raise D3Error("D3c teacher top-k repeats a code")
        # stored bf16 top-k can exceed mass 1 by rounding; the bucket KL clamps the tail at 0 (D2-T5 rule)
        self.max_teacher_mass = max(self.max_teacher_mass, float(topk_lp_band.float().exp().sum(-1).max()))

    def head_rows(self) -> torch.Tensor:
        """A fresh copy of the head master on the model's device / dtype.  Never the master itself:
        `Tensor.to` returns the same object when device and dtype already match (CPU fp32), and the
        grad would then accumulate on the master across steps (1.0.1 trained D3c on a running sum)."""
        return self.head_master.detach().to(self.model.device, self.model.dtype, copy=True)

    def step(self, segment: dict[str, Any], prompt_len: int, topk_codes: torch.Tensor, topk_lp_band: torch.Tensor) -> float:
        """segment ids = [prompt (text ids) | band ids of the codes]; position prompt_len-1+k
        predicts code k (record k of the ACE LM cache)."""

        model = self.model
        n = int(topk_codes.shape[0])
        self.check_step_inputs(segment["ids"][0], int(prompt_len), topk_codes, topk_lp_band)
        positions = torch.arange(prompt_len - 1, prompt_len - 1 + n)
        head = self.head_rows().requires_grad_(True)

        def head_loss(normed: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
            rows = normed[0, positions.to(normed.device)]
            return band_kl(rows, head, topk_codes.to(normed.device), topk_lp_band.to(normed.device))

        ids = segment["ids"][0]
        x_grad_needed = bool((ids >= model.base_vocab).any())
        loss, grad_x0, _ = full_depth_forward_backward(model, segment, head_loss, input_grad=x_grad_needed)
        self.head_opt.step(torch.arange(self.codes), head.grad.detach().float().cpu())
        if x_grad_needed and grad_x0 is not None:
            band = ids >= model.base_vocab
            rows, grads = unique_row_grads((ids[band] - model.base_vocab).cpu(), grad_x0[0][band.to(grad_x0.device)].float().cpu())
            self.embed_opt.step(rows, grads)
        self.check_frozen()
        return loss


def answer_kd_targets(length: int, positions: torch.Tensor, next_ids: torch.Tensor, q_ids: torch.Tensor,
                      q_lp: torch.Tensor, *, w_ce: float = 0.0) -> dict[str, torch.Tensor]:
    """The T(ii) targets dict (`chunked_objective`) for D3d: teacher top-k at the answer positions
    (position p carries the teacher's distribution for token p+1), zero weight elsewhere."""

    k = int(q_ids.shape[-1])
    out = {"next_ids": torch.full((length,), -1, dtype=torch.long), "ce_w": torch.zeros(length), "eos_w": torch.zeros(length),
           "q_w": torch.zeros(length), "q_ids": torch.full((length, k), -1, dtype=torch.long), "q_lp": torch.zeros(length, k)}
    out["next_ids"][positions] = next_ids.long()
    out["ce_w"][positions] = float(w_ce)
    out["q_w"][positions] = 1.0
    out["q_ids"][positions] = q_ids.long()
    out["q_lp"][positions] = q_lp.float()
    return out


def d3d_head_loss(model: HybridD3, targets: dict[str, torch.Tensor], *, denominator: float) -> Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]:
    """The text KD over rows [0, base) through the trainer's `chunked_objective`; it backwards per
    chunk into a detached copy, so the returned surrogate carries that gradient to `normed`."""

    T = model.T
    objective = T.ObjectiveConfig(w_q38=1.0, lambda_k3=0.0, w_ce=0.0, w_eos=0.0, w_ul=0.0,
                                  text_vocab_rows=model.base_vocab, head_chunk_positions=1024)

    def loss_fn(normed: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        head = model.engine.head_weight
        parts, grad = T.chunked_objective(normed.to(head.device, head.dtype), head, targets, objective,
                                          denominator=denominator, backward=True)
        surrogate = (normed * grad.to(normed.device, normed.dtype)).sum()
        value = torch.tensor(float(parts["total"]), device=normed.device)
        return surrogate - surrogate.detach() + value

    return loss_fn


def check_answer_targets(targets: dict[str, torch.Tensor], ids: torch.Tensor, assistant_mask: torch.Tensor, base_vocab: int) -> int:
    """D3d / D3e_in answer supervision: position p carries the teacher distribution for token p + 1,
    so the supervised positions are exactly the assistant-token mask shifted left by one; weight 1
    there and 0 elsewhere; every teacher id and every next id a native text row [0, base_vocab).
    Returns the supervised position count (the KD denominator).  Codex vision_cache rule: the
    teacher answers must be in the student's own vocabulary, aligned to the same token ids -- no
    guessed remapping of a ViT-only cache."""

    ids = ids.reshape(-1)
    mask = assistant_mask.reshape(-1).to(torch.bool)
    if mask.shape != ids.shape:
        raise D3Error("assistant mask must cover the sample's ids")
    shifted = torch.zeros_like(mask)
    shifted[:-1] = mask[1:]
    q_w = targets["q_w"].reshape(-1)
    if q_w.shape != ids.shape or not torch.equal(q_w > 0, shifted) or bool((q_w[shifted] != 1).any()):
        raise D3Error("answer KD weights must be 1 on exactly the positions before assistant tokens")
    if int(shifted.sum()) == 0:
        raise D3Error("no assistant answer position to supervise")
    q_ids = targets["q_ids"].reshape(ids.numel(), -1)[shifted]
    if bool(((q_ids < 0) | (q_ids >= int(base_vocab))).any()):
        raise D3Error(f"teacher answer ids outside the native text rows [0, {base_vocab})")
    nxt = targets["next_ids"].reshape(-1)
    if not torch.equal(nxt[:-1][shifted[:-1]], ids[1:][shifted[:-1]]):
        raise D3Error("answer next ids differ from the sample's own assistant tokens")
    if not bool(torch.isfinite(targets["q_lp"]).all()):
        raise D3Error("teacher answer log-probs are not finite")
    return int(shifted.sum())


def check_vision_alignment(segment: dict[str, Any], teacher: dict[str, torch.Tensor], bands: dict[int, dict[str, Any]], *,
                           qvl_type: int = 6, patch_width: int | None = None) -> dict[str, Any]:
    """D3d C11 pixels / positions / feature rows against one sealed QVL32 teacher shard (Claude's
    `media_b_teacher_cache` qvl32_vit_l13 tensors: block_in / block_out [N, W], grid_thw, yx,
    positions [N, 4], cu_seqlens, optional patch_flat [N, 1536]).

    Every media row is the QVL type; the C11 feature columns hold the patch rows and nothing sits
    outside them; when the shard stored patch_flat the C11 columns equal it bit for bit; the
    segment's native positions equal the shard's; block_out has one row per media row (the unit's
    packed owned rows) and cu_seqlens is the single image [0, N]."""

    types, features = segment["types"], segment["features"]
    n = int(types.shape[-1])
    lo, hi = bands[qvl_type]["feature"]
    width = hi - lo if patch_width is None else int(patch_width)
    if types.shape != (1, n) or not bool((types == qvl_type).all()):
        raise D3Error(f"vision rows must all be type {qvl_type}")
    if features.shape[:2] != (1, n) or hi - lo != width:
        raise D3Error(f"C11 band [{lo}, {hi}) is not {width} wide or the features are not [1, {n}, F]")
    outside = features.clone()
    outside[..., lo:hi] = 0
    if bool(outside.any()):
        raise D3Error("vision features write outside the C11 band")
    block_out = teacher["block_out"]
    if block_out.ndim != 2 or int(block_out.shape[0]) != n or not bool(torch.isfinite(block_out).all()):
        raise D3Error(f"teacher block_out {tuple(block_out.shape)} has not one finite row per media row ({n})")
    if "block_in" in teacher and tuple(teacher["block_in"].shape) != tuple(block_out.shape):
        raise D3Error("teacher block_in / block_out shapes differ")
    grid = teacher["grid_thw"].reshape(-1).tolist()
    if len(grid) != 3 or int(grid[0]) * int(grid[1]) * int(grid[2]) != n:
        raise D3Error(f"teacher grid_thw {grid} does not hold {n} patch rows")
    cu = teacher["cu_seqlens"].reshape(-1).tolist()
    if [int(v) for v in cu] != [0, n]:
        raise D3Error(f"teacher cu_seqlens {cu} is not one image of {n} rows")
    if not torch.equal(segment["positions"][0].to("cpu").float(), teacher["positions"].to("cpu").float()):
        raise D3Error("segment native positions differ from the teacher shard's")
    pixels_checked = "patch_flat" in teacher
    if pixels_checked and not torch.equal(features[0, :, lo:hi].to("cpu").float(), teacher["patch_flat"].to("cpu").float()):
        raise D3Error("C11 pixel rows differ from the shard's stored patches")
    return {"rows": n, "pixels_checked": pixels_checked, "feature_width": int(block_out.shape[1])}


def qvl_patches_from_image(image_path: str | Path, item_sha256: str, *, max_pixels: int) -> tuple[torch.Tensor, list[int]]:
    """The patch rows media_b_teacher_cache.run_qvl fed the ViT for this image (for a shard sealed
    without --store-patches): media_b's own smart_resize + its PIL bicubic / (x/255 - 0.5)/0.5 lines
    + s7_pairs_parent.qwen_vl_patchify (loaded through media_b's version-gated `load_s7`).  The image
    bytes must still hash to the teacher item's sha256.  Returns (patch_flat [N, 1536], [h, w])."""

    from PIL import Image

    media_b = trainer_module().load_peer_module("media_b_teacher_cache")
    if file_sha256(image_path) != item_sha256:
        raise D3Error(f"{image_path}: bytes differ from the teacher item's sha256")
    rgb = Image.open(image_path).convert("RGB")
    h, w = media_b.smart_resize(rgb.height, rgb.width, max_pixels=int(max_pixels))
    img = rgb.resize((w, h), Image.BICUBIC)
    pixels = torch.tensor(list(img.getdata()), dtype=torch.float32).view(h, w, 3).permute(2, 0, 1) / 255.0
    pixels = (pixels - 0.5) / 0.5
    flat, _grid, _yx = media_b.load_s7().qwen_vl_patchify(pixels)
    return flat.float(), [int(h), int(w)]


# --------------------------------------------------------------------------
# D3e: speech on the H3 audio channel (speech proposal 2026-10-14 section 4,
# option B; Codex's D3d_audio_ce / D3d_speech_flow, ported).
# --------------------------------------------------------------------------


class prompt_only_unit:
    """Restrict one unit's read AND write scope to rows before `prompt_length` whose type is text (0)
    or audio (3): Codex train_audio.prompt_only_h3.  In D3e_in the clean audio sits inside the prompt
    and the answer follows it, so the H3 unit's joint attention never sees a teacher-forced answer
    token (no answer -> audio -> earlier answer leak).  D3e_out passes the whole length: the sequence
    is generation prompt + noisy audio only."""

    def __init__(self, model: HybridD3, container: str, prompt_length: int) -> None:
        if container not in model.units:
            raise D3Error(f"unit {container!r} is not in the layout")
        self.model, self.container, self.prompt_length = model, container, int(prompt_length)

    def __enter__(self) -> None:
        model, container, prompt_length = self.model, self.container, self.prompt_length
        if "unit_write" in model.__dict__:
            raise D3Error("unit_write is already overridden")
        original = model.unit_write

        def write(placement: UnitPlacement, x: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
            if placement.container != container:
                return original(placement, x, context)
            types = context["row_types"]
            if types is None:
                raise D3Error("prompt-only read scope needs the context's row types")
            positions = None if context["positions"] is None else context["positions"].get(placement.spec.source)
            prefix = torch.arange(types.shape[1], device=types.device).unsqueeze(0) < prompt_length
            visible = context["valid"] & prefix & (types.eq(0) | types.eq(3))
            out = model.units[container](x.to(model.options.unit_dtype), types, context["levels"], visible, source_positions=positions)
            return out.to(x.dtype)

        model.unit_write = write

    def __exit__(self, *exc: Any) -> bool:
        del self.model.unit_write
        return False


def h3_speech_unit(model: HybridD3) -> UnitPlacement:
    placed = [p for p in model.layout if p.source == "h3" and 3 in p.type_ids]
    if len(placed) != 1:
        raise D3Error("D3e needs exactly one H3 unit that owns type 3")
    return placed[0]


def speech_in_targets(model: HybridD3, segment: dict[str, Any], target_mask: torch.Tensor, prompt_length: int,
                      completion_eos_id: int) -> tuple[dict[str, torch.Tensor], int]:
    """D3e_in supervision (Codex train_audio.assistant_ce_targets): one contiguous assistant body that
    starts at `prompt_length` and ends with exactly one completion EOS; CE weight 1 on the positions
    that predict those tokens (target ownership shifted once), nothing else."""

    ids = segment["ids"]
    if (ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.int64 or target_mask.shape != ids.shape
            or target_mask.dtype != torch.bool or not 0 < int(prompt_length) < ids.shape[1]
            or not 0 <= int(completion_eos_id) < model.base_vocab):
        raise D3Error("speech-in needs [1, L] int64 ids, a bool target mask, the prompt boundary and a native EOS id")
    owned = target_mask[0].nonzero().flatten()
    if (owned.numel() < 2 or int(owned[0]) != int(prompt_length)
            or not torch.equal(owned, torch.arange(int(prompt_length), int(owned[-1]) + 1, device=owned.device))
            or int(ids[0, owned[-1]]) != int(completion_eos_id) or int(ids[target_mask].eq(int(completion_eos_id)).sum()) != 1
            or bool(((ids[target_mask] < 0) | (ids[target_mask] >= model.base_vocab)).any())):
        raise D3Error("speech-in supervision must be one native assistant body after the prompt, ending in one EOS")
    n = ids.shape[1]
    next_ids = torch.full((n,), -1, dtype=torch.int64, device=ids.device)
    weights = torch.zeros(n, dtype=torch.float32, device=ids.device)
    selected = target_mask[0, 1:]
    next_ids[:-1] = torch.where(selected, ids[0, 1:], torch.full_like(ids[0, 1:], -1))
    weights[:-1] = selected.float()
    return {"next_ids": next_ids, "ce_w": weights, "eos_w": torch.zeros_like(weights)}, int(owned.numel())


def check_speech_in(model: HybridD3, segment: dict[str, Any], prompt_length: int) -> None:
    """D3e_in input (Codex train_audio.validate_clean_observation): clean (sigma 0) type-3 rows, all
    inside the prompt, on media placeholder ids, features only in the type-3 feature band, finite
    native and H3 source positions, nothing carrying a gradient."""

    ids, types, indices, features = (segment[k] for k in ("ids", "types", "indices", "features"))
    rows = int(types.numel())
    if (types.shape != (1, rows) or types.dtype != torch.int64 or rows < 1 or bool((types != 3).any())
            or indices.shape != types.shape or bool(((indices < 0) | (indices >= int(prompt_length))).any())
            or torch.unique(indices).numel() != rows or features.shape != (1, rows, int(model.cfg.native_feature_dim))
            or segment["sigma"] is None or segment["sigma"].shape != types.shape or bool((segment["sigma"] != 0).any())
            or segment["positions"].shape != (1, rows, 4)):
        raise D3Error("speech-in needs clean (sigma 0) type-3 observation rows wholly inside the prompt")
    if bool((ids[0, indices[0]] != model.media_placeholder).any()):
        raise D3Error("speech-in observation rows must sit on media placeholder ids")
    lo, hi = model.bands[3]["feature"]
    outside = features.clone()
    outside[..., lo:hi] = 0
    source = (segment.get("source_positions") or {}).get("h3")
    if (bool(outside.any()) or source is None or source.shape != (*ids.shape, 3)
            or any(not bool(torch.isfinite(v).all()) for v in (features, segment["positions"], source))
            or any(v.requires_grad for v in (features, segment["positions"], source))):
        raise D3Error("speech-in features must sit in the type-3 band with finite detached native / H3 positions")


def speech_in_head_loss(model: HybridD3, targets: dict[str, torch.Tensor], positions: int, *, kd: dict[str, torch.Tensor] | None = None,
                        w_kd: float = 0.5) -> Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]:
    """D3e_in loss over text rows [0, base): answer CE weight 1.0 (`speech_in_targets`) + optional ASR
    soft-target KD at `w_kd` (speech proposal C-s5: top-32 mapped to native rows) on the same answer
    positions.  Denominator: the supervised answer positions."""

    T = model.T
    full = dict(targets)
    if kd is not None:
        q_w = (targets["ce_w"] > 0).float() * float(w_kd)
        full.update({"q_w": q_w, "q_ids": kd["q_ids"], "q_lp": kd["q_lp"]})
        if bool(((kd["q_ids"][q_w > 0] < 0) | (kd["q_ids"][q_w > 0] >= model.base_vocab)).any()):
            raise D3Error("speech-in KD ids outside the native text rows")
    objective = T.ObjectiveConfig(w_q38=float(w_kd) if kd is not None else 0.0, lambda_k3=0.0, w_ce=1.0, w_eos=0.0, w_ul=0.0,
                                  text_vocab_rows=model.base_vocab, head_chunk_positions=256)

    def loss_fn(normed: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        head = model.engine.head_weight
        parts, grad = T.chunked_objective(normed.to(head.device, head.dtype), head, full, objective,
                                          denominator=float(positions), backward=True)
        surrogate = (normed * grad.to(normed.device, normed.dtype)).sum()
        return surrogate - surrogate.detach() + torch.tensor(float(parts["total"]), device=normed.device)

    return loss_fn


def check_speech_out(model: HybridD3, segment: dict[str, Any], target: torch.Tensor, weights: torch.Tensor, prompt_length: int) -> None:
    """D3e_out input (Codex train_speech_flow.validate_inputs): [generation prompt | noisy type-3 rows]
    in that order, sigma in (0, 1], x_t only in the type-3 noisy sub-band (on OMNI1 the whole type-3
    feature band [480, 512), Codex's rule), the clean x0 only in its emission band (never an
    input), detached, finite, H3 source positions present."""

    ids, types, indices, features, sigma = (segment[k] for k in ("ids", "types", "indices", "features", "sigma"))
    n, width = int(types.numel()), int(model.cfg.native_feature_dim)
    flo, fhi = model.noisy_feature_bands()[3]
    elo, ehi = model.bands[3]["emission"]
    p = int(prompt_length)
    if (p < 1 or ids.shape != (1, p + n) or types.shape != (1, n) or bool((types != 3).any())
            or not torch.equal(indices[0].to("cpu"), torch.arange(p, p + n))
            or bool(((ids[:, :p] < 0) | (ids[:, :p] >= model.base_vocab)).any())
            or bool((ids[:, p:] != model.media_placeholder).any())
            or features.shape != (1, n, width) or target.shape != features.shape or weights.shape != types.shape
            or features.requires_grad or target.requires_grad or sigma is None or sigma.shape != types.shape
            or bool(((sigma <= 0) | (sigma > 1)).any()) or fhi - flo != ehi - elo or not bool((weights == 1).all())):
        raise D3Error("speech-out needs [prompt | noisy type-3 rows], sigma in (0, 1], detached targets and unit row weights")
    outside_in, outside_out = features.clone(), target.clone()
    outside_in[..., flo:fhi] = 0
    outside_out[..., elo:ehi] = 0
    source = (segment.get("source_positions") or {}).get("h3")
    if (bool(outside_in.any()) or bool(outside_out.any()) or source is None or source.shape != (*ids.shape, 3)
            or any(not bool(torch.isfinite(v).all()) for v in (features, target, sigma, source, segment["positions"], weights))):
        raise D3Error("speech-out x_t / x0 outside the type-3 bands, or non-finite inputs / positions")


def speech_out_flow_loss(model: HybridD3, segment: dict[str, Any], target: torch.Tensor,
                         weights: torch.Tensor) -> Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]:
    """D3e_out (`supervised_speech_rectified_flow`): x0 head, x0 targets, velocity-form MSE with x_t read
    from the type-3 noisy sub-band (OMNI1: the whole type-3 feature band, as Codex passes it)."""

    noisy = {3: tuple(model.noisy_feature_bands()[3])}

    def loss_fn(normed: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        prediction = model.emission(normed, context)
        return flow_velocity_loss(prediction, target.to(prediction.device), segment["types"].to(prediction.device),
                                  weights.to(prediction.device), model.bands, prediction_kind="x0", target_kind="x0",
                                  x_t=segment["features"].to(prediction.device), sigma=segment["sigma"].to(prediction.device),
                                  noisy_feature_bands=noisy)

    return loss_fn


def speech_lane_step(model: HybridD3, lane: str, segment: dict[str, Any], *, prompt_length: int,
                     target_mask: torch.Tensor | None = None, completion_eos_id: int | None = None, kd: dict[str, torch.Tensor] | None = None,
                     target: torch.Tensor | None = None, weights: torch.Tensor | None = None, loss_scale: float = 1.0) -> dict[str, Any]:
    """One D3e forward + backward (no optimiser step): the input checks, then the full-depth streamed
    pass with the H3 unit's prompt-only read scope.  The caller owns the optimiser (fresh per stage)."""

    unit = h3_speech_unit(model)
    if lane == "speech_in_ce":
        check_speech_in(model, segment, prompt_length)
        targets, count = speech_in_targets(model, segment, target_mask, prompt_length, completion_eos_id)
        loss_fn = speech_in_head_loss(model, targets, count, kd=kd)
        scope = prompt_length
        extra = {"answer_positions": count}
    elif lane == "speech_out_flow":
        check_speech_out(model, segment, target, weights, prompt_length)
        loss_fn = speech_out_flow_loss(model, segment, target, weights)
        scope = int(segment["ids"].shape[1])
        extra = {"audio_rows": int(segment["types"].numel()), "sigma": sorted({round(float(s), 6) for s in segment["sigma"].flatten()})}
    else:
        raise D3Error(f"speech lane {lane!r}: speech_in_ce or speech_out_flow")
    with prompt_only_unit(model, unit.container, scope):
        loss, _, _ = full_depth_forward_backward(model, segment, loss_fn, loss_scale=loss_scale)
    return {"lane": lane, "loss": loss, "unit": unit.container, **extra}


def h3_replay_schedule(speech_ids: list[str], replay_ids: list[str], *, fraction: float = H3_REPLAY_FRACTION,
                       seed: int = 20261013) -> dict[str, Any]:
    """D3e_out's step order with `fraction` H3 AV replay (speech proposal 4.2): replay count =
    round(n_speech * f / (1 - f)), each replay item used AT MOST once (a short replay pool is used
    once each and its deficit reported, never duplicated -- Codex's D4 rule), replay steps spread
    evenly through the speech steps by a seeded permutation."""

    if not 0.0 <= float(fraction) < 1.0:
        raise D3Error("replay fraction must be in [0, 1)")
    if len(set(speech_ids)) != len(speech_ids) or len(set(replay_ids)) != len(replay_ids):
        raise D3Error("speech and replay ids must be distinct")
    if set(speech_ids) & set(replay_ids):
        raise D3Error("an id is both a speech row and an H3 replay row")
    wanted = int(round(len(speech_ids) * float(fraction) / (1.0 - float(fraction))))
    g = torch.Generator().manual_seed(int(seed))
    pool = [replay_ids[i] for i in torch.randperm(len(replay_ids), generator=g).tolist()]
    chosen = pool[:wanted]
    total = len(speech_ids) + len(chosen)
    # slot k = floor((2k + 1) * total / (2R)) in integers: consecutive slots differ by >= 1 because
    # total >= R, so the R slots are distinct and < total
    slot_set = {((2 * k + 1) * total) // (2 * len(chosen)) for k in range(len(chosen))} if chosen else set()
    order: list[dict[str, str]] = []
    s_iter, r_iter = iter(speech_ids), iter(chosen)
    for position in range(total):
        if position in slot_set:
            order.append({"kind": "h3_replay", "id": next(r_iter)})
        else:
            order.append({"kind": "speech", "id": next(s_iter)})
    return {"order": order, "speech": len(speech_ids), "replay": len(chosen), "replay_wanted": wanted,
            "replay_deficit": max(0, wanted - len(chosen)), "fraction_realised": len(chosen) / total if total else 0.0, "seed": int(seed)}


#: the type-3 position charts A3 chooses between (speech proposal 4.4 item 1)
TYPE3_CHART_PLANS = ("audio40hz", "zero")


def type3_chart(text_count: int, frames: int, *, plan: str, channels: int = 2) -> torch.Tensor:
    """H3 source positions [channels * frames, 3] for type-3 rows in channel-major stereo order (row r:
    channel c = r // frames, frame i = r % frames).  "audio40hz": Codex's measured 40 Hz chart
    (text_count + i, 0, c); "zero": the s7 zero plan.  D3b's H3 lane and both D3e lanes build their
    type-3 positions through this one function, so one plan gives identical rotary angles in all of
    them (A3 picks the plan)."""

    if plan not in TYPE3_CHART_PLANS:
        raise D3Error(f"type-3 chart plan {plan!r}: one of {TYPE3_CHART_PLANS}")
    if int(frames) < 1 or int(channels) < 1 or int(text_count) < 0:
        raise D3Error("type-3 chart needs frames >= 1, channels >= 1, text_count >= 0")
    rows = int(frames) * int(channels)
    if plan == "zero":
        return torch.zeros(rows, 3)
    r = torch.arange(rows)
    return torch.stack([(int(text_count) + r % int(frames)).float(), torch.zeros(rows), (r // int(frames)).float()], dim=-1)


# --------------------------------------------------------------------------
# Native state across D3 stages + exact stage checkpoints (Codex
# native_state_inheritance / stage_inheritance, ported).
# --------------------------------------------------------------------------


def native_inventory(model: HybridD3) -> dict[str, Any]:
    """Shapes / dtypes of everything a native state may carry, plus the source and layout identity."""

    def describe(module: nn.Module) -> dict[str, Any]:
        return {name: {"shape": list(p.shape), "dtype": str(p.dtype)} for name, p in sorted(module.named_parameters())}

    return json.loads(json.dumps({
        "units": {name: describe(unit) for name, unit in sorted(model.units.items())},
        "bridge": describe(model.bridge), "emission": describe(model.media_head),
        "bands": {str(k): {"feature": list(v["feature"]), "emission": list(v["emission"])} for k, v in sorted(model.bands.items())},
        "layout": [{"container": p.container, "depth": p.depth, "type_ids": list(p.type_ids)} for p in model.layout],
        "music": {"begin": model.music_band_begin, "end": model.music_band_end, "base_vocab": model.base_vocab, "hidden": model.hidden},
        "lora": None if model.bank is None else {n: list(model.bank.tensors[n].shape) for n in model.bank.names},
        "source": model.source.identity()}, sort_keys=True))


def _checked_interface_value(model: HybridD3, kind: str, desc: dict[str, Any], value: torch.Tensor) -> None:
    module = model.bridge if kind == "bridge" else model.media_head
    allowed = {"bridge": {"feature_proj.weight": (1, "feature"), "type_embed.weight": (0, None)},
               "emission": {"out.weight": (0, "emission"), "type_affine.weight": (0, None)}}[kind]
    name, axis, ranges = desc.get("name"), desc.get("axis"), desc.get("ranges")
    if name not in allowed or axis != allowed[name][0] or list(desc.get("shape", [])) != list(dict(module.named_parameters())[name].shape):
        raise D3Error(f"{kind} descriptor {desc} is not a native interface weight / axis")
    band = allowed[name][1]
    valid = {tuple(model.bands[t][band]) if band else (t, t + 1) for t in model.bands}
    if not ranges or any(tuple(r) not in valid for r in ranges) or sorted(map(tuple, ranges)) != [tuple(r) for r in ranges]:
        raise D3Error(f"{kind}.{name}: ranges {ranges} are not sorted native ownership bands")
    if list(value.shape) != list(desc["shape"]) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise D3Error(f"{kind}.{name}: master must be a finite fp32 tensor of the weight's shape")
    keep = torch.zeros(value.shape[axis], dtype=torch.bool)
    for a, b in ranges:
        keep[a:b] = True
    outside = value[~keep] if axis == 0 else value[:, ~keep]
    if bool((outside != 0).any()):
        raise D3Error(f"{kind}.{name}: master writes outside its owned ranges")


def cumulative_interfaces(model: HybridD3) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    """Inherited + current-stage interface deltas as fp32 full-shape masters, with merged descriptors
    (keys "bridge.<param>" / "emission.<param>")."""

    values: dict[str, torch.Tensor] = {}
    descriptors: dict[str, dict[str, Any]] = {}
    for kind, inherited, deltas in (("bridge", model.inherited_bridge, model.bridge_deltas), ("emission", model.inherited_head, model.head_deltas)):
        current = {d.name: d for d in deltas}
        for name in sorted(set(inherited) | set(current)):
            key = f"{kind}.{name}"
            value = torch.zeros(0)
            ranges: set[tuple[int, int]] = set()
            desc: dict[str, Any] | None = None
            if name in inherited:
                value = inherited[name].detach().float().cpu().clone()
                desc = dict(model.inherited_descriptors[key])
                ranges |= {tuple(r) for r in desc["ranges"]}
            if name in current:
                d = current[name]
                scattered = d.scatter("cpu").detach()
                value = scattered if desc is None else value + scattered
                cur = d.descriptor(kind)
                if desc is not None and (desc["axis"] != cur["axis"] or desc["shape"] != cur["shape"]):
                    raise D3Error(f"{key}: inherited and current deltas differ in axis / shape")
                desc = desc or cur
                ranges |= {tuple(r) for r in cur["ranges"]}
            desc = {**desc, "kind": kind, "name": name, "ranges": [list(r) for r in sorted(ranges)]}
            _checked_interface_value(model, kind, desc, value)
            values[key], descriptors[key] = value.contiguous(), desc
    return values, descriptors


def export_native_state(model: HybridD3, out: str | Path, *, stage: str, step: int, total_steps: int,
                        trained_units: list[str] | tuple[str, ...] = (), band: BandTrainer | None = None,
                        order: tuple[str, ...] = D3_STAGE_ORDER) -> dict[str, Any]:
    """Write the cumulative native state after a COMPLETE stage (step == total_steps): units ever
    trained (fp32 replacements), cumulative fp32 bridge / emission masters, music tables (fp32, all
    allocated rows; from `band` when this stage trained them, else the inherited ones), the whole
    shared LoRA bank.  `out` must not exist.  Returns the sealed manifest."""

    from safetensors.torch import save_file

    stage = canonical_stage(stage)
    predecessor(stage, order)   # membership check
    if type(total_steps) is not int or total_steps < 1 or int(step) != total_steps:
        raise D3Error(f"a native state is exported after the whole declared budget (step {step} of {total_steps})")
    out = Path(out)
    if out.exists():
        raise D3Error(f"{out} exists; a native state is written once")
    values: dict[str, torch.Tensor] = {}
    descriptors: dict[str, dict[str, Any]] = {}
    for unit in sorted(set(model.inherited_units) | set(trained_units)):
        if unit not in model.units:
            raise D3Error(f"unit {unit!r} is not in the layout")
        for name, p in model.units[unit].named_parameters():
            if name in ("h3_time.curve_table", "attn.rope_inv_freq"):
                continue
            key = f"unit.{unit}.{name}"
            values[key] = p.detach().float().cpu().contiguous().clone()
            descriptors[key] = {"kind": "unit", "shape": list(p.shape)}
    masters, descs = cumulative_interfaces(model)
    values.update(masters)
    descriptors.update(descs)
    music = None
    if band is not None:
        music = model.music_contract(band.codes)
        band.check_frozen()
        values.update(band.music_tables())
    elif model.inherited_band_head is not None:
        music = model.music_contract(model.inherited_descriptors["music.head"]["written_rows"])
        o = model.music_offset
        values["music.embed"] = model.band_embed_override[o:o + model.music_rows].detach().float().cpu().clone()
        values["music.head"] = model.inherited_band_head.detach().float().cpu().clone()
    if music is not None:
        c = music["written_rows"]
        if (tensor_sha256(values["music.embed"][c:]) != music["reserved_embed_sha256"]
                or tensor_sha256(values["music.head"][c:]) != music["reserved_head_sha256"]):
            raise D3Error("music tables changed a reserved row")
        for key in ("music.embed", "music.head"):
            descriptors[key] = {"kind": "music", "shape": list(values[key].shape), "written_rows": c}
    if model.bank is not None:
        for name in model.bank.names:
            values[f"lora.{name}"] = model.bank.tensors[name].detach().float().cpu().contiguous().clone()
            descriptors[f"lora.{name}"] = {"kind": "lora", "shape": list(model.bank.tensors[name].shape)}
    if not values or any(not bool(torch.isfinite(v).all()) for v in values.values()):
        raise D3Error("a native state must hold finite learned tensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.partial"
    if staging.exists():
        raise D3Error(f"{staging} exists (an interrupted export): remove it after review")
    staging.mkdir()
    save_file(values, str(staging / "native_state.safetensors"), metadata={"schema": NATIVE_STATE_SCHEMA})
    manifest = {"schema": NATIVE_STATE_SCHEMA, "tool": tool_identity(), "stage": stage, "order": list(order),
                "step": int(step), "total_steps": int(total_steps), "complete": True,
                "parent_manifest_sha256": model.native_parent,
                "semantics": {"unit": "fp32 replacement", "bridge": "cumulative fp32 delta", "emission": "cumulative fp32 delta",
                              "music": "fp32 master replacement (all allocated rows; reserved rows = base)", "lora": "shared r16 bank replacement"},
                "tensors": {"file": "native_state.safetensors", "sha256": file_sha256(staging / "native_state.safetensors")},
                "descriptors": descriptors, "music_contract": music, "inventory": native_inventory(model),
                "optimizer_state_exported": False, "registered": False}
    manifest["seal_sha256"] = canonical_sha256(manifest)
    trainer_module().write_json_atomic(staging / "manifest.json", manifest)
    staging.rename(out)
    return manifest


def import_native_state(model: HybridD3, directory: str | Path, *, stage: str, order: tuple[str, ...] = D3_STAGE_ORDER) -> dict[str, Any]:
    """Load the previous stage's native state into a FRESH hybrid before any optimiser exists.

    Refuses: a manifest whose seal or tensor file moved; a producer that is not the immediate
    predecessor of `stage` in `order`; a layout / shape / source inventory that differs from this
    hybrid's; an interface master outside its ownership bands; music tables whose ledger or reserved
    rows differ from this base; a hybrid that already has deltas, overrides or gradients."""

    from safetensors.torch import load_file

    directory = Path(directory)
    stage = canonical_stage(stage)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != NATIVE_STATE_SCHEMA or manifest.get("complete") is not True:
        raise D3Error(f"{directory} is not a complete native state")
    if canonical_sha256({k: v for k, v in manifest.items() if k != "seal_sha256"}) != manifest.get("seal_sha256"):
        raise D3Error("native state manifest seal changed")
    if file_sha256(directory / manifest["tensors"]["file"]) != manifest["tensors"]["sha256"]:
        raise D3Error("native state tensors do not hash like the manifest")
    want = predecessor(stage, order)
    if manifest.get("stage") != want or tuple(manifest.get("order", ())) != tuple(order):
        raise D3Error(f"{stage} inherits from {want!r} only (the state is from {manifest.get('stage')!r}, order {manifest.get('order')})")
    if (model.head_deltas or model.bridge_deltas or model.band_embed_override is not None or model.inherited_bridge
            or model.inherited_head or model.inherited_band_head is not None or model.native_parent is not None):
        raise D3Error("native state imports once, into a fresh hybrid, before any stage delta")
    if any(p.grad is not None for unit in model.units.values() for p in unit.parameters()):
        raise D3Error("native state import must precede every backward")
    if manifest["inventory"] != native_inventory(model):
        raise D3Error("native state layout / shape / source inventory differs from this hybrid")
    values = load_file(str(directory / manifest["tensors"]["file"]), device="cpu")
    descriptors = manifest["descriptors"]
    if set(values) != set(descriptors) or any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in values.values()):
        raise D3Error("native state tensors are incomplete, non-finite or not fp32")
    units: set[str] = set()
    unit_params = {f"unit.{u}.{n}": p for u, m in model.units.items() for n, p in m.named_parameters()}
    for key, desc in descriptors.items():
        if desc["kind"] == "unit":
            if key not in unit_params or list(values[key].shape) != list(unit_params[key].shape):
                raise D3Error(f"{key}: no such unit parameter of that shape")
            units.add(key.split(".")[1])
        elif desc["kind"] in ("bridge", "emission"):
            if key != f"{desc['kind']}.{desc['name']}":
                raise D3Error(f"{key}: descriptor name mismatch")
            _checked_interface_value(model, desc["kind"], desc, values[key])
        elif desc["kind"] == "music":
            pass
        elif desc["kind"] == "lora":
            if model.bank is None or key[len("lora."):] not in model.bank.tensors:
                raise D3Error(f"{key}: this hybrid has no such LoRA tensor")
        else:
            raise D3Error(f"{key}: unknown native state family {desc['kind']!r}")
    music = manifest.get("music_contract")
    if music is not None:
        if music != model.music_contract(music["written_rows"]):
            raise D3Error("music tables were written for another base (ledger / reserved rows differ)")
        if {"music.embed", "music.head"} - set(values):
            raise D3Error("music contract without both music tables")
        c = music["written_rows"]
        if (tensor_sha256(values["music.embed"][c:]) != music["reserved_embed_sha256"]
                or tensor_sha256(values["music.head"][c:]) != music["reserved_head_sha256"]):
            raise D3Error("imported music tables move a reserved row")
    # every check passed: mutate
    with torch.no_grad():
        for key, desc in descriptors.items():
            if desc["kind"] == "unit":
                unit_params[key].copy_(values[key].to(unit_params[key].device, unit_params[key].dtype))
            elif desc["kind"] == "lora":
                t = model.bank.tensors[key[len("lora."):]]
                t.copy_(values[key].to(t.device, t.dtype))
    model.inherited_bridge = {d["name"]: values[k].to(model.device) for k, d in descriptors.items() if d["kind"] == "bridge"}
    model.inherited_head = {d["name"]: values[k].to(model.device) for k, d in descriptors.items() if d["kind"] == "emission"}
    model.inherited_descriptors = {k: d for k, d in descriptors.items() if d["kind"] in ("bridge", "emission", "music")}
    model.inherited_units = units
    if music is not None:
        o = model.music_offset
        table = model.band_embed.float().clone()
        table[o:o + model.music_rows] = values["music.embed"]
        model.band_embed_override = table
        model.inherited_band_head = values["music.head"].clone()
    model.native_parent = file_sha256(manifest_path)
    return {"source_manifest_sha256": model.native_parent, "source_stage": manifest["stage"], "stage": stage,
            "units": sorted(units), "interfaces": sorted(k for k, d in descriptors.items() if d["kind"] in ("bridge", "emission")),
            "music": music is not None, "lora_tensors": sum(1 for d in descriptors.values() if d["kind"] == "lora"),
            "optimizer_state_imported": False}


def save_stage_checkpoint(path: str | Path, *, identity: dict[str, Any], total_steps: int, step: int, sampler_counter: int,
                          trainables: dict[str, torch.Tensor], optimizers: dict[str, Any] | None = None,
                          lazy: dict[str, LazyRowAdam] | None = None, tables: dict[str, torch.Tensor] | None = None) -> str:
    """An exact-resume checkpoint inside one stage: trainables, torch optimiser states, lazy-row Adam
    states, extra fp32 tables (music masters), step / sampler counter, RNG, and `total_steps` pinned
    in both the state and the receipt.  Written to a staging dir and renamed; `path` must not exist."""

    path = Path(path)
    if path.exists():
        raise D3Error(f"{path} exists; a checkpoint is written once")
    if type(total_steps) is not int or total_steps < 1 or not 0 <= int(step) <= total_steps:
        raise D3Error(f"checkpoint step {step} outside the declared budget {total_steps}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.partial"
    if staging.exists():
        raise D3Error(f"{staging} exists (an interrupted checkpoint): remove it after review")
    staging.mkdir()
    import random as _random

    state = {"schema": STAGE_CHECKPOINT_SCHEMA, "identity_sha256": canonical_sha256(identity), "total_steps": int(total_steps),
             "step": int(step), "sampler_counter": int(sampler_counter),
             "trainables": {k: v.detach().cpu().clone() for k, v in trainables.items()},
             "optimizers": {k: o.state_dict() for k, o in (optimizers or {}).items()},
             "lazy": {k: {"m": a.m.clone(), "v": a.v.clone(), "steps": a.steps.clone(), "rows_updated": a.rows_updated, "lr": a.lr}
                      for k, a in (lazy or {}).items()},
             "tables": {k: v.detach().cpu().clone() for k, v in (tables or {}).items()},
             "rng_cpu": torch.get_rng_state(), "rng_python": _random.getstate()}
    torch.save(state, staging / "state.pt")
    receipt = {"schema": STAGE_CHECKPOINT_SCHEMA, "identity": identity, "identity_sha256": state["identity_sha256"],
               "total_steps": int(total_steps), "step": int(step), "sampler_counter": int(sampler_counter),
               "state_sha256": file_sha256(staging / "state.pt"), "trainable_census": {k: list(v.shape) for k, v in sorted(trainables.items())},
               "complete": True, "tool": tool_identity()}
    trainer_module().write_json_atomic(staging / "receipt.json", receipt)
    staging.rename(path)
    return file_sha256(path / "receipt.json")


def load_stage_checkpoint(path: str | Path, *, identity: dict[str, Any], total_steps: int, trainables: dict[str, torch.Tensor],
                          optimizers: dict[str, Any] | None = None, lazy: dict[str, LazyRowAdam] | None = None,
                          tables: dict[str, torch.Tensor] | None = None) -> dict[str, int]:
    """Exact resume: refuses a changed identity, a missing or different `total_steps`, a different
    trainable / optimiser / table census or a state file that does not hash like its receipt; then
    restores tensors in place, optimiser states, lazy-row states and RNG.  Returns step / sampler."""

    import random as _random

    path = Path(path)
    receipt = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema") != STAGE_CHECKPOINT_SCHEMA or receipt.get("complete") is not True:
        raise D3Error(f"{path} is not a complete stage checkpoint")
    if file_sha256(path / "state.pt") != receipt.get("state_sha256"):
        raise D3Error("stage checkpoint state does not hash like its receipt")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    if "total_steps" not in state or "total_steps" not in receipt:
        raise D3Error("stage checkpoint has no declared total_steps (exact resume needs the pinned schedule length)")
    if state["total_steps"] != int(total_steps) or receipt["total_steps"] != int(total_steps):
        raise D3Error(f"stage checkpoint total_steps {state['total_steps']} != this run's {total_steps}")
    if state["identity_sha256"] != canonical_sha256(identity) or receipt.get("identity") != json.loads(json.dumps(identity)):
        raise D3Error("exact resume refused: base, data, objective or config identity changed")
    if (set(state["trainables"]) != set(trainables) or set(state["optimizers"]) != set(optimizers or {})
            or set(state["lazy"]) != set(lazy or {}) or set(state["tables"]) != set(tables or {})):
        raise D3Error("exact resume refused: trainable / optimiser / table census changed")
    for key, t in {**trainables, **(tables or {})}.items():
        saved = state["trainables"].get(key, state["tables"].get(key))
        if tuple(saved.shape) != tuple(t.shape):
            raise D3Error(f"{key}: checkpoint shape {tuple(saved.shape)} != {tuple(t.shape)}")
    with torch.no_grad():
        for key, t in trainables.items():
            t.copy_(state["trainables"][key].to(t.device, t.dtype))
        for key, t in (tables or {}).items():
            t.copy_(state["tables"][key].to(t.device, t.dtype))
    for key, opt in (optimizers or {}).items():
        opt.load_state_dict(state["optimizers"][key])
    for key, opt in (lazy or {}).items():
        saved = state["lazy"][key]
        opt.m.copy_(saved["m"])
        opt.v.copy_(saved["v"])
        opt.steps.copy_(saved["steps"])
        opt.rows_updated = int(saved["rows_updated"])
        opt.lr = float(saved["lr"])
    torch.set_rng_state(state["rng_cpu"])
    _random.setstate(state["rng_python"])
    return {"step": int(state["step"]), "sampler_counter": int(state["sampler_counter"])}


# --------------------------------------------------------------------------
# D4 driver prep: authentic receipt joins and the consolidation guards
# (Codex consolidation / consolidation_dispatch findings, ported).
# --------------------------------------------------------------------------


def sigma_grid_of(manifest: dict[str, Any]) -> tuple[float, ...]:
    """The producer's noise grid: Codex's producers write `sigma_grid` (geometry, image edit, speech);
    older readers expected `sigmas`.  Either is accepted; both present and different is a refusal;
    neither is a refusal."""

    grid, legacy = manifest.get("sigma_grid"), manifest.get("sigmas")
    if grid is None and legacy is None:
        raise D3Error("manifest declares no sigma_grid / sigmas")
    if grid is not None and legacy is not None and [float(v) for v in grid] != [float(v) for v in legacy]:
        raise D3Error(f"manifest sigma_grid {grid} contradicts sigmas {legacy}")
    values = tuple(float(v) for v in (grid if grid is not None else legacy))
    if not values or any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in values):
        raise D3Error(f"sigma grid {values} is not a non-empty set of values in [0, 1]")
    return values


#: Codex stage-cache schemas whose receipts carry `tensors_sha256` at the top level
CODEX_TOP_LEVEL_SCHEMAS = ("chimera_recovery.speech_flow_cache.v1", "chimera_recovery.audio_cache.v1", "chimera_recovery.music_cache.v1",
                           "chimera_recovery.vision_cache.v1", "chimera_recovery.refit_cache.v1")
CLAUDE_SHARD_SCHEMA = "omni-campaign.teacher_cache_shard.v2"


def cache_identity(receipt: dict[str, Any], *, item: dict[str, Any] | None = None, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """One normalised identity for a cache receipt, read in its AUTHENTIC producer schema:

    - Codex `chimera_recovery.native_flow_cache.v1` (guided ACE / QIMG / H3): the tensor file is
      `cache_sha256`, the source identity is the NESTED `source` (manifest / shard / sidecar shas,
      recovery_group_id, item_id or source_id).  A top-level `tensors_sha256` / `recovery_group_id`,
      if present, must agree (no relabelling; Codex's D4 had assumed top-level fields).
    - Codex stage caches (speech flow / audio CE / music / vision / refit): top-level `tensors_sha256`,
      `recovery_group_id`, `source_row_sha256`.
    - Claude `media_b_teacher_cache` shard sidecars: `file_sha256`, `item_id`, `item_sha256`,
      `corpus_sha256`, `config_sha256` (+ the sealed cache manifest's sha when given).
    `item` (a batch-manifest entry with tensors.sha256) and `manifest` (the batch manifest) are
    cross-checked when given."""

    schema = receipt.get("schema")
    if schema == "chimera_recovery.native_flow_cache.v1":
        source = receipt.get("source")
        if not isinstance(source, dict):
            raise D3Error("native_flow_cache receipt has no nested source identity")
        tensors = receipt.get("cache_sha256")
        group = source.get("recovery_group_id")
        for key in ("manifest_sha256", "shard_sha256", "sidecar_sha256"):
            if not is_sha256(source.get(key)):
                raise D3Error(f"native_flow_cache source.{key} is not a sha256")
        source_id = source.get("source_id", source.get("item_id"))
        identity = {"source_manifest_sha256": source["manifest_sha256"], "source_shard_sha256": source["shard_sha256"],
                    "source_sidecar_sha256": source["sidecar_sha256"], "source_id": source_id}
        for key, expected in (("tensors_sha256", tensors), ("recovery_group_id", group)):
            if key in receipt and receipt[key] != expected:
                raise D3Error(f"contradictory receipt: top-level {key} {receipt[key]!r} != nested / cache value {expected!r}")
        if receipt.get("split", "train") != "train" or receipt.get("held", False) is not False:
            raise D3Error("guided cache receipt is not a train-split, non-held row")
        if manifest is not None and manifest.get("source_cache_manifest_sha256") not in (None, source["manifest_sha256"]):
            raise D3Error("batch manifest source cache differs from the receipt's nested source manifest")
    elif schema in CODEX_TOP_LEVEL_SCHEMAS:
        tensors, group = receipt.get("tensors_sha256"), receipt.get("recovery_group_id")
        identity = {"source_row_sha256": receipt.get("source_row_sha256"), "source_id": receipt.get("source_id", receipt.get("id"))}
        if receipt.get("split") != "train" or receipt.get("held") is not False:
            raise D3Error(f"{schema} receipt is not a train-split, non-held row")
    elif schema == CLAUDE_SHARD_SCHEMA:
        tensors, group = receipt.get("file_sha256"), None
        identity = {"item_id": receipt.get("item_id"), "item_sha256": receipt.get("item_sha256"),
                    "corpus_sha256": receipt.get("corpus_sha256"), "config_sha256": receipt.get("config_sha256"),
                    "teacher": receipt.get("teacher")}
        if (receipt.get("item") or {}).get("held"):
            raise D3Error("Claude teacher shard is a held item")
        if manifest is not None:
            shard = next((s for s in manifest.get("shards", []) if s.get("name") == receipt.get("file")), None)
            if shard is None or shard.get("sha256") != tensors:
                raise D3Error("Claude shard sidecar is not in its sealed cache manifest with this sha")
    else:
        raise D3Error(f"unknown cache receipt schema {schema!r}")
    if not is_sha256(tensors):
        raise D3Error(f"{schema}: tensor file sha256 missing")
    if item is not None and (item.get("tensors") or {}).get("sha256") != tensors:
        raise D3Error("batch item tensors sha differs from the receipt's cache sha")
    if item is not None and group is not None and item.get("recovery_group_id") not in (None, group):
        raise D3Error("batch item recovery group differs from the receipt's")
    return {"schema": schema, "tensors_sha256": tensors, "recovery_group_id": group, "source": identity}


def inflate_native_batch(stored: dict[str, torch.Tensor], contract: dict[str, Any], *, width: int) -> dict[str, torch.Tensor]:
    """Codex's compact `chimera_recovery.native_band.v1` storage -> dense [B, M, width] feature / target
    (/ channel-mask) tensors (Codex media_cache.inflate_native_batch semantics: every stored band
    consumed exactly once, shapes / dtypes as the allocation says, no dense key stored beside its bands)."""

    if contract.get("schema") != "chimera_recovery.native_band.v1" or int(contract.get("width", -1)) != int(width):
        raise D3Error(f"unsupported compact native storage {contract.get('schema')!r} width {contract.get('width')}")
    types = stored["native_media_type_ids"]
    bands = contract.get("bands")
    if not isinstance(bands, dict) or set(bands) != {str(k) for k in types.unique().tolist()}:
        raise D3Error("compact allocation differs from the batch's native types")
    keys = {"native_media_features": "feature", "native_media_targets": "emission"}
    if contract.get("channel_loss_mask") is True:
        keys["native_media_channel_loss_mask"] = "emission"
    elif contract.get("channel_loss_mask") is not False:
        raise D3Error("compact channel-mask presence must be explicit")
    consumed: set[str] = set()
    out = {k: v for k, v in stored.items() if ".band" not in k}
    for key, layout in keys.items():
        if key in stored:
            raise D3Error(f"dense {key} stored beside its compact bands")
        dense = torch.zeros((*types.shape, int(width)), dtype=torch.bool if key.endswith("channel_loss_mask") else torch.float32)
        for kind, band in bands.items():
            lo, hi = (int(v) for v in band[layout])
            name = f"{key}.band{kind}"
            selected = types == int(kind)
            value = stored[name]
            if tuple(value.shape) != (int(selected.sum()), hi - lo) or value.dtype != dense.dtype:
                raise D3Error(f"{name}: shape / dtype differs from the allocation")
            dense[..., lo:hi][selected] = value
            consumed.add(name)
        out[key] = dense
    if {k for k in stored if ".band" in k} != consumed:
        raise D3Error("unrecognised compact band tensors in the batch")
    return out


def native_batch_segment(batch: dict[str, torch.Tensor], *, media_placeholder: int) -> dict[str, Any]:
    """A Codex native batch (one unpadded sample) as a HybridD3 segment; the media rows must sit on the
    base's media placeholder id (Codex train_media.make_segment)."""

    if batch["input_ids"].shape[0] != 1 or not bool(batch["native_media_valid_mask"].all()):
        raise D3Error("HybridD3 runs one unpadded sample per call")
    if not bool((batch["input_ids"].gather(1, batch["native_media_indices"]) == int(media_placeholder)).all()):
        raise D3Error("native media rows are not on this base's media placeholder id")
    seg = {"ids": batch["input_ids"], "features": batch["native_media_features"], "types": batch["native_media_type_ids"],
           "positions": batch["native_media_positions"], "indices": batch["native_media_indices"],
           "sigma": batch["native_media_noise_levels"], "media_rows": int(batch["native_media_type_ids"].shape[1]),
           "prompt_tokens": int(batch["native_media_indices"].min())}
    if "charted_source_positions" in batch:
        seg["source_positions"] = {"h3": batch["charted_source_positions"]}
    return seg


#: Codex training.ROUTES, its D3b (flow) routes: route -> the native types the route may supervise
D3B_FLOW_ROUTES = {"image_generate": (1,), "image_edit": (1,), "audio_generate": (3,), "music_generate": (4,),
                   "joint_av": (2, 3), "volume_generate": (7,)}
D3B_ROUTE_ALIASES = {"image": "image_generate", "audio": "audio_generate", "music": "music_generate", "h3": "joint_av",
                     "volume": "volume_generate"}
#: OMNI1's C18 band start (Codex training.MUSIC_BEGIN; the ACE conditioning code ids are read against it)
OMNI1_MUSIC_BEGIN = 252_416


def flow_route(route: str) -> str:
    name = D3B_ROUTE_ALIASES.get(str(route), str(route))
    if name not in D3B_FLOW_ROUTES:
        raise D3Error(f"unsupported D3b flow route {route!r} (no silent fallback)")
    return name


def codex_tensor_sha256(tensor: torch.Tensor) -> str:
    """Codex `training.tensor_sha256` (the digest its receipts' conditioning bindings use): sha256 of
    the dtype name, the JSON shape and the contiguous CPU bytes.  Not `tensor_sha256` above (another
    header layout)."""

    value = tensor.detach().contiguous().cpu()
    out = hashlib.sha256()
    out.update(str(value.dtype).replace("torch.", "").encode())
    out.update(json.dumps(list(value.shape)).encode())
    out.update(value.numpy().tobytes())
    return out.hexdigest()


def check_native_flow_batch(route: str, batch: dict[str, torch.Tensor]) -> None:
    """Codex `training.validate_native_batch`, D3b part, on one native flow batch: shapes / dtypes, media
    indices inside the sequence, the route's own types only, finite inputs / targets / positions, a
    detached target that does not alias the inputs, row weights finite, >= 0, zero on padding rows and
    positive somewhere, sigma in (0, 1] on valid rows, a channel mask with real channels on every
    supervised row and none on padding, and authentic [B, S, 3] H3 source positions for types 2 / 3."""

    route = flow_route(route)
    ids, features, target = batch["input_ids"], batch["native_media_features"], batch["native_media_targets"]
    types, indices, valid = batch["native_media_type_ids"], batch["native_media_indices"], batch["native_media_valid_mask"]
    positions, weights, sigma = batch["native_media_positions"], batch["native_media_loss_mask"], batch["native_media_noise_levels"]
    shape = tuple(types.shape)
    if (ids.ndim != 2 or ids.dtype != torch.int64 or features.ndim != 3 or tuple(features.shape[:2]) != shape
            or features.shape[0] != ids.shape[0] or types.dtype != torch.int64 or tuple(indices.shape) != shape
            or indices.dtype != torch.int64 or tuple(valid.shape) != shape or valid.dtype != torch.bool
            or tuple(positions.shape) != (*shape, 4) or target.shape != features.shape or tuple(weights.shape) != shape
            or tuple(sigma.shape) != shape):
        raise D3Error("native flow batch: int64 ids [B, L], features / targets [B, M, F], int64 types / indices, bool valid, "
                      "positions [B, M, 4], weights / sigma [B, M]")
    if not bool(valid.any()) or bool(((indices < 0) | (indices >= ids.shape[1]))[valid].any()):
        raise D3Error("valid native rows need real in-sequence media indices")
    active = sorted(int(t) for t in torch.unique(types[valid]).tolist())
    if any(t not in D3B_FLOW_ROUTES[route] for t in active):
        raise D3Error(f"native types {active} differ from route {route}'s {list(D3B_FLOW_ROUTES[route])}")
    if target.requires_grad or target.data_ptr() == features.data_ptr():
        raise D3Error("target leakage: the teacher target carries a graph or aliases the input features")
    if (not bool(torch.isfinite(features[valid]).all()) or not bool(torch.isfinite(positions[valid]).all())
            or not bool(torch.isfinite(target[valid]).all())):
        raise D3Error("non-finite native input, position or target")
    if (not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or bool((weights[~valid] != 0).any())
            or not bool((weights > 0).any())):
        raise D3Error("flow loss weights are non-finite / negative, supervise padding, or supervise no media row")
    if not bool(torch.isfinite(sigma[valid]).all()) or bool(((sigma <= 0) | (sigma > 1))[valid].any()):
        raise D3Error("flow training needs finite 0 < sigma <= 1 on every valid row")
    channels = batch.get("native_media_channel_loss_mask")
    if channels is not None and (channels.shape != target.shape or channels.dtype != torch.bool or bool(channels[~valid].any())
                                 or not bool(channels[weights > 0].any(-1).all())):
        raise D3Error("flow channel mask must be bool [B, M, F] with real channels on supervised rows and none on padding")
    if any(t in (2, 3) for t in active):
        source = batch.get("charted_source_positions")
        if source is None or tuple(source.shape) != (ids.shape[0], ids.shape[1], 3) or not bool(torch.isfinite(source).all()):
            raise D3Error("H3 rows need authentic packed [B, S, 3] charted source positions")


def bind_native_flow_batch(receipt: dict[str, Any], batch: dict[str, torch.Tensor], bands: dict[int, dict[str, Any]], *,
                           route: str, flow_param: str = "x0", signs: dict[int, float] | None = None
                           ) -> tuple[Callable[[torch.Tensor], torch.Tensor], dict[str, Any]]:
    """A Codex guided native-flow batch (`native_flow_cache.v1`) as a D3b `loss_fn(prediction)` on THIS
    engine flow head (`flow_param`, x0 by the 2026-08-28 proof).  The receipt supplies the teacher
    target parametrisation, the noisy-channel sub-bands (digest re-checked) and the channel mask; its
    own `prediction_kind` (the student assumption Codex sealed for) must be declared and is recorded, not used.

    Codex `ReadyObjectiveAdapter` (__post_init__ + bind) guards, all ported (1.1.1): the sealed / finite
    / structural / guided / input-target-separation flags and the cache / corpus / teacher / amendment
    sha pins; the caller `route` equal to the receipt route; `check_native_flow_batch`; every supervised
    type allocated; joint AV = both streams from one pinned source window; ACE generation = audited
    noisy channels [256, 384) -> emission [0, 128) with authentic code conditioning whose band / code
    digests re-hash.  A parent-raw head or target needs `signs` FROM THE CALLER (s8_media_smoke.SIGN,
    `flow_signs`), equal to the receipt signs on every type the receipt declares -- two independent
    sources must agree; the receipt alone never sets a sign."""

    if receipt.get("schema") != "chimera_recovery.native_flow_cache.v1" or any(
            receipt.get(k) is not True for k in ("sealed", "all_finite", "all_structural_ok", "guided_teacher", "no_clean_target_in_inputs")):
        raise D3Error("not a sealed guided native-flow receipt with reviewed input / target separation")
    for key in ("cache_sha256", "corpus_sha256", "teacher_sha256", "recovery_amendment_sha256"):
        if not is_sha256(receipt.get(key)):
            raise D3Error(f"native flow receipt {key} is not a sha256 pin")
    if receipt.get("input_layout") != "feature" or receipt.get("target_layout") != "emission":
        raise D3Error("native flow receipt must declare feature inputs and emission targets")
    target_kind = receipt.get("target_kind")
    if target_kind not in ("x0", "canonical_velocity", "parent_raw"):
        raise D3Error(f"teacher target parametrisation {target_kind!r} not declared")
    if receipt.get("prediction_kind") not in ("x0", "raw", "canonical_velocity"):
        raise D3Error("native flow receipt must declare the student head parametrisation it was sealed for")
    route = flow_route(route)
    sealed_route = receipt.get("route")
    if sealed_route != route:
        raise D3Error(f"sealed native cache route {sealed_route!r} differs from this binding route {route!r}")
    allocation = receipt.get("noisy_feature_bands")
    if (not isinstance(allocation, dict) or not allocation
            or any(not isinstance(k, str) or not k.isdecimal() or not isinstance(v, list) or len(v) != 2
                   or any(type(n) is not int for n in v) for k, v in allocation.items())
            or canonical_sha256(allocation) != receipt.get("noisy_feature_bands_sha256")):
        raise D3Error("noisy-channel sub-band allocation missing, malformed, or its digest moved")
    noisy = {int(k): (int(v[0]), int(v[1])) for k, v in allocation.items()}
    for t, (lo, hi) in noisy.items():
        if t not in bands:
            raise D3Error(f"type {t}: the receipt allocates a type this base has no interface band for")
        flo, fhi = bands[t]["feature"]
        elo, ehi = bands[t]["emission"]
        if not (0 <= flo <= lo < hi <= fhi) or hi - lo != ehi - elo:
            raise D3Error(f"type {t}: receipt noisy band [{lo}, {hi}) disagrees with this base interface bands")
    head_kind = {"x0": "x0", "raw": "raw", "canonical_velocity": "canonical_velocity"}.get(flow_param)
    if head_kind is None:
        raise D3Error(f"flow head parametrisation {flow_param!r}")
    check_native_flow_batch(route, batch)
    types, mask = batch["native_media_type_ids"], batch["native_media_loss_mask"]
    active = {int(t) for t in types[mask > 0].unique().tolist()}
    if not active <= set(noisy):
        raise D3Error("a supervised type has no noisy-channel allocation in the receipt")
    if target_kind == "parent_raw" or head_kind == "raw":
        receipt_signs = receipt.get("parent_signs")
        if not isinstance(receipt_signs, dict) or not receipt_signs:
            raise D3Error("a parent-raw head or target needs the sealed receipt parent_signs")
        declared = {int(k): float(v) for k, v in receipt_signs.items()}
        if signs is None:
            raise D3Error("a parent-raw head or target needs the caller audited signs (flow_signs()); the receipt alone never sets them")
        if not active <= set(declared) or any(signs.get(t) is None or float(signs[t]) != v for t, v in declared.items()):
            raise D3Error(f"parent signs {signs} differ from the sealed receipt signs {declared} (or miss a supervised type)")
        signs = {t: float(signs[t]) for t in declared}
    else:
        signs = None
    if route == "joint_av":
        alignment = receipt.get("av_alignment") or {}
        start = alignment.get("start_seconds")
        if (active != {2, 3} or alignment.get("same_origin") is not True or not is_sha256(alignment.get("source_media_sha256"))
                or isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start) or start < 0):
            raise D3Error("joint AV needs both supervised streams from one pinned source window (same_origin, source sha, start >= 0)")
    if route == "music_generate":
        x_t_all = batch["native_media_features"]
        if noisy.get(4) != (256, 384) or tuple(bands[4]["feature"]) != (0, 384) or tuple(bands[4]["emission"]) != (0, 128):
            raise D3Error("ACE generation needs the audited noisy channels [256, 384) -> emission [0, 128)")
        condition = receipt.get("conditioning_contract") or {}
        if (condition.get("ace_source_latents") != "authentic_code_detokenizer"
                or condition.get("ace_chunk_masks") != "ones_with_right_padding_zero"
                or condition.get("regime") != "teacher_audio_code_conditioned_generation"
                or condition.get("full_clean_latent_in_inputs") is not False):
            raise D3Error("ACE generation needs authentic code conditioning; clean-target reference / zero-source fixtures are refused")
        for key in ("source_band_sha256", "chunk_band_sha256", "codes_sha256", "conditioning_manifest_sha256", "ace_step15_sha256",
                    "model_base_sha256"):
            if not is_sha256(condition.get(key)):
                raise D3Error(f"ACE conditioning {key} is not a sha256 pin")
        ace_rows = types == 4
        if (codex_tensor_sha256(x_t_all[..., :128][ace_rows]) != condition["source_band_sha256"]
                or codex_tensor_sha256(x_t_all[..., 128:256][ace_rows]) != condition["chunk_band_sha256"]):
            raise D3Error("ACE native conditioning band digest moved")
        code_indices = batch.get("ace_code_indices")
        if (code_indices is None or code_indices.ndim != 2 or code_indices.shape[0] != types.shape[0]
                or code_indices.dtype != torch.int64):
            raise D3Error("ACE needs exact in-sequence native music-code conditioning indices")
        code_ids = batch["input_ids"].gather(1, code_indices) - OMNI1_MUSIC_BEGIN
        if (bool(((code_ids < 0) | (code_ids >= ACE_CODES)).any())
                or codex_tensor_sha256(code_ids.to(torch.int32).flatten()) != condition["codes_sha256"]):
            raise D3Error("ACE native conditioning code ids differ from the captured teacher codes")
    target = batch["native_media_targets"]
    x_t, sigma = batch["native_media_features"], batch["native_media_noise_levels"]
    channel_mask = batch.get("native_media_channel_loss_mask")

    def loss_fn(prediction: torch.Tensor) -> torch.Tensor:
        dev = prediction.device
        return flow_velocity_loss(prediction, target.to(dev), types.to(dev), mask.to(dev), bands, prediction_kind=head_kind,
                                  target_kind=target_kind, x_t=x_t.to(dev), sigma=sigma.to(dev), signs=signs,
                                  noisy_feature_bands=noisy, channel_mask=None if channel_mask is None else channel_mask.to(dev))

    return loss_fn, {"route": route, "target_kind": target_kind, "engine_prediction_kind": head_kind,
                     "receipt_prediction_kind": receipt.get("prediction_kind"), "noisy_feature_bands": {str(k): list(v) for k, v in noisy.items()},
                     "signs": None if signs is None else {str(k): v for k, v in signs.items()}, "identity": cache_identity(receipt)}


def d4_check_optimizer_union(groups: list[dict[str, Any]], *, families: tuple[str, ...] = D4_OPTIMIZER_FAMILIES) -> dict[str, Any]:
    """D4 optimiser plan: every declared family present, every physical parameter owned by exactly one
    group, every group's lr EXACTLY 0.3 x its producing stage's base lr (decimal arithmetic, the
    producer's rate read from its own ledger, never inferred), every census entry with a numel."""

    from decimal import Decimal, InvalidOperation

    owners: dict[str, tuple[str, int]] = {}
    seen: set[str] = set()
    for group in groups:
        family = group.get("family")
        if family not in families:
            raise D3Error(f"unknown D4 optimiser family {family!r}")
        seen.add(family)
        try:
            previous, current = Decimal(str(group["previous_base_lr"])), Decimal(str(group["lr"]))
        except (InvalidOperation, KeyError) as error:
            raise D3Error(f"{family}: lr / previous_base_lr missing or invalid") from error
        if not previous.is_finite() or previous <= 0 or not current.is_finite() or current != previous * Decimal(D4_LR_MULTIPLIER):
            raise D3Error(f"{family}: lr {current} != {D4_LR_MULTIPLIER} x producer lr {previous}")
        if not group.get("parameters"):
            raise D3Error(f"{family}: no parameters")
        for parameter in group["parameters"]:
            ident = parameter.get("parameter_id")
            if not ident or ident in owners:
                raise D3Error(f"parameter {ident!r} has two D4 optimiser owners (or no id)")
            if type(parameter.get("numel")) is not int or parameter["numel"] < 1:
                raise D3Error(f"parameter {ident!r}: census numel missing")
            owners[ident] = (family, parameter["numel"])
    if seen != set(families):
        raise D3Error(f"D4 optimiser plan misses families {sorted(set(families) - seen)}")
    return {"families": sorted(seen), "parameters": len(owners), "numel": sum(v[1] for v in owners.values()),
            "ownership_sha256": canonical_sha256({k: list(v) for k, v in owners.items()})}


def d4_check_base(base_lineage: list[str], *, bank_contains_d2: bool = True) -> None:
    """D4 composes the D2 r64 / K3 r32 + gain bank exactly once onto the PRE-D2 text weights: a base
    whose lineage already holds a D2 delta (e.g. OMNI1D2 or a later clone) would apply D2 twice."""

    composed = [entry for entry in base_lineage if canonical_stage(entry).split(".")[0] == "D2" or "OMNI1D2" in entry
                or "OMNI1D3" in entry]
    if bank_contains_d2 and composed:
        raise D3Error(f"D4 base already contains D2 ({composed}); the D2 bank would be applied twice -- use the pre-D2 clone")


def d4_select_replay(lane: str, rows: list[dict[str, Any]], *, quota: int, unit: str, excluded_groups: set[str] | frozenset[str] = frozenset(),
                     excluded_ids: set[str] | frozenset[str] = frozenset()) -> dict[str, Any]:
    """Deterministic D4 replay selection in the given row order: each distinct row at most once, never
    a held / excluded group or id (the BFCL-62 rows of A2 3a are excluded ids), until `quota` (tokens
    for unit "real_tokens", distinct pairs for unit "pairs") is met.  A lane short of distinct rows is
    used once each and its deficit reported -- no duplication (Codex: "do not manufacture 300 distinct
    pairs from 32 caches").  1.1.1 (Codex select_replay row rules): a selected row is an existing sealed
    target (`sealed_target` true) pinned by corpus / teacher-manifest / target-shard sha256; a native
    pair_id is the canonical digest of its input / target / conditioning sha256, and an input is never
    its own target."""

    if unit not in ("real_tokens", "pairs"):
        raise D3Error("replay unit is real_tokens or pairs")
    ids: set[str] = set()
    pairs: set[str] = set()
    selected: list[str] = []
    amount = 0
    for row in rows:
        if amount >= int(quota):
            break
        ident, group = row.get("id"), row.get("recovery_group_id")
        if not ident or not group:
            raise D3Error("replay row without id / recovery_group_id")
        if ident in ids:
            raise D3Error(f"replay row {ident!r} listed twice")
        ids.add(ident)
        if row.get("split") != "train" or row.get("held") is not False:
            raise D3Error(f"replay row {ident!r} is not a train, non-held row")
        if group in excluded_groups or ident in excluded_ids:
            continue
        if row.get("lane") != lane:
            raise D3Error(f"replay row {ident!r} belongs to lane {row.get('lane')!r}")
        if row.get("sealed_target") is not True:
            raise D3Error(f"replay row {ident!r} is not an existing sealed target")
        for key in ("corpus_sha256", "teacher_manifest_sha256", "target_shard_sha256"):
            if not is_sha256(row.get(key)):
                raise D3Error(f"replay row {ident!r}: source pin {key} missing")
        if unit == "pairs":
            pair = row.get("pair_id")
            if not pair:
                raise D3Error(f"replay row {ident!r} has no pair identity")
            binding = {key: row.get(key) for key in ("input_sha256", "target_sha256", "conditioning_sha256")}
            if any(not is_sha256(v) for v in binding.values()) or pair != canonical_sha256(binding):
                raise D3Error(f"replay row {ident!r}: pair_id must be the digest of its input / target / conditioning shas")
            if binding["input_sha256"] == binding["target_sha256"]:
                raise D3Error(f"replay row {ident!r}: the target cannot be the replay input")
            if pair in pairs:
                continue   # the same native pair twice adds nothing; it never fills the quota
            pairs.add(pair)
            count = 1
        else:
            count = row.get("real_tokens")
            if type(count) is not int or count < 1:
                raise D3Error(f"replay row {ident!r}: real token count missing")
        selected.append(ident)
        amount += count
    return {"lane": lane, "unit": unit, "quota": int(quota), "selected": selected, "amount": amount,
            "deficit": max(0, int(quota) - amount), "overshoot": max(0, amount - int(quota)), "duplicated": 0}


def d4_check_stage_chain(states: list[dict[str, Any]], *, initial_shared_bank_sha256: str,
                         optional: tuple[str, ...] = ()) -> dict[str, Any]:
    """The prior-stage chain D4 consolidates: D2 then the D3 stages in `stage_order(optional)` order,
    once each (so speech-in before speech-out, after vision), each state's parent the one before it;
    the shared r16 bank one lineage -- D3a leaves it unchanged, every later D3 stage replaces it
    (complete banks are never summed), D2's r64 bank is never claimed as the D3 r16 bank."""

    order = ("D2",) + stage_order(optional)
    roles = [canonical_stage(s.get("stage_role", "")) for s in states]
    if roles != list(order):
        raise D3Error(f"D4 prior stages {roles} != the preregistered order {list(order)}")
    if not is_sha256(initial_shared_bank_sha256):
        raise D3Error("initial shared r16 bank identity required")
    previous, bank, seen = None, initial_shared_bank_sha256, set()
    for state, role in zip(states, roles):
        sha = state.get("state_sha256")
        if not is_sha256(sha) or sha in seen:
            raise D3Error(f"{role}: state identity missing or repeated")
        if previous is not None and state.get("parent_state_sha256") != previous:
            raise D3Error(f"{role}: parent state is not the previous stage's")
        if state.get("shared_lora_semantics") == "add":
            raise D3Error(f"{role}: complete shared LoRA banks cannot be summed")
        if role.startswith("D3"):
            expected = "unchanged" if role == "D3a" else "full_replacement"
            if (state.get("shared_lora_semantics") != expected or not is_sha256(state.get("shared_lora_state_sha256"))
                    or state.get("shared_lora_parent_sha256") != bank):
                raise D3Error(f"{role}: shared r16 continuation not proven")
            if expected == "unchanged" and state["shared_lora_state_sha256"] != bank:
                raise D3Error("D3a (unit refit only) cannot replace the shared r16 bank")
            bank = state["shared_lora_state_sha256"]
        elif state.get("shared_lora_state_sha256") is not None:
            raise D3Error("the D2 r64 bank cannot be claimed as the D3 shared r16 bank")
        previous = sha
        seen.add(sha)
    return {"terminal_state_sha256": previous, "shared_r16_state_sha256": bank, "stages": len(states),
            "lineage_sha256": canonical_sha256(states)}


# --------------------------------------------------------------------------
# Parity arm on OMNI1 (CPU fp32) against the Chimera reference.
# --------------------------------------------------------------------------


def trim_segment(segment: dict[str, Any], max_rows_per_type: int, placeholder: int) -> dict[str, Any]:
    """Keep the first `max_rows_per_type` media rows of every type (order kept; the HY3D time row
    is row 0 of the volume segment and so is kept).  0 = keep all."""

    if max_rows_per_type <= 0:
        return segment
    types = segment["types"][0]
    keep: list[int] = []
    for t in dict.fromkeys(int(v) for v in types.tolist()):
        keep.extend(torch.nonzero(types == t).flatten()[:max_rows_per_type].tolist())
    keep = sorted(keep)
    idx = torch.tensor(keep, dtype=torch.long)
    plen = int(segment["prompt_tokens"])
    m = len(keep)
    out = dict(segment)
    out["ids"] = torch.cat([segment["ids"][:, :plen], torch.full((1, m), int(placeholder), dtype=torch.long)], dim=1)
    out["features"] = segment["features"][:, idx]
    out["types"] = segment["types"][:, idx]
    out["positions"] = segment["positions"][:, idx]
    out["sigma"] = None if segment["sigma"] is None else segment["sigma"][:, idx]
    out["indices"] = torch.arange(plen, plen + m, dtype=torch.long).view(1, m)
    out["media_rows"] = m
    out["trimmed_from"] = int(segment["media_rows"])
    if "host_rows_s5" in segment:
        out["host_rows_s5"] = segment["host_rows_s5"][idx]
    return out


def with_text_suffix(segment: dict[str, Any], count: int) -> dict[str, Any]:
    """[prompt | media rows] -> [prompt | media rows | `count` answer-text tokens].  The answer text is
    the prompt's own token ids, cycled (real text rows, nothing past `prompt_tokens` is media); media
    rows, indices and features are unchanged; per-position source positions get zero rows (the H3
    zero plan of `h3_zero_positions`)."""

    if count <= 0:
        return segment
    prompt = int(segment["prompt_tokens"])
    if prompt <= 0:
        raise D3Error("a text suffix needs at least one prompt token to cycle")
    ids = segment["ids"]
    tail = ids[:, :prompt].repeat(1, -(-count // prompt))[:, :count]
    out = dict(segment)
    out["ids"] = torch.cat([ids, tail], dim=1)
    out["suffix_tokens"] = int(segment.get("suffix_tokens", 0)) + int(count)
    positions = segment.get("source_positions")
    if positions is not None:
        out["source_positions"] = {k: torch.cat([v, torch.zeros(v.shape[0], count, *v.shape[2:], dtype=v.dtype)], dim=1)
                                   for k, v in positions.items()}
    return out


def h3_zero_positions(cfg: Any, length: int) -> dict[str, torch.Tensor] | None:
    """s7_omni1_check's plan: H3 source positions zeros (the h3_thw_v1 positions have never run)."""

    out = {spec.source: torch.zeros(1, length, 3) for spec in cfg.charted_blocks if spec.modulation == "h3_curve6"}
    return out or None


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def command_parity(args: argparse.Namespace) -> int:
    torch.set_num_threads(int(args.threads))
    started = time.time()
    s7 = composite_tool("s7_omni1_check")
    dtype = torch.float32
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "kind": "parity", "tool": tool_identity(), "argv": sys.argv[1:],
                               "torch": torch.__version__, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "s7_omni1_check": peer_record(s7), "device": "cpu", "dtype": "float32"}
    refusals: list[str] = []
    omni_dir = Path(args.omni1)
    log("opening the hybrid (P1 reader, streamed qwen3_5 layers, Chimera units from extras)")
    hybrid = HybridD3(omni_dir, Path(args.parent_config), HybridOptions(dtype=dtype, device="cpu", kernels="torch",
                                                                         mode="streamed", prefetch=True))
    receipt["hybrid"] = hybrid.identity()
    receipt["layout"] = [{"block": p.block, "container": p.container, "after_text_layer": p.after_text_layer,
                          "depth_d": p.depth, "type_ids": list(p.type_ids)} for p in hybrid.layout]
    log("opening the Chimera reference (s7_omni1_check.Checkpoint, sha-verified)")
    ref = s7.Checkpoint(omni_dir, "omni1", verify=True)
    top = s7.build_top(ref, device="cpu")
    meta = top["meta"]
    bridge32 = top["native_media_bridge"].to(dtype=dtype)
    setattr(meta, "native_media_bridge", bridge32)
    head32 = top["native_media_head"].to(dtype=dtype)
    norm32 = top["norm"].to(dtype=dtype)

    # G-a: the hybrid's media modules are the reference's, bit for bit
    identity: dict[str, Any] = {}
    for name, a, b in [("native_media_bridge", hybrid.bridge, bridge32), ("native_media_head", hybrid.media_head, head32)]:
        sa, sb = a.state_dict(), b.state_dict()
        same = set(sa) == set(sb) and all(torch.equal(sa[k].float(), sb[k].float()) for k in sa)
        identity[name] = bool(same)
        if not same:
            refusals.append(f"G-a: hybrid {name} differs from the reference's")

    bands_feature = {t: v["feature"] for t, v in hybrid.bands.items()}
    segments = []
    for p in hybrid.layout:
        seg = s7.media_segment(p.container, Path(args.tap_rows), bands_feature, ref.cfg)
        seg = trim_segment(seg, int(args.max_rows_per_type), hybrid.media_placeholder)
        seg["source_positions"] = h3_zero_positions(ref.cfg, int(seg["ids"].shape[1]))
        segments.append(seg)
    receipt["segments"] = [{"unit": s["unit"], "forward": s["forward"], "prompt_tokens": s["prompt_tokens"],
                            "media_rows": s["media_rows"], "trimmed_from": s.get("trimmed_from"), "tokens": int(s["ids"].shape[1]),
                            "record": s["record"]} for s in segments]
    log(f"{len(segments)} segments: " + ", ".join(f"{s['unit']} {int(s['ids'].shape[1])} tok" for s in segments))

    # reference inputs (s7_omni1_check's own recipe, fp32)
    ref_state = []
    hyb_state = []
    with torch.no_grad():
        for seg in segments:
            ids = seg["ids"]
            x = top["embed"](ids).to(dtype)
            types = seg["types"]
            valid_media = torch.ones(types.shape, dtype=torch.bool)
            noise = None if seg["sigma"] is None else seg["sigma"]
            x, count = type(meta)._add_native_media(meta, x, seg["features"], types, seg["positions"], seg["indices"], valid_media, noise)
            row_types, levels = type(meta)._charted_layout(meta, ids, count, seg["indices"], types, valid_media, noise)
            valid = torch.ones(ids.shape, dtype=torch.bool)
            ref_state.append({"x": x, "valid": valid, "row_types": row_types, "levels": levels, "positions": seg["source_positions"]})
            ctx = hybrid.prepare(seg)
            hyb_state.append(ctx)
            identity.setdefault("input_rel_l2", []).append(rel_l2(ctx["x"], x))
    receipt["identity"] = identity

    text_blocks = list(hybrid.reader.text_blocks)
    per_layer: list[dict[str, Any]] = []
    units_report: dict[str, Any] = {}
    x_d_rel: dict[str, float] = {}
    contexts_cache: dict[int, Any] = {}

    def run_ref_block(block: Any, s: dict[str, Any]) -> torch.Tensor:
        y = block(s["x"], native_media_slot_mask=None, valid_media_mask=None, enable_native_media_refinement=False,
                  token_valid_mask=s["valid"], charted_row_types=s["row_types"], charted_noise_levels=s["levels"],
                  charted_source_positions=s["positions"])
        return torch.where(s["valid"].unsqueeze(-1), y, torch.zeros_like(y))

    with torch.no_grad():
        for t, j in enumerate(text_blocks):
            blk = s7.build_block(ref, j, device="cpu", dtype=dtype)
            for s in ref_state:
                s["x"] = run_ref_block(blk, s)
            del blk
            layer = hybrid.engine.layer(t, t + 1 if t + 1 < len(text_blocks) else None)
            for ctx in hyb_state:
                contexts = hybrid.engine._context(1, int(ctx["x"].shape[1]))
                ctx["x"] = hybrid.engine._run(layer, t, ctx["x"], contexts)
            row = {"layer": t, "block": j, "rel_l2": [rel_l2(c["x"], s["x"]) for c, s in zip(hyb_state, ref_state)],
                   "finite": all(bool(torch.isfinite(c["x"]).all()) for c in hyb_state)}
            per_layer.append(row)
            if not row["finite"]:
                refusals.append(f"layer {t}: hybrid hidden not finite")
            following = hybrid.units_after(t)
            if following:
                for si, (seg, c, s) in enumerate(zip(segments, hyb_state, ref_state)):
                    x_d_rel[seg["unit"]] = rel_l2(c["x"], s["x"])
            for placement in following:
                cblk = s7.build_block(ref, placement.block, device="cpu", dtype=dtype)
                unit = hybrid.units[placement.container]
                sa, sb = unit.state_dict(), cblk.charted.state_dict()
                same_params = set(sa) == set(sb) and all(torch.equal(sa[k].float(), sb[k].float()) for k in sa)
                rep: dict[str, Any] = {"block": placement.block, "params_bit_identical": bool(same_params), "segments": {}}
                if not same_params:
                    refusals.append(f"G-a: {placement.container} unit params differ from the reference block")
                for seg, c, s in zip(segments, hyb_state, ref_state):
                    w_ref = cblk.charted_output(s["x"], s["row_types"], s["levels"], s["valid"], s["positions"])
                    # G-b: the hybrid unit on the REFERENCE's own input
                    w_iso = hybrid.unit_write(placement, s["x"], {**c, "row_types": s["row_types"], "levels": s["levels"],
                                                                  "valid": s["valid"], "positions": s["positions"]})
                    # the hybrid's own write on its own stream
                    w_hyb = hybrid.unit_write(placement, c["x"], c)
                    owned = unit.owned_rows(c["row_types"], c["valid"])
                    n_owned = 0 if owned is None else int(owned.sum())
                    off = w_hyb if owned is None else w_hyb.masked_fill(owned.unsqueeze(-1), 0.0)
                    own_norm = 0.0 if owned is None else float(w_hyb[owned].norm())
                    own_segment = seg["unit"] == placement.container
                    r = {"rows_owned": n_owned, "write_iso_bit_identical": bool(torch.equal(w_iso, w_ref)),
                         "write_iso_max_abs": float((w_iso - w_ref).abs().max()) if w_ref.numel() else 0.0,
                         "zeros_off_owned": bool((off == 0).all()), "write_finite": bool(torch.isfinite(w_hyb).all()),
                         "write_norm_owned": own_norm, "own_segment": own_segment,
                         "write_rel_l2_vs_reference": rel_l2(w_hyb, w_ref) if float(w_ref.norm()) > 0 else (0.0 if float(w_hyb.norm()) == 0 else math.inf)}
                    if not r["write_iso_bit_identical"]:
                        refusals.append(f"G-b: {placement.container} write on {seg['unit']} differs from the reference on the same input")
                    if not r["zeros_off_owned"]:
                        refusals.append(f"G-c: {placement.container} writes off its own rows on {seg['unit']}")
                    if not r["write_finite"]:
                        refusals.append(f"G-c: {placement.container} non-finite write on {seg['unit']}")
                    if own_segment and (n_owned == 0 or own_norm == 0.0):
                        refusals.append(f"G-c: {placement.container} owns / writes nothing on its own segment")
                    if not own_segment and own_norm != 0.0:
                        refusals.append(f"G-c: {placement.container} writes on the {seg['unit']} segment")
                    rep["segments"][seg["unit"]] = r
                    c["x"] = c["x"] + w_hyb
                    s["x"] = torch.where(s["valid"].unsqueeze(-1), s["x"] + w_ref, torch.zeros_like(s["x"]))
                units_report[placement.container] = rep
                del cblk
            if t % 8 == 0 or t == len(text_blocks) - 1:
                worst = max(row["rel_l2"]) if row["rel_l2"] else 0.0
                log(f"text layer {t}/{len(text_blocks) - 1} (block {j}): worst rel_l2 {worst:.3e} "
                    f"({time.time() - started:.0f}s, rss {hybrid.T.host_rss_gb() or 0:.1f} GB)")

        # heads
        head_report: dict[str, Any] = {}
        lm_head = top["lm_head_weight"]
        base = hybrid.base_vocab
        for seg, c, s in zip(segments, hyb_state, ref_state):
            n_h = hybrid.final_norm(c["x"])
            n_r = norm32(s["x"])
            em_h = hybrid.emission(n_h, c)
            em_r = head32(n_r.gather(1, seg["indices"].unsqueeze(-1).expand(-1, -1, n_r.shape[-1])), seg["types"])
            band_rel = {}
            for t in sorted(set(int(v) for v in seg["types"][0].tolist())):
                lo, hi = hybrid.bands[t]["emission"]
                rows = seg["types"][0] == t
                band_rel[str(t)] = rel_l2(em_h[0, rows, lo:hi], em_r[0, rows, lo:hi])
            positions = torch.arange(int(seg["prompt_tokens"]))
            lg_h = hybrid.text_logits(n_h, positions)
            lg_r = torch.cat([F.linear(n_r[0, positions], lm_head[a:min(a + 32768, base)].to(dtype)) for a in range(0, base, 32768)], dim=-1)
            agree = int((lg_h.argmax(-1) == lg_r.argmax(-1)).sum())
            head_report[seg["unit"]] = {
                "final_hidden_rel_l2": rel_l2(n_h, n_r), "emission_band_rel_l2": band_rel,
                "text_logits_rel_l2": rel_l2(lg_h, lg_r), "text_top1_agree": agree, "text_positions": int(positions.numel()),
                "finite": bool(torch.isfinite(n_h).all() and torch.isfinite(em_h).all() and torch.isfinite(lg_h).all()),
            }
    tol = float(args.tol_rel)
    e2e: dict[str, Any] = {"tol_rel": tol, "x_d_rel_l2": x_d_rel, "heads": head_report}
    for unit, value in x_d_rel.items():
        if not value <= tol:
            refusals.append(f"G-d: x_d rel_l2 {value:.3e} > {tol} on {unit}")
    for unit, rep in head_report.items():
        if not rep["finite"]:
            refusals.append(f"G-d: non-finite hybrid output on {unit}")
        for key in ("final_hidden_rel_l2", "text_logits_rel_l2"):
            if not rep[key] <= tol:
                refusals.append(f"G-d: {key} {rep[key]:.3e} > {tol} on {unit}")
        for t, value in rep["emission_band_rel_l2"].items():
            if not value <= tol:
                refusals.append(f"G-d: emission band type {t} rel_l2 {value:.3e} > {tol} on {unit}")
    receipt.update({"per_layer": per_layer, "units": units_report, "end_to_end": e2e, "refusals": refusals,
                    "verdict": "PASS" if not refusals else "FAIL", "seconds": round(time.time() - started, 1),
                    "peak_rss_gb": hybrid.T.host_rss_gb()})
    sha = hybrid.T.write_json_atomic(Path(args.receipt), receipt)
    for line in refusals[:20]:
        log(f"REFUSAL {line}")
    worst_layer = max((max(r["rel_l2"]) for r in per_layer), default=0.0)
    log(f"D3 PARITY {'PASS' if not refusals else 'FAIL'}: {len(hybrid.layout)} units, {len(segments)} segments, "
        f"worst per-layer rel_l2 {worst_layer:.3e}, x_d {max(x_d_rel.values(), default=0.0):.3e}, "
        f"heads {max((max([r['final_hidden_rel_l2'], r['text_logits_rel_l2'], *r['emission_band_rel_l2'].values()]) for r in head_report.values()), default=0.0):.3e} "
        f"(tol {tol}); receipt {args.receipt} sha256 {sha}")
    return 0 if not refusals else 3


# --------------------------------------------------------------------------
# The synthetic arm: a tiny OMNI-shaped child, built with the real importer.
# --------------------------------------------------------------------------


TINY = SimpleNamespace(LAYERS=4, HIDDEN=64, WIDTH=24, TRUNK=8, K3=16, VOCAB=40, HEADS=2, KV_HEADS=1, HEAD_DIM=32,
                       V_HEADS=4, K_HEADS=2, K_DIM=16, V_DIM=16, FEATURE=48,
                       BANDS=[("control", 8), ("media", 8), ("music_codes", 16)])


def tiny_hf_config() -> dict[str, Any]:
    t = TINY
    return {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "tie_word_embeddings": False,
            "text_config": {
                "model_type": "qwen3_5_text", "hidden_size": t.HIDDEN, "intermediate_size": t.WIDTH, "num_hidden_layers": t.LAYERS,
                "num_attention_heads": t.HEADS, "num_key_value_heads": t.KV_HEADS, "head_dim": t.HEAD_DIM,
                "linear_num_value_heads": t.V_HEADS, "linear_num_key_heads": t.K_HEADS, "linear_key_head_dim": t.K_DIM,
                "linear_value_head_dim": t.V_DIM, "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
                "layer_types": ["linear_attention"] * 3 + ["full_attention"], "vocab_size": t.VOCAB, "rms_norm_eps": 1e-6,
                "max_position_embeddings": 256, "hidden_act": "silu", "attn_output_gate": True, "attention_bias": False,
                "attention_dropout": 0.0, "tie_word_embeddings": False,
                "rope_parameters": {"mrope_interleaved": True, "mrope_section": [2, 1, 1], "partial_rotary_factor": 0.25,
                                    "rope_theta": 10000.0, "rope_type": "default"},
                "dtype": "float32"}}


def tiny_unit_specs(insert_after: int, *, with_qvl: bool = False) -> list[Any]:
    """An ACE-like (adaln6, RoPE, window), an H3-like (h3_curve6, reads text) and an HY3D-like
    (no modulation, layer norm, gelu, attention bias) unit, all after text layer `insert_after`;
    `with_qvl` adds a QVL-like vision unit (type 6, no modulation) for the D3d feature-KD tests."""

    from chimera_omni.charted import ChartedBlockSpec

    first = insert_after + 1
    return [
        ChartedBlockSpec(layer_index=first, source="ace", component="tiny.ace", type_ids=(4,), width=32, heads=2, kv_heads=1,
                         head_dim=16, mlp_width=40, mlp_kind="swiglu", modulation="adaln6", rope_theta=10000.0,
                         sliding_window=6, time_hidden=24, time_freq_dim=16, time_embed_r=True),
        ChartedBlockSpec(layer_index=first + 1, source="h3", component="tiny.h3", type_ids=(2, 3), width=48, heads=3, kv_heads=3,
                         head_dim=16, mlp_width=56, mlp_kind="swiglu", modulation="h3_curve6", norm_eps=1e-5,
                         h3_curve_grid=9, h3_curve_dim=4, h3_rope_freqs=2),
        ChartedBlockSpec(layer_index=first + 2, source="hy3d", component="tiny.hy3d", type_ids=(7,), width=32, heads=2, kv_heads=2,
                         head_dim=16, mlp_width=36, mlp_kind="gelu", modulation="none", norm_kind="layer", attn_bias=True),
    ] + ([ChartedBlockSpec(layer_index=first + 3, source="qvl32", component="tiny.qvl", type_ids=(6,), width=32, heads=2, kv_heads=2,
                           head_dim=16, mlp_width=36, mlp_kind="gelu", modulation="none", norm_kind="layer", attn_bias=True)]
         if with_qvl else [])


TINY_BANDS = {4: ((0, 12), (0, 6)), 2: ((12, 20), (6, 10)), 3: ((20, 24), (10, 12)), 7: ((24, 32), (12, 16)), 6: ((32, 48), (16, 17))}


def build_tiny_child(directory: Path, *, seed: int = 5, insert_after: int = 1, with_qvl: bool = False) -> dict[str, Any]:
    """A tiny Chimera child shaped like OMNI1: text layers imported from a random qwen3_5 state by
    the REAL importer, three charted units inserted after text layer `insert_after`, real bridge /
    head tensors (randomised so the interface is not the identity), a composite manifest with
    placements, the trunk map and interface bands.  Returns paths + the reference model."""

    from safetensors.torch import save_file
    from chimera_omni.importing.profiles import qwen35_chimera_config
    from chimera_omni.importing.qwen35 import import_qwen35_state
    from chimera_omni.modeling import ChimeraOmniModel

    t = TINY
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape: int, scale: float = 0.08) -> torch.Tensor:
        return torch.randn(*shape, generator=gen) * scale

    key_dim, value_dim = t.K_HEADS * t.K_DIM, t.V_HEADS * t.V_DIM
    p = "model.language_model."
    qwen = {f"{p}embed_tokens.weight": rand(t.VOCAB, t.HIDDEN, scale=0.5), "lm_head.weight": rand(t.VOCAB, t.HIDDEN),
            f"{p}norm.weight": rand(t.HIDDEN, scale=0.1)}
    for i in range(t.LAYERS):
        q = f"{p}layers.{i}."
        qwen[f"{q}input_layernorm.weight"] = rand(t.HIDDEN, scale=0.1)
        qwen[f"{q}post_attention_layernorm.weight"] = rand(t.HIDDEN, scale=0.1)
        qwen[f"{q}mlp.gate_proj.weight"] = rand(t.WIDTH, t.HIDDEN)
        qwen[f"{q}mlp.up_proj.weight"] = rand(t.WIDTH, t.HIDDEN)
        qwen[f"{q}mlp.down_proj.weight"] = rand(t.HIDDEN, t.WIDTH)
        if i % 4 != 3:
            a = f"{q}linear_attn."
            qwen[f"{a}in_proj_qkv.weight"] = rand(2 * key_dim + value_dim, t.HIDDEN)
            qwen[f"{a}conv1d.weight"] = rand(2 * key_dim + value_dim, 1, 4)
            qwen[f"{a}in_proj_z.weight"] = rand(value_dim, t.HIDDEN)
            qwen[f"{a}in_proj_b.weight"] = rand(t.V_HEADS, t.HIDDEN)
            qwen[f"{a}in_proj_a.weight"] = rand(t.V_HEADS, t.HIDDEN)
            qwen[f"{a}dt_bias"] = rand(t.V_HEADS, scale=1.0)
            qwen[f"{a}A_log"] = rand(t.V_HEADS, scale=1.0)
            qwen[f"{a}norm.weight"] = 1.0 + rand(t.V_DIM, scale=0.1)
            qwen[f"{a}out_proj.weight"] = rand(t.HIDDEN, value_dim)
        else:
            s = f"{q}self_attn."
            qwen[f"{s}q_proj.weight"] = rand(t.HEADS * 2 * t.HEAD_DIM, t.HIDDEN)
            qwen[f"{s}k_proj.weight"] = rand(t.KV_HEADS * t.HEAD_DIM, t.HIDDEN)
            qwen[f"{s}v_proj.weight"] = rand(t.KV_HEADS * t.HEAD_DIM, t.HIDDEN)
            qwen[f"{s}o_proj.weight"] = rand(t.HIDDEN, t.HEADS * t.HEAD_DIM)
            qwen[f"{s}q_norm.weight"] = rand(t.HEAD_DIM, scale=0.1)
            qwen[f"{s}k_norm.weight"] = rand(t.HEAD_DIM, scale=0.1)
    base_cfg = qwen35_chimera_config(tiny_hf_config(), ledger_bands=t.BANDS, max_sequence_length=256, native_feature_dim=t.FEATURE,
                                     metadata={"c10_trunk_intermediate_size": str(t.TRUNK), "c10_k3_channels": str(t.K3)})
    imported, _receipt = import_qwen35_state(qwen, base_cfg, dtype=torch.float32)
    text_state = {n: v.detach().clone() for n, v in imported.state_dict().items()}
    specs = tiny_unit_specs(insert_after, with_qvl=with_qvl)
    total = t.LAYERS + len(specs)
    charted_at = [s.layer_index for s in specs]
    text_of = [b for b in range(total) if b not in charted_at]
    pattern = list(base_cfg.block_pattern)
    text_kinds = iter(pattern[i % len(pattern)] for i in range(t.LAYERS))
    full_pattern = tuple("charted" if b in charted_at else next(text_kinds) for b in range(total))
    cfg = dataclasses.replace(base_cfg, num_layers=total, block_pattern=full_pattern, charted_blocks=tuple(specs))
    cfg.validate()
    torch.manual_seed(seed + 1)
    model = ChimeraOmniModel(cfg).to(torch.float32)
    state = model.state_dict()
    for name, value in text_state.items():
        if name.startswith("blocks."):
            _, index, rest = name.split(".", 2)
            state[f"blocks.{text_of[int(index)]}.{rest}"] = value
        elif name in state:
            state[name] = value
    # non-trivial interface: random type rows / affine rows (the constructor zeroes some)
    for name in list(state):
        if name.startswith(("native_media_bridge.", "native_media_head.")) or ".charted." in name:
            if "curve_table" in name:
                state[name] = torch.linspace(-1.0, 1.0, state[name].numel()).view_as(state[name])
            elif "rope_inv_freq" in name:
                state[name] = 1.0 / (100.0 ** (torch.arange(state[name].numel(), dtype=torch.float32) / state[name].numel()))
            elif name.endswith("norm.weight") and ".charted." not in name:
                state[name] = 1.0 + rand(*state[name].shape, scale=0.1)
            else:
                state[name] = rand(*state[name].shape, scale=0.1)
    # band rows are written (C18-like) in embed and head: the importer leaves them zero, and a zero
    # band head makes the band logits independent of the hidden state (no trunk gradient in D3c)
    for name in ("embed.weight", "lm_head.weight"):
        rows = state[name][t.VOCAB:]
        state[name][t.VOCAB:] = rand(*rows.shape, scale=0.3)
    model.load_state_dict(state, strict=True)
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(state)
    halves = {"model-00001-of-00002.safetensors": names[: len(names) // 2], "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    weight_map = {}
    for filename, chunk in halves.items():
        save_file({n: state[n].contiguous() for n in chunk}, str(directory / filename),
                  metadata={"format": "pt", "schema": "chimera_omni.checkpoint.v1"})
        weight_map.update({n: filename for n in chunk})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps({"schema": "chimera_omni.checkpoint.v1", "chimera_commit": "tiny-d3",
                                                       "config": cfg.to_manifest()}), encoding="utf-8")
    manifest = {
        "schema": "chimera_omni.composite_manifest.v1",
        "blocks": {"trunk_layer_map": {str(i): b for i, b in enumerate(text_of)},
                   "placements": [{"layer_index": s.layer_index, "container": s.component.replace(".", "_"), "source": s.source,
                                   "type_ids": list(s.type_ids), "pairing": {"insert_after": insert_after}} for s in specs]},
        "interface": {"entries": [{"type_id": k, "feature_band": list(v[0]), "emission_band": list(v[1]), "name": f"tiny{k}", "set": "c15"}
                                  for k, v in TINY_BANDS.items()]},
    }
    (directory / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    parent = directory.parent / f"{directory.name}-parent-config"
    parent.mkdir(exist_ok=True)
    (parent / "config.json").write_text(json.dumps(tiny_hf_config()), encoding="utf-8")
    return {"dir": directory, "parent_config": parent, "cfg": cfg, "model": model.eval(), "text_of": text_of, "specs": specs}


def h3_sigmas(sigma_v: float) -> tuple[float, float]:
    """The H3 plain AV schedule ChartedSourceBlock checks: video shift 12, audio shift 3."""

    base = sigma_v / (12.0 + sigma_v * (1.0 - 12.0))
    return float(sigma_v), float(3.0 * base / (1.0 + 2.0 * base))


def tiny_segment(cfg: Any, kind: str, *, prompt: int = 5, rows: int = 6, seed: int = 0, suffix: int = 0) -> dict[str, Any]:
    """[prompt | media rows | suffix text] of one kind ("ace", "h3", "hy3d", "ocr", "text"),
    features in the type's feature band, positions, sigma, H3 zero source positions."""

    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, TINY.VOCAB, (1, prompt), generator=g)
    if kind == "text":
        return {"ids": ids, "prompt_tokens": prompt}
    placeholder = int(cfg.ledger().range("media").begin)
    if kind == "h3":
        types = torch.tensor([[3, 3] + [2] * (rows - 2)])
        sv, sa = h3_sigmas(0.6)
        sigma = torch.where(types == 2, torch.tensor(sv), torch.tensor(sa)).float()
    else:
        t = {"ace": 4, "hy3d": 7, "ocr": 6}[kind]
        types = torch.full((1, rows), t)
        sigma = None if kind == "ocr" else torch.full((1, rows), 0.4)
    feats = torch.zeros(1, rows, TINY.FEATURE)
    for r in range(rows):
        lo, hi = TINY_BANDS[int(types[0, r])][0]
        feats[0, r, lo:hi] = torch.randn(hi - lo, generator=g)
    m = rows
    tail = torch.randint(0, TINY.VOCAB, (1, suffix), generator=g)
    seg = {"ids": torch.cat([ids, torch.full((1, m), placeholder), tail], dim=1), "features": feats, "types": types,
           "positions": torch.rand(1, m, 4, generator=g), "sigma": sigma, "indices": torch.arange(prompt, prompt + m).view(1, m),
           "prompt_tokens": prompt, "media_rows": m}
    seg["source_positions"] = h3_zero_positions(cfg, prompt + m + suffix)
    return seg


#: the tiny teachers' raw-output signs (x0 = x_t + SIGN sigma raw), stand-ins for s8_media_smoke.SIGN
TINY_SIGNS = {4: -1.0, 2: 1.0, 3: 1.0, 7: 1.0}


def tiny_flow_batch(cfg: Any, kind: str, *, prompt: int = 5, rows: int = 6, seed: int = 0, suffix: int = 0) -> dict[str, Any]:
    """A GENUINE flow fixture (1.0.2's random features had no x_t / sigma / x0 relation -- Codex's
    "D3b fixture timing" finding): per row a clean x0 and noise eps, x_t = (1 - sigma) x0 + sigma eps
    written to the type's audited noisy sub-band (`TINY_NOISY_FEATURE_BANDS`) of its feature band
    (other feature columns keep tiny_segment's conditioning values), and the teacher output in its
    parent-raw form raw = -SIGN (eps - x0) on the emission band.  Returns the segment plus "x0",
    "raw", "eps" (emission layout [1, M, F]) and the signs."""

    seg = tiny_segment(cfg, kind, prompt=prompt, rows=rows, seed=seed, suffix=suffix)
    if seg.get("sigma") is None:
        raise D3Error(f"{kind}: a flow fixture needs a generated (noised) type")
    g = torch.Generator().manual_seed(seed + 7)
    m = int(seg["media_rows"])
    x0 = torch.zeros(1, m, TINY.FEATURE)
    eps = torch.zeros(1, m, TINY.FEATURE)
    raw = torch.zeros(1, m, TINY.FEATURE)
    features = seg["features"].clone()
    for r in range(m):
        t = int(seg["types"][0, r])
        elo, ehi = TINY_BANDS[t][1]
        flo, fhi = TINY_NOISY_FEATURE_BANDS[t]
        s = float(seg["sigma"][0, r])
        clean = torch.randn(ehi - elo, generator=g)
        noise = torch.randn(ehi - elo, generator=g)
        x0[0, r, elo:ehi], eps[0, r, elo:ehi] = clean, noise
        features[0, r, flo:fhi] = (1.0 - s) * clean + s * noise
        raw[0, r, elo:ehi] = -TINY_SIGNS[t] * (noise - clean)
    return {**seg, "features": features, "x0": x0, "eps": eps, "raw": raw, "signs": dict(TINY_SIGNS)}


def chimera_reference(model: Any, segment: dict[str, Any]) -> dict[str, torch.Tensor]:
    """`ChimeraOmniModel.forward` itself (whole tiny model, fp32)."""

    kwargs: dict[str, Any] = {}
    if segment.get("features") is not None:
        kwargs = {"native_media_features": segment["features"], "native_media_type_ids": segment["types"],
                  "native_media_positions": segment["positions"], "native_media_indices": segment["indices"],
                  "native_media_valid_mask": torch.ones(segment["types"].shape, dtype=torch.bool),
                  "native_media_noise_levels": segment["sigma"], "charted_source_positions": segment.get("source_positions"),
                  "return_aux": True}
        if segment["sigma"] is not None:
            # the tree refuses noise levels under "reconstruction"; "x0" changes nothing else here
            # (native_bidirectional_refinement is off in the config, so no refinement layout)
            kwargs.update({"native_media_target_kind": "x0",
                           "native_media_update_mask": torch.ones(segment["types"].shape, dtype=torch.bool)})
    with torch.no_grad():
        out = model(segment["ids"], **kwargs)
    return {"hidden": out["hidden"], "logits": out["logits"], "emission": out["aux"].get("native_media_prediction") if kwargs else None}


def command_synthetic(args: argparse.Namespace) -> int:
    """Tiny OMNI-shaped child: hybrid == Chimera forward; streamed grads == autograd; D3a-d train."""

    import tempfile

    torch.manual_seed(0)
    started = time.time()
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "kind": "synthetic", "tool": tool_identity(), "argv": sys.argv[1:],
                               "torch": torch.__version__, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    refusals: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        built = build_tiny_child(Path(tmp) / "child")
        cfg, ref_model = built["cfg"], built["model"]
        T = trainer_module()
        lora = T.SitePlanConfig(rank=4, alpha=4.0, k3_rank=0, trunk_ffn_width=TINY.TRUNK, gain=False)
        hybrid = HybridD3(built["dir"], built["parent_config"], HybridOptions(dtype=torch.float32, device="cpu", kernels="torch",
                                                                              lora=lora, lora_seed=3, pin_boundaries=False))
        receipt["layout"] = [{"block": p.block, "container": p.container, "after_text_layer": p.after_text_layer} for p in hybrid.layout]
        # 1. forward parity vs ChimeraOmniModel.forward
        parity = {}
        for kind in ("text", "ace", "h3", "hy3d"):
            seg = tiny_segment(cfg, kind, seed=11)
            ref = chimera_reference(ref_model, seg)
            out = hybrid.forward(seg)
            r = {"hidden_rel_l2": rel_l2(out["normed"], ref["hidden"])}
            positions = torch.arange(int(seg["prompt_tokens"]))
            lg = hybrid.text_logits(out["normed"], positions)
            r["text_logits_rel_l2"] = rel_l2(lg, ref["logits"][0, positions, : hybrid.base_vocab])
            if ref["emission"] is not None:
                r["emission_rel_l2"] = rel_l2(out["emission"], ref["emission"])
            parity[kind] = r
            worst = max(r.values())
            if not worst <= 1e-4:
                refusals.append(f"tiny parity {kind}: rel_l2 {worst:.3e} > 1e-4")
        receipt["tiny_parity"] = parity
        log("tiny parity " + ", ".join(f"{k} {max(v.values()):.2e}" for k, v in parity.items()))
        # 2. D3b streamed grads vs full autograd on the same step, both objectives
        receipt["grad_parity"] = {}
        for objective in ("guided_velocity", "x0_endpoint"):
            rep = d3b_grad_parity(hybrid, objective=objective)
            receipt["grad_parity"][objective] = rep
            if not rep["pass"]:
                refusals.append(f"D3b grad parity ({objective}): worst rel {rep['worst_rel']:.3e}")
            log(f"D3b grad parity {objective}: worst rel {rep['worst_rel']:.3e} over {rep['tensors']} tensors "
                f"({rep['zero_reference_grads']} with a zero reference gradient)")
        # 3. training: each sub-stage's loss decreases
        receipt["training"] = synthetic_training(hybrid, steps=int(args.steps), objective="guided_velocity")
        # 4. D3d with the QVL feature KD on a child that has a QVL-like unit
        built_q = build_tiny_child(Path(tmp) / "child-qvl", with_qvl=True)
        hybrid_q = HybridD3(built_q["dir"], built_q["parent_config"], HybridOptions(dtype=torch.float32, device="cpu", kernels="torch",
                                                                                    lora=lora, lora_seed=3, pin_boundaries=False))
        receipt["training"]["d3d_qvl"] = synthetic_training(hybrid_q, steps=int(args.steps))["d3d"]
        for stage, rep in receipt["training"].items():
            if not rep["last"] < rep["first"] * float(args.decrease):
                refusals.append(f"{stage}: loss {rep['first']:.4g} -> {rep['last']:.4g} (needs < x{args.decrease})")
            log(f"{stage}: loss {rep['first']:.4g} -> {rep['last']:.4g} in {rep['steps']} steps")
    receipt.update({"refusals": refusals, "verdict": "PASS" if not refusals else "FAIL", "seconds": round(time.time() - started, 1)})
    sha = trainer_module().write_json_atomic(Path(args.receipt), receipt)
    log(f"D3 SYNTHETIC {receipt['verdict']}: receipt {args.receipt} sha256 {sha}")
    return 0 if not refusals else 3


def _tiny_d3b_setup(hybrid: HybridD3, seed: int = 1, *, objective: str = "x0_endpoint"
                    ) -> tuple[dict[str, Any], torch.Tensor, int, Callable[[torch.Tensor], torch.Tensor]]:
    """One D3b step's inputs on the tiny ACE unit.  "x0_endpoint": 1.0.2's random-target endpoint MSE
    (kept bit-identical for the parity arm); "guided_velocity": the genuine flow fixture
    (`tiny_flow_batch`) with the parent-raw teacher and the velocity objective."""

    cfg = hybrid.cfg
    if objective == "x0_endpoint":
        seg = tiny_segment(cfg, "ace", rows=8, seed=seed)
    else:
        seg = tiny_flow_batch(cfg, "ace", rows=8, seed=seed)
    context = hybrid.prepare(seg)
    depth = hybrid.layout[0].depth
    with torch.no_grad():
        x_d = hybrid.run_layers(context["x"], 0, depth, context, units_at_stop=False) if depth > 0 else context["x"]
    mask = torch.ones(1, seg["media_rows"])
    if objective == "x0_endpoint":
        g = torch.Generator().manual_seed(seed + 100)
        target = torch.randn(1, seg["media_rows"], TINY.FEATURE, generator=g)

        def loss_fn(prediction: torch.Tensor) -> torch.Tensor:
            return flow_x0_loss(prediction, target, context["types"], mask, hybrid.bands)

        return context, x_d, depth, loss_fn
    loss_fn = make_flow_loss(objective, context, seg, seg["raw"], mask, hybrid.bands, target_kind="parent_raw",
                             noisy_feature_bands=hybrid.noisy_feature_bands(), signs=seg["signs"])
    return context, x_d, depth, loss_fn


def d3b_grad_parity(hybrid: HybridD3, *, objective: str = "x0_endpoint") -> dict[str, Any]:
    """Streamed D3b gradients vs one full autograd graph of the same computation (fp32)."""

    unit_name = hybrid.layout[0].container
    params = hybrid.unit_parameters([unit_name])
    deltas = hybrid.add_emission_deltas([4])
    with torch.no_grad():  # non-zero deltas and LoRA B so every gradient is exercised
        g = torch.Generator().manual_seed(9)
        for d in deltas:
            d.delta.copy_(torch.randn(d.delta.shape, generator=g) * 0.01)
        for name in hybrid.bank.names:
            if name.endswith(".b"):
                hybrid.bank.tensors[name].copy_(torch.randn(hybrid.bank.tensors[name].shape, generator=g) * 0.01)
    context, x_d, depth, loss_fn = _tiny_d3b_setup(hybrid, objective=objective)
    hybrid.freeze_lora_outside(range(depth, hybrid.num_layers))
    lora = hybrid.lora_parameters(range(depth, hybrid.num_layers))
    tensors = params + [d.delta for d in deltas] + lora
    for p in tensors:
        p.grad = None
    loss_streamed = d3b_forward_backward(hybrid, context, x_d, depth, loss_fn)
    streamed = [p.grad.detach().clone() for p in tensors]
    for p in tensors:
        p.grad = None
    # reference: one graph through units + layers d..L-1 + head (layers run with autograd on)
    contexts = hybrid.contexts(x_d)
    with torch.enable_grad():
        h = hybrid.apply_units(x_d, depth - 1, context)
        for i in range(depth, hybrid.num_layers):
            layer = hybrid.engine.layer(i)
            h = hybrid.engine._run(layer, i, h, contexts)
            if hybrid.units_after(i):
                h = hybrid.apply_units(h, i, context)
        loss_ref = loss_fn(hybrid.emission(hybrid.final_norm(h), context))
    loss_ref.backward()
    reference = [p.grad.detach().clone() for p in tensors]
    worst = 0.0
    zero = 0
    for a, b in zip(streamed, reference):
        worst = max(worst, float((a - b).norm() / b.norm().clamp_min(1e-30)))
        zero += int(float(b.norm()) == 0.0)  # a zero reference gradient would make the comparison vacuous
    for p in tensors:
        p.grad = None
        p.requires_grad_(False)
    hybrid.head_deltas.clear()
    for name in hybrid.bank.names:  # restore the bank to a fresh state (B = 0, all trainable)
        with torch.no_grad():
            if name.endswith(".b"):
                hybrid.bank.tensors[name].zero_()
        hybrid.bank.tensors[name].requires_grad_(True)
    return {"pass": worst <= 1e-5 and zero == 0, "worst_rel": worst, "tensors": len(tensors), "zero_reference_grads": zero,
            "objective": objective, "loss_streamed": loss_streamed,
            "loss_reference": float(loss_ref.detach()), "threshold": 1e-5}


def synthetic_training(hybrid: HybridD3, *, steps: int, objective: str = "guided_velocity") -> dict[str, Any]:
    """D3a, D3b, D3c, D3d on the tiny child; every loss must come down (in-memory only).

    1.1.0: D3a trains chart_out against a declared host-write target; D3b runs the guided-velocity
    objective on the genuine flow fixture (or 1.0.2's x0 endpoint with objective="x0_endpoint") and
    leaves no emission delta behind; D3c checks the reserved rows stay frozen; D3d adds the QVL
    feature KD when the child has a QVL-like unit."""

    report: dict[str, Any] = {}
    # D3a: the HY3D-like unit, chart_in -> a fixed random linear map of the host rows, block on pairs,
    # chart_out on the teacher delta -> a fixed random host-write map (the declared target)
    unit_place = [p for p in hybrid.layout if p.source == "hy3d"][0]
    unit = hybrid.units[unit_place.container]
    for module in hybrid.units.values():
        module.requires_grad_(False)
    params = list(d3a_parameters(unit, write_target_declared=True).values())
    for p in params:
        p.requires_grad_(True)
    g = torch.Generator().manual_seed(21)
    host = torch.randn(16, hybrid.hidden, generator=g)
    w_true = torch.randn(unit.chart_in.out_features, hybrid.hidden, generator=g) * 0.1
    w_out_true = torch.randn(hybrid.hidden, unit.chart_in.out_features, generator=g) * 0.1
    p_in = host @ w_true.T
    p_out = p_in + 0.1 * torch.tanh(p_in)
    write_target = ((p_out - p_in) * float(unit.delta_scale)) @ w_out_true.T
    opt = torch.optim.Adam(params, lr=3e-3)
    first = last = None
    write_first = write_last = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, parts = d3a_loss(unit, host, p_in, parent_out=p_out, levels=torch.zeros(16), row_types=torch.full((16,), 7),
                               host_write_target=write_target)
        loss.backward()
        opt.step()
        first = float(loss.detach()) if first is None else first
        last = float(loss.detach())
        write_first = parts["chart_out_rel_mse"] if write_first is None else write_first
        write_last = parts["chart_out_rel_mse"]
    report["d3a"] = {"first": first, "last": last, "steps": steps, "chart_out_first": write_first, "chart_out_last": write_last}
    for p in params:
        p.requires_grad_(False)

    # D3b: ACE-like unit + its emission rows + LoRA on layers d..L-1
    context, x_d, depth, loss_fn = _tiny_d3b_setup(hybrid, seed=4, objective=objective)
    ace = hybrid.layout[0].container
    hybrid.head_deltas.clear()
    deltas = hybrid.add_emission_deltas([4])
    hybrid.freeze_lora_outside(range(depth, hybrid.num_layers))
    trainable = hybrid.unit_parameters([ace]) + [d.delta for d in deltas] + hybrid.lora_parameters(range(depth, hybrid.num_layers))
    opt = torch.optim.Adam(trainable, lr=1e-2)
    first = last = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = d3b_forward_backward(hybrid, context, x_d, depth, loss_fn)
        opt.step()
        first = loss if first is None else first
        last = loss
    report["d3b"] = {"first": first, "last": last, "steps": steps, "objective": objective,
                     "trainable": sum(p.numel() for p in trainable)}
    for p in trainable:
        p.requires_grad_(False)
    hybrid.head_deltas.clear()
    for name in hybrid.bank.names:
        hybrid.bank.tensors[name].requires_grad_(True)

    # D3c: band rows by KL to a synthetic "ACE LM" top-k over the written code rows + LoRA all layers
    codes = 12
    band = BandTrainer(hybrid, lr=2e-2, codes=codes)
    lora_opt = torch.optim.Adam(hybrid.lora_parameters(), lr=5e-3)
    g = torch.Generator().manual_seed(33)
    prompt = torch.randint(0, TINY.VOCAB, (1, 4), generator=g)
    seq = torch.randint(0, codes, (6,), generator=g)
    ids = torch.cat([prompt, (seq + hybrid.music_band_begin).view(1, -1)], dim=1)
    k = 4
    teacher_logits = torch.randn(seq.numel(), codes, generator=g) * 2.0
    lp_full = torch.log_softmax(teacher_logits, -1)
    top_lp, top_ids = lp_full.topk(k, dim=-1)
    lse = torch.zeros(seq.numel())
    lp_band = band_teacher_logprobs(top_lp, lse, lse)
    first = last = None
    for step in range(steps):
        lora_opt.zero_grad(set_to_none=True)
        loss = band.step({"ids": ids}, int(prompt.shape[1]), top_ids, lp_band)
        lora_opt.step()
        first = loss if first is None else first
        last = loss
    report["d3c"] = {"first": first, "last": last, "steps": steps, "embed_rows_updated": band.embed_opt.rows_updated,
                     "head_rows_updated": band.head_opt.rows_updated, "reserved_rows": hybrid.music_rows - codes,
                     "frozen_rows_unchanged": band._frozen_digests() == band.frozen}
    hybrid.band_embed_override = None

    # D3d: answer KD through a type-6 "vision" segment: the bridge's C11-like rows + LoRA (+ the QVL-like
    # unit and its feature KD at 0.1 when the child has one)
    qvl = next((p for p in hybrid.layout if 6 in p.type_ids), None)
    hybrid.bridge_deltas.clear()
    hybrid.add_bridge_deltas([6])
    seg = tiny_segment(hybrid.cfg, "ocr", prompt=6, rows=5, seed=8, suffix=4)
    length = int(seg["ids"].shape[1])
    answer = torch.arange(length - 4, length - 1)  # the answer text after the image rows
    g = torch.Generator().manual_seed(44)
    q_logits = torch.randn(answer.numel(), hybrid.base_vocab, generator=g) * 3.0
    q_lp_all = torch.log_softmax(q_logits, -1)
    q_lp, q_ids = q_lp_all.topk(5, dim=-1)
    targets = answer_kd_targets(length, answer, torch.randint(0, TINY.VOCAB, (answer.numel(),), generator=g), q_ids, q_lp)
    loss_fn = d3d_head_loss(hybrid, targets, denominator=float(answer.numel()))
    trainable = [d.delta for d in hybrid.bridge_deltas] + hybrid.lora_parameters()
    feature = None
    if qvl is not None:
        trainable += hybrid.unit_parameters([qvl.container])
        width = int(qvl.spec.width)
        feature = UnitFeatureKD(qvl.container, torch.randn(1, 5, width, generator=g) * 0.5)
    opt = torch.optim.Adam(trainable, lr=1e-2)
    first = last = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _, _ = full_depth_forward_backward(hybrid, seg, loss_fn, feature=feature)
        opt.step()
        first = loss if first is None else first
        last = loss
    report["d3d"] = {"first": first, "last": last, "steps": steps, "feature_kd": feature is not None,
                     "feature_last": None if feature is None else feature.value}
    for p in trainable:
        p.requires_grad_(False)
    for name in hybrid.bank.names:
        hybrid.bank.tensors[name].requires_grad_(True)
    hybrid.bridge_deltas.clear()
    return report


# --------------------------------------------------------------------------
# Rate cell on real OMNI1 weights (lr = 0; nothing changes, asserted).
# --------------------------------------------------------------------------


#: answer-text tokens a D3d rate cell appends after the media rows (KD at D3D_RATE_SUFFIX - 1 positions)
D3D_RATE_SUFFIX = 8


def grad_group_norms(groups: dict[str, list[torch.Tensor]]) -> tuple[dict[str, float | None], list[str]]:
    """L2 norm of each named group's gradients (None when no tensor in it has a grad).  A group whose
    norm is exactly 0 or not finite is a problem: the path into it is dead (1.0.1's D3d cell reported
    `grads_finite_and_present` with all-zero unit and bridge gradients).  Groups, not tensors: a LoRA
    A factor legitimately gets an exact-zero gradient while its B factor is still zero."""

    norms: dict[str, float | None] = {}
    problems: list[str] = []
    for name, tensors in groups.items():
        grads = [p.grad.detach().float() for p in tensors if p.grad is not None]
        if not grads:
            norms[name] = None
            problems.append(f"{name}: no gradient")
            continue
        value = float(torch.sqrt(sum((g * g).sum().cpu() for g in grads)))
        norms[name] = value
        if not math.isfinite(value):
            problems.append(f"{name}: gradient norm not finite")
        elif value == 0.0:
            problems.append(f"{name}: gradient norm exactly 0 (dead path)")
    return norms, problems


def rate_cell(hybrid: HybridD3, placement: UnitPlacement, seg: dict[str, Any], *, stage: str, steps: int,
              flow_objective: str = "guided_velocity") -> tuple[dict[str, Any], list[str]]:
    """One lr = 0 rate cell: D3b (x_d recapture + units / layers d..L-1 / flow head fwd+bwd) or a
    D3d-shaped full-depth step (bridge deltas + every layer + the unit + text KD on answer text
    appended after the media rows, so the KD gradient reaches the unit and the bridge rows).
    AdamW at lr 0 runs every step; the first trainable tensors are asserted bit-unchanged after, and
    every trainable group (unit, each delta, LoRA) must get a finite non-zero gradient."""

    device = str(hybrid.device)
    cuda = device.startswith("cuda")
    container = placement.container
    if stage == "d3d":
        media_end = int(seg["ids"].shape[1])
        seg = with_text_suffix(seg, D3D_RATE_SUFFIX)
    tokens = int(seg["ids"].shape[1])
    refusals: list[str] = []
    context = hybrid.prepare(seg)
    cell: dict[str, Any] = {"unit": container, "stage": stage, "tokens": tokens, "media_rows": int(seg["media_rows"]),
                            "depth_d": placement.depth}
    hybrid.head_deltas.clear()
    hybrid.bridge_deltas.clear()
    if hybrid.bank is not None:  # a previous cell leaves the bank frozen; every cell starts from a trainable bank
        for name in hybrid.bank.names:
            hybrid.bank.tensors[name].requires_grad_(True)
    if stage == "d3b":
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            x_d = hybrid.run_layers(context["x"], 0, placement.depth, context, units_at_stop=False)
        if cuda:
            torch.cuda.synchronize()
        cell["recapture_seconds"] = time.perf_counter() - t0
        cell["recapture_tok_per_s"] = tokens / cell["recapture_seconds"]
        deltas = hybrid.add_emission_deltas(list(placement.type_ids))
        hybrid.freeze_lora_outside(range(placement.depth, hybrid.num_layers))
        groups = {"unit": hybrid.unit_parameters([container]),
                  **{f"emission_delta.{d.name}": [d.delta] for d in deltas},
                  "lora": hybrid.lora_parameters(range(placement.depth, hybrid.num_layers))}
        g = torch.Generator().manual_seed(1)
        target = torch.randn(1, int(seg["media_rows"]), int(hybrid.cfg.native_feature_dim), generator=g).to(device)
        mask = torch.ones(1, int(seg["media_rows"]), device=device)
        if flow_objective == "x0_endpoint":
            loss_fn = make_flow_loss("x0_endpoint", context, seg, target, mask, hybrid.bands)
        else:
            # the random target is read as canonical velocity: timing only, never production supervision
            loss_fn = make_flow_loss(flow_objective, context, seg, target, mask, hybrid.bands, target_kind="canonical_velocity",
                                     noisy_feature_bands=hybrid.noisy_feature_bands())
        cell.update({"flow_objective": flow_objective, "flow_head": "x0",
                     "teacher_targets": "random rate fixture, not production supervision"})
        run = lambda: d3b_forward_backward(hybrid, context, x_d, placement.depth, loss_fn)  # noqa: E731
    elif stage == "d3d":
        groups = {**{f"bridge_delta.{d.name}": [d.delta] for d in hybrid.add_bridge_deltas(list(placement.type_ids))},
                  "lora": hybrid.lora_parameters(), "unit": hybrid.unit_parameters([container])}
        # KD only on the appended answer text: position p predicts token p + 1, every p after the media
        answer = torch.arange(media_end, tokens - 1)
        if answer.numel() == 0:
            raise D3Error(f"{container}: no answer-text position after the media rows for the D3d-shaped loss")
        cell.update({"answer_positions": [int(answer[0]), int(answer[-1])], "media_end": media_end,
                     "suffix_tokens": tokens - media_end})
        g = torch.Generator().manual_seed(2)
        q_ids = torch.randint(0, hybrid.base_vocab, (answer.numel(), 64), generator=g)
        q_lp = torch.log_softmax(torch.randn(answer.numel(), 64, generator=g), -1) - 1.0
        targets = answer_kd_targets(tokens, answer, seg["ids"][0, answer + 1], q_ids, q_lp)
        loss_fn2 = d3d_head_loss(hybrid, targets, denominator=float(answer.numel()))
        run = lambda: full_depth_forward_backward(hybrid, seg, loss_fn2)[0]  # noqa: E731
    else:
        raise D3Error(f"rate stage {stage!r}")
    groups = {name: tensors for name, tensors in groups.items() if tensors}  # no LoRA bank -> no LoRA group
    trainable = [p for tensors in groups.values() for p in tensors]
    opt = torch.optim.AdamW(trainable, lr=0.0, weight_decay=0.0)
    before = [p.detach().clone() for p in trainable[: min(4, len(trainable))]]
    cell["trainable_parameters"] = int(sum(p.numel() for p in trainable))
    times: list[float] = []
    losses: list[float] = []
    load_before = hybrid.engine.load_seconds
    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = run()
        missing = sum(1 for p in trainable if p.grad is None)
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in trainable)
        norms, dead = grad_group_norms(groups)
        cell.setdefault("grad_norms", []).append(norms)
        refusals.extend(f"{container} {stage} step {step}: {problem}" for problem in dead)
        opt.step()
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss)
        if not finite:
            refusals.append(f"{container} {stage} step {step}: non-finite gradient")
        if missing:
            refusals.append(f"{container} {stage} step {step}: {missing} trainable tensors got no gradient")
        log(f"{container} {stage} step {step}: {times[-1]:.1f}s ({tokens / times[-1]:.1f} tok/s) loss {loss:.4g} grad norms "
            + ", ".join(f"{k} {'none' if v is None else f'{v:.3e}'}" for k, v in norms.items()))
    unchanged = all(torch.equal(a, p.detach()) for a, p in zip(before, trainable))
    if not unchanged:
        refusals.append(f"{container}: an lr = 0 step changed a parameter")
    steady = times[1:] if len(times) > 1 else times
    cell.update({"step_seconds": times, "losses": losses, "tok_per_s": tokens / (sum(steady) / len(steady)),
                 "params_unchanged_at_lr0": unchanged, "grads_finite_and_present": not refusals,
                 "layer_load_seconds": round(hybrid.engine.load_seconds - load_before, 1)})
    if cuda:
        cell["peak_vram_mib"] = round(torch.cuda.max_memory_allocated(hybrid.device) / 2**20, 1)
    cell["host_rss_gb"] = hybrid.T.host_rss_gb()
    for p in trainable:
        p.requires_grad_(False)
        p.grad = None
    hybrid.head_deltas.clear()
    hybrid.bridge_deltas.clear()
    log(f"RATE {container} {stage}: {cell['tok_per_s']:.1f} tok/s steady over {len(steady)} step(s), {tokens} tokens"
        + (f", recapture {cell['recapture_tok_per_s']:.1f} tok/s" if "recapture_tok_per_s" in cell else "")
        + (f", peak VRAM {cell['peak_vram_mib']} MiB" if "peak_vram_mib" in cell else "") + f", RSS {cell['host_rss_gb']} GB")
    return cell, refusals


def command_rate(args: argparse.Namespace) -> int:
    started = time.time()
    torch.set_num_threads(int(args.threads))
    dtype = getattr(torch, args.dtype)
    device = args.device
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    s7 = composite_tool("s7_omni1_check")
    T = trainer_module()
    lora = T.SitePlanConfig(rank=16, alpha=16.0, k3_rank=0, trunk_ffn_width=int(args.trunk_ffn_width), gain=False)
    hybrid = HybridD3(Path(args.omni1), Path(args.parent_config),
                      HybridOptions(dtype=dtype, device=device, head_device="cpu" if args.stage == "d3b" else device,
                                    kernels=args.kernels, lora=lora, lora_seed=0))
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "kind": "rate", "stage": args.stage, "tool": tool_identity(),
                               "argv": sys.argv[1:], "torch": torch.__version__, "device": device, "dtype": args.dtype,
                               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hybrid": hybrid.identity(),
                               "lr": 0.0, "flow_objective": args.flow_objective, "s7_omni1_check": peer_record(s7)}
    if device.startswith("cuda"):
        receipt["card"] = torch.cuda.get_device_name(torch.device(device))
    refusals: list[str] = []
    bands_feature = {t: v["feature"] for t, v in hybrid.bands.items()}
    cells = []
    for container in [u.strip() for u in args.units.split(",") if u.strip()]:
        placement = next((p for p in hybrid.layout if p.container == container), None)
        if placement is None:
            raise D3Error(f"unit {container} is not in the layout {[p.container for p in hybrid.layout]}")
        seg = s7.media_segment(container, Path(args.tap_rows), bands_feature, hybrid.cfg)
        seg = trim_segment(seg, int(args.max_rows_per_type), hybrid.media_placeholder)
        seg["source_positions"] = h3_zero_positions(hybrid.cfg, int(seg["ids"].shape[1]))
        cell, cell_refusals = rate_cell(hybrid, placement, seg, stage=args.stage, steps=int(args.steps),
                                         flow_objective=args.flow_objective)
        cells.append(cell)
        refusals.extend(cell_refusals)
    receipt.update({"cells": cells, "refusals": refusals, "verdict": "PASS" if not refusals else "FAIL",
                    "seconds": round(time.time() - started, 1)})
    sha = T.write_json_atomic(Path(args.receipt), receipt)
    log(f"D3 RATE {receipt['verdict']}: receipt {args.receipt} sha256 {sha}")
    return 0 if not refusals else 3


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("parity", help="hybrid vs the Chimera charted forward on OMNI1, CPU fp32")
    p.add_argument("--omni1", required=True)
    p.add_argument("--parent-config", required=True, help="HF dir whose config.json P1 widens (the Q38 parent)")
    p.add_argument("--tap-rows", required=True, help="S5 host rows root (<unit>/paired_rows.*), s7_omni1_check's input")
    p.add_argument("--receipt", required=True)
    p.add_argument("--max-rows-per-type", type=int, default=24)
    p.add_argument("--tol-rel", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=16)
    s = sub.add_parser("synthetic", help="tiny OMNI-shaped child: parity, grad parity, D3a-d losses decrease")
    s.add_argument("--receipt", required=True)
    s.add_argument("--steps", type=int, default=40)
    s.add_argument("--decrease", type=float, default=0.7, help="last loss must be below first x this")
    r = sub.add_parser("rate", help="lr = 0 fwd+bwd rate on real OMNI1 weights")
    r.add_argument("--omni1", required=True)
    r.add_argument("--parent-config", required=True)
    r.add_argument("--tap-rows", required=True)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    r.add_argument("--kernels", default="torch", choices=("torch", "auto"))
    r.add_argument("--stage", default="d3b", choices=("d3b", "d3d"))
    r.add_argument("--flow-objective", default="guided_velocity", choices=FLOW_OBJECTIVES,
                   help="D3b loss: guided_velocity (the prereg lane loss, x0 head) or x0_endpoint (1.0.2, diagnostic)")
    r.add_argument("--units", default="ace_dit_L12,hy3d_dit_L10")
    r.add_argument("--max-rows-per-type", type=int, default=0)
    r.add_argument("--steps", type=int, default=3)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--trunk-ffn-width", type=int, default=4352)
    r.add_argument("--receipt", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "parity":
            return command_parity(args)
        if args.command == "synthetic":
            return command_synthetic(args)
        if args.command == "rate":
            return command_rate(args)
    except D3Error as error:
        log(f"REFUSED: {error}")
        return 3
    return 2


if __name__ == "__main__":
    sys.exit(main())
