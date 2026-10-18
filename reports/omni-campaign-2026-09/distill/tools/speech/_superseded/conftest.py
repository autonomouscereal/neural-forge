"""conftest.py -- CPU tests never see a GPU (card X-T1; STAGE_D_RECONCILED_PLAN_2026-10-14 sections 4 (D1) and 7.4.4).

Written 2026-10-14 by Claude, lane infra.  One identical copy sits in every directory under distill/tools/ (server
$OC/distill/tools and the local reports tree) that holds test files; test_oc_cuda_guard.py pins that every such
directory has it, byte-equal to this one.

Why: the 2026-10-14 16:26:08Z test run of lane p2p11 (test_p2_v133_summary.py + test_p11_needle_engine.py, from this
directory on the AI server, without CUDA_VISIBLE_DEVICES) put CUDA contexts on the GPUs while D1 and K3 B2 held
them (plan section 2 item 1, X01).  p11_needle_engine.py calls torch.cuda.empty_cache() whenever
torch.cuda.is_available(), and test_k3_teacher_cache_v130.py adds a "cuda" parameter when a card is visible: any
CPU suite can reach a card that a live lane owns.  Hiding has to be a mechanism, not a habit.

What it does, at import (pytest imports a conftest before it collects any test module, so before their torch import):
  * CUDA_VISIBLE_DEVICES="" in os.environ (inherited by every subprocess a test starts: trainers, entry points);
  * pytest_configure: if torch imports, torch.cuda.is_available() must be False and device_count() 0, else the
    session stops with a UsageError before collection.  A torch that already initialised CUDA before this file ran
    is refused too (the variable would be read too late).
Opt-in GPU tests: OC_TEST_ALLOW_GPU=1 together with an explicit, non-empty CUDA_VISIBLE_DEVICES naming the card the
caller owns (the lane's lock / p40.claims line).  OC_TEST_ALLOW_GPU=1 alone is refused: it would expose every card.
A caller that names a card WITHOUT the opt-in is refused too (1.0.1): 1.0.0 silently replaced the card with "", so the
L-K3ACC driver's GPU unit tests (CUDA_VISIBLE_DEVICES=3, `-k "kda_graph or exact_dequant"`) collected no cuda case,
skipped the graph-replay test and still exited 0 (run v130-20261014T231558Z: "3 passed, 1 skipped" where the card run
gives "7 passed").  A gate that cannot reach the card it was given must fail loudly, never pass on CPU.  The variable is
still set to "" first, so even a refused session never opens a card.  "" and "-1" (no card) are not "naming a card".
Not covered: `python test_x.py` (unittest.main) bypasses pytest and this file; bash tests (test_*.sh) stub nvidia-smi.
"""

from __future__ import annotations

import os
import sys

CONFTEST_VERSION = "oc_cuda_guard 1.0.1"  # 1.0.1: a caller-named card without the opt-in is refused, not hidden
ALLOW_ENV = "OC_TEST_ALLOW_GPU"

_ALLOW = os.environ.get(ALLOW_ENV, "") == "1"
_CALLER_CARDS = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()  # what the caller asked for, before we hide it
_CALLER_NAMED_A_CARD = _CALLER_CARDS not in ("", "-1")
_EARLY_TORCH_CUDA = False
if not _ALLOW:
    _torch = sys.modules.get("torch")
    if _torch is not None:
        try:
            _EARLY_TORCH_CUDA = bool(_torch.cuda.is_initialized())
        except Exception:  # a torch without the cuda module cannot have initialised it
            _EARLY_TORCH_CUDA = False
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _refuse(message: str) -> None:
    import pytest

    raise pytest.UsageError(f"[{CONFTEST_VERSION}] {message}")


def pytest_configure(config):
    if _ALLOW:
        if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            _refuse(f"{ALLOW_ENV}=1 needs an explicit CUDA_VISIBLE_DEVICES naming the card this lane holds")
        return
    if _CALLER_NAMED_A_CARD:
        _refuse(f"the caller named card(s) CUDA_VISIBLE_DEVICES={_CALLER_CARDS!r} without {ALLOW_ENV}=1: GPU tests "
                f"need {ALLOW_ENV}=1 (they would otherwise run on CPU and pass without testing the card), CPU tests "
                "need CUDA_VISIBLE_DEVICES='' (or unset)")
    if _EARLY_TORCH_CUDA:
        _refuse("torch initialised CUDA before conftest.py ran (a plugin?); CUDA_VISIBLE_DEVICES='' came too late")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        _refuse("CUDA_VISIBLE_DEVICES was changed after conftest.py set it")
    try:
        import torch
    except ImportError:  # no torch in this interpreter: nothing can open a card through it
        return
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        _refuse(f"torch {torch.__version__} still sees {torch.cuda.device_count()} CUDA device(s) with "
                "CUDA_VISIBLE_DEVICES=''")


def pytest_report_header(config):
    if _ALLOW:
        return f"{CONFTEST_VERSION}: GPU tests allowed ({ALLOW_ENV}=1), CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
    return f"{CONFTEST_VERSION}: CUDA hidden (CUDA_VISIBLE_DEVICES=''); {ALLOW_ENV}=1 + an explicit card to opt in"
