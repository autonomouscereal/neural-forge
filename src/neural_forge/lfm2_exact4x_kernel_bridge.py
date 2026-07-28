"""Lazy bridge to the repository exact-4x composite kernel.

This is deliberately marked transitional.  The dynamic semantic backend is a
normal package module, but executing its current LFM2 kernel still requires
the repository's authenticated experiment runner.  Importing Neural Forge
does not require that runner; execution fails with a precise packaging error
until the large kernel is moved without duplicating it.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any


TRANSITIONAL_REPOSITORY_RUNTIME_DEPENDENCY = True
REPOSITORY_KERNEL_MODULE = "experiments.run_lfm2_exact4x_composite_fold"


class Lfm2Exact4xKernelPackagingError(RuntimeError):
    """The transitional repository kernel is unavailable."""


def repository_kernel_available() -> bool:
    try:
        return importlib.util.find_spec(REPOSITORY_KERNEL_MODULE) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def run_existing_target(**kwargs: Any) -> dict[str, Any]:
    try:
        module = importlib.import_module(REPOSITORY_KERNEL_MODULE)
    except (ImportError, ModuleNotFoundError) as exc:
        raise Lfm2Exact4xKernelPackagingError(
            "The exact-4x semantic backend currently requires the transitional "
            "Neural Forge repository runtime; the experiment kernel is not "
            "included in this installed package."
        ) from exc
    callback = getattr(module, "run_existing_target", None)
    if not callable(callback):
        raise Lfm2Exact4xKernelPackagingError(
            "The transitional exact-4x repository kernel lacks "
            "run_existing_target()."
        )
    result = callback(**kwargs)
    if not isinstance(result, dict):
        raise Lfm2Exact4xKernelPackagingError(
            "The transitional exact-4x kernel returned no JSON receipt."
        )
    return result


__all__ = [
    "Lfm2Exact4xKernelPackagingError",
    "REPOSITORY_KERNEL_MODULE",
    "TRANSITIONAL_REPOSITORY_RUNTIME_DEPENDENCY",
    "repository_kernel_available",
    "run_existing_target",
]
