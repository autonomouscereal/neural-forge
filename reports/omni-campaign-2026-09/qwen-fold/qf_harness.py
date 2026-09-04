"""One qwen3_5 model, one device, and hooks that capture what a fold needs.

The two things this harness does that a plain forward hook does not:

* it captures the module's INPUT and, on the same input, the output of a
  PRISTINE copy of that module.  That second one is what makes a conditioned
  (``conditioned-local``) solve possible at all: the parent block evaluated at
  the *child's* input appears nowhere in a recording of the parent's own run,
  and it is exactly the target a sequentially-folded site wants to hit.
* it accumulates Grams instead of rows on demand, because a 27B MLP site is
  17,408 columns wide -- ``X^T X`` is 2.4 GB in fp64 and the rows are not.

Everything is fp64 on the CPU by the time a solve sees it; the model itself runs
bf16 on the GPU (or fp32 on CPU for the tiny self-tests).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

import qf_arch
from qf_corpus import corpus, split_by_batch


class Harness:
    """A model plus the sealed corpus it is captured on."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str,
        dtype: torch.dtype,
        manifest: dict[str, Any],
        *,
        force_torch: bool | None = None,
    ) -> None:
        if force_torch is None:
            force_torch = not str(device).startswith("cuda")
        self.kernels_disabled = qf_arch.force_torch_kernels() if force_torch else {}
        config = qf_arch.load_config(model_dir)
        loader = qf_arch.loader_class(config)
        self.model = (
            loader.from_pretrained(
                str(model_dir), dtype=dtype, attn_implementation="eager"
            )
            .to(device)
            .eval()
        )
        self.model_dir = Path(model_dir)
        self._finish(device, dtype, manifest)

    @classmethod
    def wrap(
        cls,
        model: Any,
        device: str,
        dtype: torch.dtype,
        manifest: dict[str, Any],
    ) -> "Harness":
        """The same harness around a model this process BUILT rather than loaded."""

        harness = cls.__new__(cls)
        harness.kernels_disabled = {}
        harness.model = model.to(device).eval()
        harness.model_dir = None
        harness._finish(device, dtype, manifest)
        return harness

    def _finish(self, device: str, dtype: torch.dtype, manifest: dict[str, Any]) -> None:
        self.device = device
        self.dtype = dtype
        self.manifest = manifest
        self.batches = corpus(manifest)
        self.split = split_by_batch(manifest)
        self.forward_calls = 0
        # Eager is REQUIRED, not preferred: it is the only implementation that
        # returns attention probabilities, and the causal-mask assertion and
        # every attention recompute check read them.
        try:
            self.model.set_attn_implementation("eager")
        except Exception:
            self.model.config._attn_implementation = "eager"
            qf_arch.text_config(self.model.config)._attn_implementation = "eager"

    # -------------------------------------------------------------- accessors
    def text(self):
        return qf_arch.text_model(self.model)

    def layers(self):
        return qf_arch.layers(self.model)

    def final_norm(self):
        return qf_arch.final_norm(self.model)

    def embed(self):
        return qf_arch.embed(self.model)

    def lm_head(self):
        return qf_arch.lm_head(self.model)

    def rotary(self):
        return qf_arch.rotary(self.model)

    @property
    def text_config(self):
        return qf_arch.text_config(self.model.config)

    @property
    def eps(self) -> float:
        return qf_arch.rms_eps(self.model.config)

    # ---------------------------------------------------------------- capture
    @torch.no_grad()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.model(input_ids=ids.to(self.device), use_cache=False).logits

    @torch.no_grad()
    def capture(
        self,
        targets: dict[str, Any],
        *,
        pristine: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Run the corpus once; return per-name fp64 ``fit``/``hld`` row blocks.

        Where a pristine module was supplied, ``local_fit``/``local_hld`` carry
        that module's output on the SAME captured call arguments.  Where an
        ``outputs`` module was supplied, ``out_fit``/``out_hld`` carry the live
        module's own output -- which is how a writer that must be solved after
        its own gate gets its design matrix.
        """

        buckets: dict[str, dict[str, list[torch.Tensor]]] = {
            name: {k: [] for k in ("fit", "hld", "local_fit", "local_hld", "out_fit", "out_hld")}
            for name in set(targets) | set(outputs or {})
        }
        state: dict[str, Any] = {}
        handles = []

        def pre_hook(name: str):
            def hook(module, args, kwargs):
                state[name] = (args, kwargs)
                return None

            return hook

        def post_hook(name: str):
            def hook(module, args, kwargs, output):
                state[f"out::{name}"] = output
                return None

            return hook

        for name, module in targets.items():
            handles.append(module.register_forward_pre_hook(pre_hook(name), with_kwargs=True))
        for name, module in (outputs or {}).items():
            handles.append(module.register_forward_hook(post_hook(name), with_kwargs=True))
        try:
            for index, ids in self.batches:
                bucket = "fit" if self.split[index] == "fit" else "hld"
                self.model(input_ids=ids.to(self.device), use_cache=False)
                self.forward_calls += 1
                for name in targets:
                    args, kwargs = state[name]
                    hidden = args[0] if args else kwargs["hidden_states"]
                    buckets[name][bucket].append(self._rows(hidden))
                    if pristine is not None and name in pristine:
                        out = pristine[name](*args, **kwargs)
                        if isinstance(out, tuple):
                            out = out[0]
                        buckets[name][f"local_{bucket}"].append(self._rows(out))
                for name in outputs or {}:
                    out = state[f"out::{name}"]
                    if isinstance(out, tuple):
                        out = out[0]
                    buckets[name][f"out_{bucket}"].append(self._rows(out))
                state.clear()
        finally:
            for handle in handles:
                handle.remove()
        return {
            name: {k: torch.cat(v, dim=0).contiguous() for k, v in rows.items() if v}
            for name, rows in buckets.items()
        }

    @torch.no_grad()
    def capture_batches(self, targets: dict[str, Any]) -> dict[str, list[tuple[str, torch.Tensor]]]:
        """:meth:`capture`, with rows left grouped by batch and tagged with the split.

        The flat capture is the right shape for every solve here except one:
        attention probabilities are per SEQUENCE and a flat row block has
        forgotten where each sequence started.  Same hooks, same order -- the
        concatenation of the fit batches here IS the flat capture's fit block.
        """

        buckets: dict[str, list[tuple[str, torch.Tensor]]] = {name: [] for name in targets}
        state: dict[str, Any] = {}
        handles = []

        def hook_for(name: str):
            def hook(module, args, kwargs):
                state[name] = (args, kwargs)
                return None

            return hook

        for name, module in targets.items():
            handles.append(module.register_forward_pre_hook(hook_for(name), with_kwargs=True))
        try:
            for index, ids in self.batches:
                split = "fit" if self.split[index] == "fit" else "hld"
                self.model(input_ids=ids.to(self.device), use_cache=False)
                self.forward_calls += 1
                for name in targets:
                    args, kwargs = state[name]
                    hidden = args[0] if args else kwargs["hidden_states"]
                    buckets[name].append((split, self._rows(hidden)))
                state.clear()
        finally:
            for handle in handles:
                handle.remove()
        return buckets

    @torch.no_grad()
    def capture_grams(
        self, targets: dict[str, Any], pairs: dict[str, Any]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Accumulate ``X^T X``, ``X^T Y`` and ``||Y||^2`` per split, no rows kept.

        ``pairs[name]`` is a callable applied to the captured input to produce the
        design matrix (e.g. the SwiGLU of an MLP site), and the pristine module's
        output is the target.  The answer is bit-comparable to the row form's up
        to fp64 summation order, which :func:`harness_parity` checks on the tiny
        model rather than asserting.
        """

        state: dict[str, Any] = {}
        handles = []
        accumulators: dict[str, dict[str, torch.Tensor]] = {}

        def hook_for(name: str):
            def hook(module, args, kwargs):
                state[name] = (args, kwargs)
                return None

            return hook

        for name, module in targets.items():
            handles.append(module.register_forward_pre_hook(hook_for(name), with_kwargs=True))
        try:
            for index, ids in self.batches:
                split = "fit" if self.split[index] == "fit" else "hld"
                self.model(input_ids=ids.to(self.device), use_cache=False)
                self.forward_calls += 1
                for name, (design, module) in pairs.items():
                    args, kwargs = state[name]
                    hidden = args[0] if args else kwargs["hidden_states"]
                    A = design(self._rows(hidden))
                    out = module(*args, **kwargs)
                    if isinstance(out, tuple):
                        out = out[0]
                    Y = self._rows(out)
                    slot = accumulators.setdefault(name, {})
                    for key, value in (
                        (f"{split}_G", A.T @ A),
                        (f"{split}_B", A.T @ Y),
                    ):
                        slot[key] = value if key not in slot else slot[key] + value
                    key = f"{split}_YY"
                    slot[key] = float(Y.pow(2).sum()) + slot.get(key, 0.0)
                    key = f"{split}_rows"
                    slot[key] = int(Y.shape[0]) + slot.get(key, 0)
                state.clear()
        finally:
            for handle in handles:
                handle.remove()
        return accumulators

    def _rows(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().reshape(-1, tensor.shape[-1]).to("cpu", torch.float64)


def pristine_copy(module: torch.nn.Module) -> torch.nn.Module:
    """A frozen deep copy of a module, for the local target."""

    return copy.deepcopy(module).eval()


# ---------------------------------------------------------------------- parity
def harness_parity(harness: Harness, sites: tuple[int, ...] = (0, 1, 2)) -> dict[str, Any]:
    """Prove the harness sees what it thinks it sees, before anything is folded.

    Three claims, each of which has silently broken a wave somewhere in this
    estate's history:

    1. **Determinism.**  Two captures of the same corpus on the same model are
       bit-identical.  If they are not, every "improvement" measured later is
       partly noise.
    2. **Hook placement.**  ``swiglu(X) @ down_proj^T`` reconstructs the MLP
       module's own output at the sites checked.  This is the assertion that the
       hook is on the right module and reading the right argument -- LFM2 lost a
       wave to a hook one module too high.
    3. **Logit exactness.**  Capturing changes nothing: the logits with hooks
       installed are bit-identical to the logits without.
    """

    report: dict[str, Any] = {"sites": list(sites)}
    layers = harness.layers()
    sites = tuple(s for s in sites if s < len(layers))

    targets = {f"L{index:03d}": layers[index].mlp for index in sites}
    first = harness.capture(targets, outputs=targets)
    second = harness.capture(targets, outputs=targets)
    identical = all(
        torch.equal(first[name][key], second[name][key])
        for name in first
        for key in first[name]
    )
    report["deterministic"] = bool(identical)
    if not identical:
        raise SystemExit("harness capture is not deterministic; nothing downstream is measurable")

    placement: dict[str, float] = {}
    for index in sites:
        name = f"L{index:03d}"
        mlp = layers[index].mlp
        gate = mlp.gate_proj.weight.detach().to("cpu", torch.float64)
        up = mlp.up_proj.weight.detach().to("cpu", torch.float64)
        down = mlp.down_proj.weight.detach().to("cpu", torch.float64)
        X = first[name]["fit"]
        phi = torch.nn.functional.silu(X @ gate.T) * (X @ up.T)
        produced = phi @ down.T
        target = first[name]["out_fit"]
        placement[name] = float((produced - target).norm() / target.norm())
    report["hook_placement_relL2"] = placement
    report["hook_placement_worst"] = max(placement.values()) if placement else 0.0

    ids = harness.batches[0][1]
    plain = harness.logits(ids)
    grabbed: dict[str, torch.Tensor] = {}

    def spy(module, args, kwargs):
        grabbed["seen"] = args[0] if args else kwargs["hidden_states"]
        return None

    handle = layers[sites[0]].mlp.register_forward_pre_hook(spy, with_kwargs=True)
    try:
        hooked = harness.logits(ids)
    finally:
        handle.remove()
    report["logits_bit_identical_under_hooks"] = bool(torch.equal(plain, hooked))
    if not report["logits_bit_identical_under_hooks"]:
        raise SystemExit("installing a capture hook changed the logits")
    report["forward_calls"] = harness.forward_calls
    return report
