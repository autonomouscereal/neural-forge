"""D3a all-unit refit including the previously unconnected chart_out parameters.

Production requires host rows recaptured on the exact D2 child and independently
sealed host-write targets. Missing chart_out supervision refuses the whole step.
"""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_refit_driver", HERE / "train_media.py")
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)
R, Error = T.R, T.Error


def complete_refit_loss(module, unit, host_rows, parent_in, parent_out, host_write_target, *, levels, row_types, positions, weights, unit_weight):
    import torch
    if host_write_target is None:
        raise Error("D3a requires independent host-write supervision for chart_out; old loss leaves it untrained")
    if (host_rows.ndim != 2 or parent_in.ndim != 2 or parent_out.shape != parent_in.shape or
            host_write_target.shape != host_rows.shape or any(value.requires_grad or not bool(torch.isfinite(value).all())
                                                           for value in (host_rows, parent_in, parent_out, host_write_target))):
        raise Error("D3a native host/parent/write target shape, detachment or finite contract failed")
    if set(weights) != {"chart_in", "block", "chart_out"} or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise Error("D3a amended objective must explicitly weight all three refit terms")
    loss, parts = module.d3a_loss(unit, host_rows, parent_in, parent_out=parent_out, levels=levels, row_types=row_types, positions=positions,
                                 w_in=weights["chart_in"], w_block=weights["block"], weight=unit_weight)
    # The independently captured teacher delta supervises the outward chart.
    # It is detached: this term does not accidentally retarget the teacher block.
    delta = (parent_out - parent_in).to(unit.chart_out.weight.device, unit.chart_out.weight.dtype)
    predicted = delta @ unit.chart_out.weight.T
    owned = torch.ones(host_rows.shape[0], dtype=torch.bool, device=predicted.device)
    if row_types is not None:
        owned = torch.zeros_like(owned)
        for kind in unit.spec.type_ids:
            owned |= row_types.to(predicted.device) == kind
    if not bool(owned.any()):
        raise Error("D3a captured rows contain no owned media write")
    out_loss = module.relative_mse(predicted[owned], host_write_target.to(predicted.device)[owned])
    return loss + unit_weight * weights["chart_out"] * out_loss, {**parts, "chart_out_rel_mse": float(out_loss.detach()), "unit_weight": unit_weight}


class UnitRefitEngine(T.D3bEngine):
    checkpoint_schema = "chimera_recovery.d3a_checkpoint.v1"

    def __init__(self, module, hybrid, unit_name, optimizer, identity, *, total_steps, weights):
        import torch
        expected = {"lr", "betas", "eps", "weight_decay", "clip"}
        if set(optimizer) != expected or any(type(optimizer[key]) not in (int, float) or not math.isfinite(optimizer[key]) or optimizer[key] <= 0 for key in ("lr", "eps", "clip")):
            raise Error("D3a requires explicit finite optimizer settings")
        if (len(optimizer["betas"]) != 2 or any(not 0 <= beta < 1 for beta in optimizer["betas"]) or
                not math.isfinite(optimizer["weight_decay"]) or optimizer["weight_decay"] < 0):
            raise Error("D3a invalid Adam settings")
        self.module, self.hybrid, self.options, self.weights = module, hybrid, dict(optimizer), dict(weights)
        self.identity = {**identity, "optimizer_sha256": R.digest(optimizer), "refit_weights_sha256": R.digest(weights), "total_steps": total_steps}
        self.placement = next((p for p in hybrid.layout if p.container == unit_name), None)
        if self.placement is None or total_steps < 1 or hybrid.head_deltas or hybrid.bridge_deltas:
            raise Error("D3a requires a fresh selected unit and positive budget")
        if identity.get("fixture") is not True:
            for key in ("D2_child_sha256", "recaptured_host_cache_sha256", "teacher_cache_sha256", "host_write_target_sha256", "amendment_sha256"):
                R.require_sha(identity.get(key), "D3a " + key)
            if identity.get("heldout_groups_excluded") is not True or identity.get("chart_out_objective") != "teacher_delta_host_write":
                raise Error("D3a requires admitted independent write targets and amended chart_out objective")
        for unit in hybrid.units.values():
            unit.requires_grad_(False)
        for tensor in hybrid.bank.tensors.values():
            tensor.requires_grad_(False)
        hybrid.unit_parameters([unit_name])
        self.census = {f"unit.{unit_name}.{name}": value for name, value in hybrid.units[unit_name].named_parameters() if value.requires_grad}
        self.optimizer = torch.optim.AdamW(list(self.census.values()), lr=optimizer["lr"], betas=tuple(optimizer["betas"]),
                                          eps=optimizer["eps"], weight_decay=optimizer["weight_decay"])
        self.ownership = R.optimizer_ownership(self.census, self.optimizer)
        self.initial = {name: value.detach().cpu().clone() for name, value in self.census.items()}
        self.step, self.sampler_counter, self.consecutive_skips, self.skip_window, self.total_steps = 0, 0, 0, [], total_steps
        self.decode_review_passed = False

    def train_step(self, host_rows, parent_in, parent_out, host_write_target, *, levels=None, row_types=None, positions=None):
        import torch
        if self.step >= self.total_steps:
            raise Error("D3a exhausted one-pass budget")
        self.optimizer.zero_grad(set_to_none=True)
        unit = self.hybrid.units[self.placement.container]
        unit_weight = .1 if self.placement.type_ids == (6,) else 1.
        loss, parts = complete_refit_loss(self.module, unit, host_rows, parent_in, parent_out, host_write_target, levels=levels,
                                         row_types=row_types, positions=positions, weights=self.weights, unit_weight=unit_weight)
        if not bool(torch.isfinite(loss)):
            self.consecutive_skips += 1
            return {"skipped": True, "step": self.step, "loss": None}
        loss.backward()
        missing = [name for name, value in self.census.items() if value.grad is None]
        if missing:
            raise Error(f"D3a unit parameters unconnected to refit objective: {missing[:3]}")
        if any(not bool(torch.isfinite(value.grad).all()) for value in self.census.values()):
            self.consecutive_skips += 1
            return {"skipped": True, "step": self.step, "loss": None}
        norm = torch.nn.utils.clip_grad_norm_(list(self.census.values()), self.options["clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        self.sampler_counter += 1
        self.consecutive_skips = 0
        self.skip_window = (self.skip_window + [False])[-100:]
        return {**parts, "loss": float(loss.detach()), "skipped": False, "step": self.step, "gradient_norm": float(norm)}
