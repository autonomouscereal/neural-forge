"""Media fold harness v1 for the Chimera Omni campaign (lane: media).

Three capabilities, one package:

* ``c15_pairs`` / ``c15_fit`` -- C15 VAE absorption. A generic (raw patch <-> latent)
  pair sampler over a pluggable VAE adapter, and a closed-form (ridge / kernel-ridge)
  map fitter that composes the fitted maps into Chimera's boundary projection
  (``NativeContinuousMediaBridge.feature_proj``) and emission head
  (``UnifiedNativeMediaEmissionHead.out``) shapes, with reconstruction metrics scored
  against the parent VAE's own round-trip ceiling.
* ``ace_dit`` / ``c16_capture`` / ``c16_transplant`` -- C16 DiT block transplant. A
  self-contained, weight-faithful ACE-Step 1.5 turbo DiT (transcribed from ComfyUI's
  ``comfy/ldm/ace/ace_step15.py``), block I/O capture, residual chart fits
  (ridge + orthogonal Procrustes + SVD control), the five transplant arms
  (replace / insert / control / identity / random-chart), and a >=4x channel coreset
  of a transplanted block's MLP.
* ``h3_0_proof_rung`` (script at the lane root) -- the H3-0 rung on Chimera's 3.19M
  joint-AV proof trunk.

Hard constraints honoured here: closed form only (no gradients in the fold stage), no
quantization, storage counted in PARAMETERS, artifacts verified by sha256 rather than
exit codes, and every threshold reported as a proposed decision line rather than an
invented gate.

Runtime: python >= 3.10, torch, numpy, safetensors. Deliberately free of any ORM /
SQLAlchemy / pydantic dependency, and free of ComfyUI imports outside
``vae_adapters.ComfyVaeAdapter`` (which only ever runs inside the ComfyUI image).
"""

from __future__ import annotations

__all__ = [
    "receipts",
    "linalg",
    "vae_adapters",
    "c15_pairs",
    "c15_fit",
    "ace_dit",
    "c16_capture",
    "c16_transplant",
]

VERSION = "media-fold-harness-v1"
