# Shared r16 stage inheritance, 2026-10-13

`stage_inheritance.py` imports an authentic full shared adapter bank into the existing physical HybridD3 tensors before a stage constructs its optimizer. D3a/b/c/d production wrappers now require an explicit policy; a fresh independently initialized bank is no longer their implicit continuation behavior. The primitive also accepts the audio engine through its ordinary `hybrid`/`census`/`identity`/checkpoint interface; an audio production wrapper must call initialization before constructing that engine.

Use `shared_r16: {"mode":"initialize","first_refit":true}` only for the explicitly designated first D3a refit. Every subsequent stage uses `shared_r16: {"mode":"inherit","source":{"path":"<previous shared-r16/manifest.json>","sha256":"<actual manifest file hash>"}}`. Regenerate source-pinned stage configs after this driver change. Older same-stage checkpoints/configs have a different identity and require their original driver; they are not silently resumed by this changed implementation.

The source is schema `chimera_recovery.shared_r16.v1`, a full bank replacement. Metadata preflight verifies the actual safetensors file and bounded header, exact immutable base/parent-config/hybrid-module pins, complete name/shape/fp32 census, site plan, helper code SHA, and the producing complete checkpoint's receipt/state/identity/step pins. Runtime checks the destination bank's real site plan, dtype, leaf status, finiteness and physical storage ownership, then checks all source tensors before copying. It preserves the tensor objects and their gradient flags and imports no optimizer state. The initialization receipt and complete source manifest SHA enter the stage's exact-resume identity.

The final export includes every leaf, including frozen layers absent from the current trainable census. Export compares every current trained LoRA leaf against its actual saved checkpoint and rejects changed frozen leaves against the pre-optimizer snapshot. The prior full bank remains a complete replacement source; there is no addition of independently initialized A/B banks. Actual upstream bank content is present before D3b's x_d recapture and prefix digest.

This resolves shared adapter transport. Learned unit state, emission slices, input bridge deltas and music embed/head masters still need an explicit sequential native compose/import implementation. Every bank export records those remaining obligations and `complete_native_checkpoint_composed: false`. It does not supply a D4 union engine, numerical final merge parity, registered child/clone or native content evaluation.

Run local stdlib metadata tests with:

```text
python -B -m pytest -q tests/test_chimera_shared_r16_inheritance.py tests/test_chimera_d3a_manifest_identity.py
```

The optional actual CPU fixture requires Torch, safetensors, transformers, the real Chimera import package and the authentic campaign module. Set `CHIMERA_D3_HYBRID=/home/cereal/omni-campaign/distill/tools/d3_hybrid_module.py`, then run the same tests using the existing CPU-capable runtime. It builds random tiny weights through the genuine importer, updates the genuine bank with AdamW, checks exact HybridD3 forward parity after a different-seed import, constructs the real D3b optimizer, takes a real native forward/backward step, and verifies that frozen lower-layer leaves survive another full-bank handoff. Fixture passing proves transport behavior on that tiny model, not production model intelligence or full native composition.
