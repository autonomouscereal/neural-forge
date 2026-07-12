# Assignment-local Q/K/V/O residual fold

Date: 2026-07-11

This is the prebuilt fallback operator for the exact p35
`L19/G4 -> L21/G6` redirect. It is intentionally a reusable primitive, not a
runner, checkpoint, or promotion claim.

## Operator

For each sparse physical `(layer, GQA group)` site, the compact child retains
the shared survivor and adds four independent low-rank residuals:

```text
q_group = q_shared_group(x) + Uq (Rq x)
k_group = k_shared_group(x) + Uk (Rk x)
v_group = v_shared_group(x) + Uv (Rv x)
y       = O_shared(z_all)  + Uo (Ro z_group)
```

The Q/K/V wrappers replace the child codebook projection views at the exact
attributes called by Qwen attention. Consequently:

- Q is corrected before Q normalization and RoPE.
- K is corrected before K normalization, RoPE, and cache append.
- V is corrected before cache append.
- O consumes the real post-softmax physical-group context immediately before
  the attention output is added to the residual stream.

Each calculation is pointwise over sequence positions. The adapter introduces
no cross-token operation, looks at no future token, and does not retain a
parent runtime.

The sparse registry accepts multiple sorted, unique physical sites. The first
exact helper installs only `L19/G4`, after proving that removed source `156` is
absent and that the assignment now uses survivor source `174` from `L21/G6`.

## Initialization and learning boundary

Every role has an output factor `U` and input factor `R`. All four `U` tensors
start at exact zero. All four `R` tensors receive bounded deterministic random
initialization from one declared seed. Installation therefore leaves every
child projection, model output, and cache value bit-exact while preserving a
first gradient into `U`.

The installed handle exposes an adapter-only output-training scope. Inside the
scope every non-adapter model parameter is frozen, while the eight Q/K/V/O
factor tensors remain trainable. On exit, base `requires_grad` flags, gradient
objects, tensor versions, pointers, and model modes are checked and restored.
The contract accepts causal output actions/logits; it has no weight, QKV,
activation, attention, residual, or cache-value similarity target API.

## Rank-8 p35 accounting

For Qwen3-4B:

```text
hidden width                 h = 2560
query-group context width    m = 512
head dimension               d = 128
rank                         r = 8

Q factors = r(h + m) = 24,576
K factors = r(h + d) = 21,504
V factors = r(h + d) = 21,504
O factors = r(m + h) = 24,576
adapter total          = 92,160 parameters
```

The exact tentative one-redirect child removes a `3,276,800`-parameter coupled
prototype and introduces the `36 * 8 * 4 = 1,152`-parameter assignment role-
scale table. Therefore:

```text
net parameters saved versus p35
  = 3,276,800 - 1,152 - 92,160
  = 3,183,488

FP16/BF16 learned payload saved = 6,366,976 bytes
```

This is one-redirect structural accounting. The artifact planner remains
required before any serialized checkpoint-size claim.

## Transaction behavior

Installation owns one sparse registry and wraps Q/K/V/O only in layers that
contain declared sites. It hashes the full base attention registry and records
all base parameter/buffer identities. Rejection restores the original four
projection objects and exact model parameter count.

Rollback validates ownership, wrapper identity, immutable site indices, and
the unchanged child base, but deliberately does not require the isolated
adapter factors to be finite. A failed update that creates NaN or infinity is
therefore still safely removable.

Implemented in:

```text
src/neural_forge/hf_assignment_local_qkvo_residual_adapter.py
```

Focused tests cover exact accounting, multiple sites, state roundtrip,
physical-versus-canonical slicing, strict pointwise prefix causality, Q/K/V
placement before normalization/cache capture, exact-zero model/cache behavior,
adapter-only gradients, base-mutation rejection, finite trained-factor
rollback, and non-finite rejection rollback.

No remote upload or launch was performed. No checkpoint was saved and no fold
was committed.
