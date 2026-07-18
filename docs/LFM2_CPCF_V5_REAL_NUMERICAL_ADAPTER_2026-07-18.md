# LFM2 CPCF v5 real numerical-frontier adapter

`hf_lfm2_cpcf_short_conv_v5_numerical_adapter.py` is the concrete FIT-only
generator for the frozen generic numerical-frontier seam. It is not a toy
constant adapter and it does not alter the existing compiler, replay,
selection, validation, or installation code.

## Dynamic request contract

The adapter derives all dimensions and optional-bias topology from
`Lfm2CpcfShortConvNumericalFrontierRequest`:

- source and target hidden widths;
- source and target FIR lengths;
- source B/C/x, FIR, writer, and bias tensors;
- output-space encoder;
- blank target tensor shapes and optional biases;
- authenticated FIT trajectories and cached parent contexts; and
- the existing fixed projection-pass count and context-objective weight.

It does not contain model-family-specific layer indices or fixed hidden sizes.
The current FIR basis is a compression basis, so it fails closed when target
hidden width exceeds source hidden width.

## Audited dependency order

1. Every source FIR row is resized to the target FIR length.
2. Every FIT case contributes its own activation and residual-endpoint
   descriptor block.
3. The v5 balanced basis partitions every source channel exactly once and
   gates the actual per-lag FIR intertwining/commutator residual.
4. The v5 ridge primitive fits child-to-parent input lifts for the frozen
   condition caps `1e3` and `1e4`.
5. Each cap transports actual parent B/C/x, FIR, and writer tensors into the
   blank target topology.
6. Cached parent contexts are projected into authenticated FIT-only projection
   cases.
7. Each pass proposes:
   - an affine writer solve when a physical writer bias exists, otherwise a
     no-intercept writer solve;
   - a no-intercept FIR solve whose design contains any physical FIR-bias
     coordinate; and
   - a no-intercept linearized B/C/x solve whose design contains any physical
     in-projection-bias coordinates.
   The authenticated v5 ridge policy remains the outer bound, while the
   existing per-factor writer/FIR/B/C/x trust radii narrow the corresponding
   parameter-space correction when they are stricter.
8. A proposal joins the authenticated path only when the FIT coupled
   endpoint/context objective does not regress. Rejected proposals are
   reverted and excluded.
9. The fixed v5 interpolation grid is materialized on every adjacent accepted
   edge for both condition caps.
10. Every anchor and every interior state is converted to the frozen typed raw
    candidate. Exactly one candidate ID ends in `--anchor-00-prior`.

## Evidence and execution boundaries

The adapter has no selection or validation inputs. It performs no candidate
selection; the existing compiler ranks later using its existing replay
boundary. It also performs:

- zero parent or target model forward calls;
- zero generation calls;
- zero backward, optimizer, or training steps;
- zero pruning; and
- zero quantization.

The numerical work is deterministic detached FP64 closed-form arithmetic. The
adapter does not claim that passing its local algebraic/authentication tests
proves broad intelligence retention. That claim still requires the existing
external selection/validation replay and the end-to-end experiment gate.

## FIR gate policy

The default `Lfm2CpcfV5FirBasisPolicy` is intentionally strict. A caller may
provide a different authenticated tolerance, but the chosen policy digest,
actual commutator residual, threshold, and pass result are all bound into the
frontier provenance. The adapter never derives a permissive threshold from the
same trajectories and never silently relabels a failed basis as commuting.

Before enforcing the requested commutator threshold, the adapter performs a
deterministic measurement pass with identical descriptors, grouping, and
weights. If the requested gate fails, it raises
`Lfm2CpcfShortConvV5NumericalAdapterError` with a sealed `diagnostics` mapping
that includes the requested threshold, observed maximum, every per-lag
residual, source-to-target grouping, and the complete authenticated audit-basis
report. A runner must persist that mapping rather than retaining only the
exception string.

For the first real 1024-to-128 exploratory run, the recommended explicit
preregistered policy is projected off-diagonal gate `0.0` and actual
commutator-relative-Frobenius gate `1.0`. This is a permissive measured
representability gate, not a claim of exact commuting. The adapter default
remains `1e-12`; the runner must opt into `1.0` visibly and record both the
threshold and observed residual.
