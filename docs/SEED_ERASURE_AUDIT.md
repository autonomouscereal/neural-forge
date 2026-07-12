# Seed-erasure audit

`neural_forge.hf_seed_erasure` is a cheap, fail-closed diagnostic for comparing
two or more independently seeded compile outcomes. It is model-family neutral:
the compiled objects only need to be PyTorch modules. It does not compile a
model or establish capability retention.

The audit requires exact agreement on:

- module/config structure, logical parameter and buffer names, shapes, dtypes,
  persistence, training state, and storage aliases;
- the declared target parameter budget and the counted unique parameters;
- every named tensor's exact SHA-256 content digest;
- compiler-report topology and every undeclared report value; and
- a caller-supplied inventory of deterministic behavioral/output fingerprints.

Missing or renamed fields, budget mismatches, callback mutation, undeclared
differences, nonexistent exceptions, container-wide report exceptions, and
exceptions that do not actually differ all fail closed.

```python
import hashlib

from neural_forge.hf_seed_erasure import (
    SeedCompileOutcome,
    SeedOwnedExceptionScope,
    SeedOwnedField,
    audit_seed_erasure,
)


def behavior_fingerprints(outcome):
    # Run the same fixed, deterministic evaluation suite for every outcome.
    raw_result = run_fixed_eval(outcome.model)
    return {"fixed-eval-v1": hashlib.sha256(raw_result).hexdigest()}


outcomes = [
    SeedCompileOutcome(
        seed=seed,
        model=completed_compile.child_model,
        parameter_budget=completed_compile.report.target_parameter_budget,
        compiler_report=completed_compile.report,
        label=f"compile-{seed}",
    )
    for seed, completed_compile in independently_seeded_compiles
]

report = audit_seed_erasure(
    outcomes,
    behavior_fingerprints,
    exceptions=SeedOwnedExceptionScope(
        compiler_report_paths=(
            SeedOwnedField(
                name="/seed",
                reason="The requested initialization seed is provenance metadata.",
            ),
        ),
    ),
)

assert report.seed_erasure_proven
```

Exceptions are exact and narrow. A declared tensor or behavioral difference
still sets `seed_leak_detected=True` and `seed_erasure_proven=False`; the
declaration merely records that the difference was intentional. Compiler
report leaf exceptions remove seed-owned provenance values from report
equivalence, but cannot hide missing/renamed fields or model differences.

The proof scope is finite. `seed_erasure_proven=True` means that no initial
randomness remains in the audited named tensors or supplied behavioral suites
for the specific completed compiles. It is not a claim about untested seeds,
untested inputs, arbitrary model behavior, or a real model that has not yet
been compiled and audited.
