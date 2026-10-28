# Production batched all-channel FIR benchmark

Date: 2026-07-19

This benchmark used the current dirty canonical `neural-forge` implementation
of `solve_condition_capped_batched_all_channel_fir_ridge_candidate`.

## Correct production geometry

- Target channels: 776
- Fit cases: 8
- Valid rows per case: 5,082
- Aggregate valid rows per channel: 40,656
- FIR features: 4 (three physical lags plus one physical bias coordinate)
- Authenticated FP64 input payload: 1.175317 GiB
- Condition cap: 1,000
- CPU threads: 16

The initial 8 x 40,656 fixture interpreted the profiled aggregate row count as
a per-case count. It was stopped after the semantic correction. Its preserved
partial evidence reached 53.492966 GiB sampled peak RSS, but it represents an
8x oversized 325,248-row fixture and is not the production result.

## Measured production result

- Batched solver wall time: 22.754448 seconds
- Full process wall time from `/usr/bin/time`: 47.25 seconds
- Harness total excluding import/launch: 42.610895 seconds
- Conservative sampled peak RSS: 7.585445 GiB
- `/usr/bin/time` maximum RSS: 7,945,876 KiB (7.577778 GiB)
- Peak increment over startup baseline: 7.066792 GiB
- Margin below a 64 GiB process cap: 56.414555 GiB
- Result finite: yes
- Result authenticated: yes
- All input cases authenticated: yes
- Condition cap, parameter trust, function trust, and nullspace checks: passed
- Aggregate solver report SHA-256:
  `32ee3037536358a0bedad54af8073617d1bd49db5f717d9084211713782b202c`
- Result artifact SHA-256:
  `254918144e928fe3efd0898150a9f7e28df85026218eb6e4fdbae1b89dc31297`

For this component, a 64 GiB process cap is safe. Adding the measured 7.067 GiB
increment to the live worker's previously observed approximately 36.1 GiB RSS
gives an estimated approximately 43.2 GiB. Even adding it to the previously
observed 49.2 GiB high-water mark gives approximately 56.3 GiB. This is a
component-level measurement, not a claim that unrelated simultaneous
allocations cannot consume the remaining margin.

## Independent scalar comparison

A fixed-seed simple random sample without replacement selected 64 of 776
channels (8.247% of the population), independently of solver outputs.

- Scalar subset wall time: 4.050213 seconds
- Transparent full-population scalar extrapolation: 49.108834 seconds
- Batched solver versus extrapolated scalar wall-time ratio: 2.1582x
- Maximum FP64 absolute difference: 4.44089209850063e-16
- FP64 mismatches above 5e-11: 0
- Installed FP32 bytes: exact identity
- Installed BF16 bytes: exact identity
- Exact hypergeometric one-sided 95% upper bound after zero sampled
  mismatches: 34 of 776 channels (4.3814%)

FP32 batched/scalar payload SHA-256:
`036cf486422f0fe1990e8c8b3ef1f7efa34d5070f6088070955ede74461097d1`

BF16 batched/scalar payload SHA-256:
`ce52a5c4472c3e64602c14bb11ff74ba1ac725c2cc4c01a856724e27c9923ad5`

## Exact production command

```bash
/usr/bin/time -v \
  -o /home/cereal/model-surgery-lab/benchmarks/production-batched-fir-40656-codex-20260719/benchmark-artifacts/full-776x8x5082.time.txt \
  nice -n 10 taskset -c 56-71 \
  env \
    PYTHONPATH=/home/cereal/model-surgery-lab/benchmarks/production-batched-fir-40656-codex-20260719/src \
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    OPENBLAS_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
  /home/cereal/model-surgery-lab/venv/bin/python \
  /home/cereal/model-surgery-lab/benchmarks/production-batched-fir-40656-codex-20260719/benchmark-artifacts/production-batched-fir-40656-benchmark.py \
  --channels 776 \
  --cases 8 \
  --rows-per-case 5082 \
  --features 4 \
  --condition-cap 1000 \
  --scalar-sample 64 \
  --threads 16 \
  --seed 20260719 \
  --output /home/cereal/model-surgery-lab/benchmarks/production-batched-fir-40656-codex-20260719/benchmark-artifacts/full-776x8x5082.json
```

## Artifacts

- `full-776x8x5082.json`: authenticated benchmark result
- `full-776x8x5082.time.txt`: independent `/usr/bin/time -v` measurement
- `full-776x8x5082.log`: stage-by-stage RSS/timing log
- `production-batched-fir-40656-benchmark.py`: reproduction harness
- `full-776x8x40656.overscale-final-status.txt`: preserved incorrect
  per-case interpretation evidence
