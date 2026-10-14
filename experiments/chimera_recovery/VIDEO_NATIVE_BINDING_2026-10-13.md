# Actual temporal video target binding

`video_native.py` independently reviews the unchanged eight-row singleton K64
capture, unchanged complete stock control, authentic full-frame PyAV receipts,
source masks/token mapping, global source admission and real visual block13
features. It seals eight native safetensors rows. Inputs contain only full
native token IDs and 144 raw C11 pixel rows per example in band640:2176;
teacher features and answer targets remain detached supervision.

All eight frames are retained, with temporal patch2 grid[4,6,6], four frame-pair
timestamps and original teacher cumulative lengths[0,36,72,108,144]. The source
is eight procedural training videos; sixteen held-out sources are excluded.
This is an interface/regression baseline, insufficient for general recovery.

The actual capture and independent control agree bitwise on all five raw files
and all792 target positions, ordered top64 IDs, BF16 log probabilities and FP32
LSE. Full parent checks are unchanged. Original producer/cache bytes stay
unchanged. Their stale FFmpeg version strings differ; authenticated execution
receipts identify the same PyAV19.0.1 wheel and libraries, rather than claiming
FFmpeg decoded these videos.

`video_quantized_answer.py` is a versioned video-only consumer. It requires
finite FP32 LSE and exact BF16 representability. Its lower BF16 midpoint mass
is checked against the envelope from half a recorded FP32 LSE ULP. This is
source/rounding-compatible acceptance with actual stock parity, not proof of
exact normalization or correctly rounded library logsumexp. In the genuine
792-row cache,191 stored top64 masses exceed1,20 exceed1.001 and the maximum
is1.0033815256367713. Pure BF16 rounding bounds alone reject thirteen rows;
the selected envelope accepts all792. Target bits are preserved.

The inherited `bucket_terms` loss is unchanged: exp(storedLP), with residual
tail max(0,1-sum(p)). The stored mass and clamped tail are approximate; this
binding claims neither true KL nor an exactly nonnegative loss. Music retains
its distinct admission policy.

The receiver reauthenticates raw source/control seals, producer source pins,
actual entry receipts and stock parity semantics, then compares every tensor
against its independently reviewed native preview. It rejects changed pixels,
features, masks, target IDs, target probabilities and source identity before
model loading. `native_train_ready` remains false until a selected dated
temporal-video amendment/config and real full-depth consumer proof exist.

Actual local validation: seven tests passed, zero skips, including all792 raw
targets, all eight serialized native rows, altered LSE, impossible mass,
nonrepresentable targets, duplicate/out-of-range IDs, wrong masks and source
identity. Serialization used safetensors0.8.0 in an owned private extraction,
wheel SHA096ec1a98435df7beb08853bb5aa9081a84f23d0adc67ed1a0a10550f608373f;
no global package installation. Native serialization round-trips are bitwise.
