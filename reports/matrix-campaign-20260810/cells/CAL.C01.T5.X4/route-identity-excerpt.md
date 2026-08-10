# CAL.C01.T5.X4 -- route identity excerpt (from route-stdout.log)

## codebook phase and artifact
"phase":"lfm2-sparse-semantic-codebook","phase_key_sha256":"8c814244e9f00519f3650b180fa12ee031e9f9bd904af00e018214d51456adb0"
"artifact_sha256":"ad04189483bd1654e07d308be5408e37077ff317f82111fba90740b96d43c821","completion_sha256":"92a521032277e4070786e038bbc08087cba0579b17283097a6e2df7d7abb56ee"
"artifact_sha256":"82c497f4fd6a6b390560b57e145f62a27a3e03c0a06616c5b916f75005a7baa8","completion_sha256":"3b672fa0e5455d5996b2ff1d597c5281ddb2f89e413fe8d3af54f5a9a73906bf"

## candidate inventory phase key
"phase":"lfm2-residual-omp-semantic-candidate-inventory","phase_key_sha256":"79917b7b3dcc7ee847d6aa049419ffecde9e336fe31f336e6bdd07c7c32fb8ed"

## terminal failure
50010:run-error:
50011-Traceback (most recent call last):
50012-  File "/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952/experiments/run_lfm2_sparse_semantic_codebook.py", line 5504, in main
50013-    result = run(
50014-             ^^^^
50015-  File "/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952/experiments/run_lfm2_sparse_semantic_codebook.py", line 4715, in run
50016-    trace_cache = Lfm2CircuitTraceCache.open(
50017-                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
50018-  File "/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952/src/neural_forge/hf_lfm2_circuit_trace_cache.py", line 4862, in open
50019-    _validate_manifest(manifest)
50020-  File "/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952/src/neural_forge/hf_lfm2_circuit_trace_cache.py", line 3970, in _validate_manifest
50021-    raise Lfm2CircuitTraceCorruptionError(
50022-neural_forge.hf_lfm2_circuit_trace_cache.Lfm2CircuitTraceCorruptionError: Cache implementation differs from the current collector/runtime and is not an explicitly compatible sealed revision.
