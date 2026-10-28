import json

from neural_forge.hf_lfm2_cpcf_selective_evidence import (
    lfm2_cpcf_selective_provider_implementation,
)


print(json.dumps(dict(lfm2_cpcf_selective_provider_implementation()), sort_keys=True))
