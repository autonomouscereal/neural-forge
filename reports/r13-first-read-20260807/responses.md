# R13 published child — NON-PROTOCOL first decode read

**This is not the frozen manual whole-response review, and nothing here is
scored.** The frozen review protocol (sealed at `1499df58`) owns scoring;
this file is a raw decode smoke taken so the program can see the published
child speak at all. No response below has been graded, ranked, filtered,
truncated, or cleaned. Every character between the fences is what the model
emitted.

## Circumstances

| | |
|---|---|
| run directory (read-only) | `/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold` |
| published child | `/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold/published-child/child` |
| certificate_sha256 | `27450bcd19422a1cc73bb35f1125cce289115b8a91224861c89e9d8a99b9089b` |
| child_state_fingerprint | `53f96b385d072e6b87060ded64a22f8adde80cd01e0341e08cf10dfb9714252d` |
| artifact_tree_sha256 | `376417b1fe0b50952f7eb35a7e48c13bde02a23fa040299da461715198780232` |
| carrier_child_package_sha256 | `503f2898981ef3917503b4ec2cb3331843e81863e92cc757be52d36339fa9096` |
| loader | `Lfm2CarrierChildCheckpointIO.from_child_directory` |
| entrypoint | `semantic_step_generate` |
| device | `cuda:0` |
| dtype | `torch.float32` |
| max_new_tokens | `256` |
| maximum_prompt_tokens | `8192` |
| prompt pack | `/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a/research/lfm2_semantic_whole_response_prompts_v1.json` |
| prompt pack sha256 | `0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b` |
| runtime_sha256 of this read | `acd61a0ec3a1fff8f7901dba9acb10da02e9b0cc5700a301c21b34cbbd76e29c` |
| physical dense body parameters | `53552368` |
| effective counted parameters | `57420144` |
| standalone load seconds | `14.395` |
| decode seconds (10 prompts) | `59.992` |

Decoding is greedy: the routed runtime selects each donor token with
`torch.argmax`, no sampling, no temperature. The loader is the same
standalone child-local path `run_lfm2_resumable_review_from_packaged_child`
and the catalog's own post-publication reload use
(`Lfm2CarrierChildCheckpointIO.from_child_directory` over a
`DirectLocalArtifactIO` base), so `registration is None` — no build-time
state is bound, only what was written to disk.

Prompt selection: the first 10 non-safety rows of the frozen pack, in pack
order. Safety rows are excluded because they belong to the separate safety
observation, not to an intelligence read.

Operation audit for this read:

````````text
{
 "training_steps": 0,
 "optimizer_constructions": 0,
 "optimizer_steps": 0,
 "autograd_grad_calls": 0,
 "backward_calls": 0,
 "parent_forward_calls": 0,
 "parent_generate_calls": 0,
 "child_generate_calls": 0,
 "child_lm_head_forward_calls": 0,
 "child_body_prefill_forward_calls": 10,
 "child_body_generated_forward_calls": 2295,
 "child_body_forward_calls": 2305,
 "prompt_donor_semantic_tokens_advanced": 580,
 "generated_donor_semantic_tokens_advanced": 2295,
 "target_span_metadata_tokens_observed": 3353,
 "target_span_tokens_used_as_body_steps": 0,
 "whole_string_target_retokenizer_calls": 10,
 "undefined_donor_rows_scored": 0,
 "owned_learned_parameter_count": 0,
 "routing_fixed_continuous_state_count": 3867776,
 "routing_input_fixed_continuous_state_count": 1933888,
 "routing_output_fixed_continuous_state_count": 1933888,
 "routing_input_output_coefficient_storage_shared": false,
 "effective_child_parameter_count": 57420144,
 "semantic_code_coefficients_counted_exactly_once": true,
 "registered_parameter_count_delta": 0,
 "registered_module_count_delta": 0,
 "quantized_parameters": 0,
 "pruned_parameters": 0,
 "network_access": false
}
````````

## Responses, verbatim

### 1. `conversation-natural-greeting` — conversation

* finish_reason: `length`
* completion_status: `censored`
* ended_with_eos: `False`
* target response tokens: `267`
* donor response tokens: `256`
* target_roundtrip_exact: `True`

**Prompt**

````````text
[
 {
  "role": "user",
  "content": "Hi! I had a long day. Can you respond naturally and ask one thoughtful follow-up question?"
 }
]
````````

**Decoded response (`response`, verbatim)**

````````text
        codecode Registcodecode}/",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",","),
"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"] ", ",~\ ",~\~\ ",~\~\~\ ",~\~\~\ ",~\~\ ",~\ ",~\ ",~\ ",~\ ",~\ ",~\ ",~\ ",~\ ", Treat Treat Treat Treat Treat Treat Treat Treat Treat Treat Treat Treat Treat Treat Treat Treat Treat2222222222222222222222222222222222222222222222222222222222222222222222222222222222            
````````

### 2. `instruction-three-actions` — instruction-following

* finish_reason: `length`
* completion_status: `censored`
* ended_with_eos: `False`
* target response tokens: `266`
* donor response tokens: `256`
* target_roundtrip_exact: `True`

**Prompt**

````````text
[
 {
  "role": "user",
  "content": "I keep forgetting small household tasks. Give me exactly three practical actions I can take today. Use a numbered list and no introduction."
 }
]
````````

**Decoded response (`response`, verbatim)**

````````text
22222222222222othothothothothothothothothothothothothothothothothothoth33""""""""""""""""""),GHGHGHGHGHGHGHGHGHGHGHGHGHGHGHomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomberberberberberberberberberberberberberber--""""""""""""""""""""""""""""""""""""""""""""""""""""")",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)", case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case case
````````

### 3. `explanation-seasons` — knowledge-explanation

* finish_reason: `eos`
* completion_status: `complete`
* ended_with_eos: `True`
* target response tokens: `0`
* donor response tokens: `1`
* target_roundtrip_exact: `True`

**Prompt**

````````text
[
 {
  "role": "user",
  "content": "Explain why Earth has seasons to a curious twelve-year-old. Also correct the common idea that summer happens because Earth is much closer to the Sun."
 }
]
````````

**Decoded response (`response`, verbatim)**

````````text

````````

### 4. `arithmetic-multi-step` — reasoning

* finish_reason: `length`
* completion_status: `censored`
* ended_with_eos: `False`
* target response tokens: `388`
* donor response tokens: `256`
* target_roundtrip_exact: `True`

**Prompt**

````````text
[
 {
  "role": "user",
  "content": "A workshop makes 17 kits per hour for 6 hours. Eight kits fail inspection, then the remaining kits are packed equally into 5 boxes. How many complete kits go in each box, and how many are left over? Show the steps."
 }
]
````````

**Decoded response (`response`, verbatim)**

````````text
                 ",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",”)",",”)",",”)”)",",”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)",",",",",",",",",",",",”,”,”,)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",", We We We We We We We We We We We We We We We We We We We We We We We We We We We We We We We We We We We We We We We We We We Whe Whe Whe Whe Whe Whe Whe Whe Whe Whe Whe Whe next next next next next next next others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others
````````

### 5. `logic-ordering` — reasoning

* finish_reason: `length`
* completion_status: `censored`
* ended_with_eos: `False`
* target response tokens: `359`
* donor response tokens: `256`
* target_roundtrip_exact: `True`

**Prompt**

````````text
[
 {
  "role": "user",
  "content": "Mira finished before Theo. Theo finished before Jun. Sal finished after Jun. Put all four people in finish order and explain which facts force the answer."
 }
]
````````

**Decoded response (`response`, verbatim)**

````````text
                    ",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",”)"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]medimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedimedi
````````

### 6. `context-retrieval` — context-use

* finish_reason: `length`
* completion_status: `censored`
* ended_with_eos: `False`
* target response tokens: `358`
* donor response tokens: `256`
* target_roundtrip_exact: `True`

**Prompt**

````````text
[
 {
  "role": "system",
  "content": "Use only the supplied note. If the note does not answer something, say so."
 },
 {
  "role": "user",
  "content": "NOTE: The community garden opens at 7:30 a.m. Tuesday through Saturday. Tool checkout ends at 4:15 p.m. Visitors may bring dogs only on Saturdays, and dogs must remain leashed. The north gate is closed this month for repairs.\n\nQuestion: I want to bring my dog and borrow a shovel. Which day is allowed, what is the latest checkout time, and which gate should I avoid?"
 }
]
````````

**Decoded response (`response`, verbatim)**

````````text
",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",",”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”)”))",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",ainsainsainsainsainsainsainsainsainsainsainsainsainsainsainsainsainsainsainsomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomesomes
````````

### 7. `summarization-priorities` — summarization

* finish_reason: `length`
* completion_status: `censored`
* ended_with_eos: `False`
* target response tokens: `313`
* donor response tokens: `256`
* target_roundtrip_exact: `True`

**Prompt**

````````text
[
 {
  "role": "user",
  "content": "Summarize this update in two sentences, preserving the decision and the unresolved risk:\n\nThe team compared three battery suppliers. Supplier A is cheapest but cannot guarantee delivery before launch. Supplier B costs 11 percent more and passed every reliability test. Supplier C can deliver fastest, but two samples overheated. The team selected Supplier B. The remaining risk is that its factory expansion may slip by one week, so procurement will request weekly schedule evidence."
 }
]
````````

**Decoded response (`response`, verbatim)**

````````text
""""""""""""""""""""""""""""""""""""""""""""""")"")"")"")"")"")"")"")"")
"
"
"
"
"
"
"
"
"
"
"
")
")
")
")
")
")
")
")
")
")
")
")
")
")
")
")
")
")
")
")
")
")
"
"
"
"
"
"
"
"
"
"
"
GERbee"]"
TestsTestsTestsTestsTestsTestsTestsTestsTestsTestsTestsTestsTestsTestsTestsTestsTestsTestsTestsomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomomom-raythroughomomomomom
````````

### 8. `comparison-ram-rom` — conceptual-comparison

* finish_reason: `length`
* completion_status: `censored`
* ended_with_eos: `False`
* target response tokens: `377`
* donor response tokens: `256`
* target_roundtrip_exact: `True`

**Prompt**

````````text
[
 {
  "role": "user",
  "content": "Compare RAM and long-term storage using one simple analogy, then state two important ways the analogy is imperfect."
 }
]
````````

**Decoded response (`response`, verbatim)**

````````text
          ",",",",",",",",",",",",",",",",",",",",",",",",",",",",”,”,”,”)”)”)”)”)”)”)”)”)”)”)”)”)"));
"));
)」)」)」)」)」 pro pro pro pro pro pro pro pro pro pro pro pro pro pro pro pro pro pro pro pro pro Whe cont���)"
�� Whe Whe Whe Whe Whe Whe Whe Whe Whe Whe"]"]"]"] Whe"] Whe"] Whe"] Whe"] Whe Whe Whe Whe Whe Whe Whe Whe Whe oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper oper * *ally oper oper oper * *  build2222222222222222222222222222222222 against ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' '222222 log2 log2 log2
````````

### 9. `coding-deduplicate` — coding

* finish_reason: `length`
* completion_status: `censored`
* ended_with_eos: `False`
* target response tokens: `483`
* donor response tokens: `256`
* target_roundtrip_exact: `True`

**Prompt**

````````text
[
 {
  "role": "user",
  "content": "Write a Python function named unique_in_order(items) that removes duplicates while preserving first-seen order. It must work with hashable values. Include one short example and explain the time complexity."
 }
]
````````

**Decoded response (`response`, verbatim)**

````````text
            codecodeell""",",",",",",",",",",",",",",",",",",",",",",",",",",","),
",",","),
",","),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),",",”)"),
"),
"),"),",",”)”)"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"]"] &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ &\ (− &\ (− (− (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−medi (−
````````

### 10. `debugging-off-by-one` — debugging

* finish_reason: `length`
* completion_status: `censored`
* ended_with_eos: `False`
* target response tokens: `546`
* donor response tokens: `256`
* target_roundtrip_exact: `True`

**Prompt**

````````text
[
 {
  "role": "user",
  "content": "This function should return the last three elements, but it returns four for many inputs. Identify the bug and provide a corrected version.\n\ndef last_three(values):\n    start = max(0, len(values) - 3)\n    return values[start - 1:]"
 }
]
````````

**Decoded response (`response`, verbatim)**

````````text
"""""""""""""""""""""""""""",",",",",",",",",",",",",",",",",",",",",",",","),
",","),
","),
","),
","),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),"),
"),"),"),"),"),"),"),"),"),"),"),"),"),"),"),"),"),"),"),"),"),"),"),"),"),"),"),)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",)",*)*)"),
"),
"),
"),
"),
 Ober)", Ober2"),
"),
"),
"),
"),
"),
"),
 Ober)", Ober2"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),
"),

````````

## What this file deliberately does not say

* No pass/fail, no score, no rubric, no comparison against the frozen
  review protocol's criteria. That evaluation is a separate, sealed job.
* No claim about r13's intelligence. A 10-prompt greedy read at a 256-token
  cap is a smoke, not a measurement.
* The full frozen pack is 16 prompts including one safety row; 6 rows were
  not decoded here.

Source JSON (server, read-only):
`/home/cereal/nf-w13m/claude-r13-read/first-read.json`
