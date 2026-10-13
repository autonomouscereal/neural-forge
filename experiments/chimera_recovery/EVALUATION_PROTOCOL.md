# Recovery evaluation protocol — 2026-10-13

This protocol grades **complete recorded outputs** from the current model before
and after recovery. It loads no model, executes no historical command, and starts
no training. Passing the preparation tests does not establish model intelligence.

`python -m experiments.chimera_recovery.evaluation prepare-bfcl` extracts genuine
validation/test BFCL tasks from the sealed normalized corpus. Its required source
hashes pin the heldout and training files. Each source group must be absent from
training. Authentic function schemas and **all recorded acceptable parameter
alternatives** are retained; captured optional empty alternatives permit omission
only for non-required parameters. These tasks grade proposed calls and parallel
batches. No successful execution, error, retry or multi-turn result is invented.

`inputs.jsonl` contains only runner-visible inputs. `tasks.jsonl` also contains
private grading oracles: never feed the whole task record to the model, tokenize
its oracle, or include targets in the prompt. Keep both files frozen before the
baseline. Root can add genuine heldout tool-seed trajectories once they include
source family/group/split and independent assertions. Re-seal the **whole** task
set, then use that same set for both runs; a changing holdout is a new experiment.

Each task has `schema=recovery_eval_task.v1`, `id`, `group_id`, heldout `split`,
`route`, `input={messages,tools,media}` and an independent `oracle`. Arithmetic
oracles declare an expected value and tolerance. Python function oracles declare
the function name and independently authored input/output cases, including edge
cases. The bounded AST interpreter supports a documented pure subset; unsupported
valid programs remain ungraded for external review. Alternate numeric answers,
explanations, algorithms and valid tool argument alternatives can receive credit.
Open reasoning and media tasks require external semantic evaluation; missing
oracles stay ungraded rather than count as success.

Tool oracles declare `steps`, each with a `calls` batch; a parallel step requires
multiple calls in the same assistant message. Calls specify `name` and either
`accepted_arguments` (whole argument dictionaries) or `argument_choices`, with
explicit `optional_arguments`. An optional `outcome` declares `status=success|error`
and exact content or equivalent JSON. Error/retry tests need distinct, linked
error and success outcomes in the expected sequence. Multi-turn tests declare
exact `followup_user_messages`; the recorder supplies these scripted environment
turns without putting future answers into the initial prompt. Final answers may
have their own independent arithmetic/Python oracle. Other final-answer semantics
require external review.

A response has `schema=recovery_eval_response.v1`, the task `id`,
`input_sha256=SHA256(canonical task.input)`, complete newly generated/environment
`messages`, and `generation={stop_reason:eos|length|error,max_new_tokens,token_ids?}`.
Do not repeat the initial input messages in generated messages. Preserve raw full
answers; any transport parser must retain raw output and document exact parsing.
Tool messages keep original call IDs and JSON-string function arguments. Recorded
`tool_outcomes` contain the linked `tool_call_id`, observed `status`,
`source_verified=true`, actual `receipt_sha256` and
`result_sha256=SHA256(canonical full tool result message)`. Receipts are provenance,
not permission to execute commands. A proposed call can pass a call-only task while
execution is explicitly unobserved. Missing or wrong-ID outcomes fail tasks that
require execution. The evaluator never fabricates or runs tool results.

The `recovery_eval_run.v1` receipt declares `phase=baseline|after_recovery`,
`model.checkpoint_sha256`, all known `training_group_ids`, and `pins` for the exact
holdout task file, admitted corpus, tokenizer, actual chat template, input
configuration, decoding configuration and runtime. Pin any media adapter/config
in additional fields. `responses_sha256` is SHA256 of the canonical response list.
The recorder must declare `recording_complete=true` and `full_answers_published=true`.
Every heldout task appears exactly once, including empty, capped or error responses.
Actual source receipts and generated runtime logs must be kept beside these claims;
hash declarations alone are not independent proof a runtime used those assets.

`grade --tasks tasks.jsonl --receipt run.json --responses responses.jsonl --output NEWDIR`
publishes a sealed JSON report and `full_answers_and_outcomes.md` with all inputs,
answers and observed outcomes. Three distinct result groups are reported:

1. Semantic correctness: pass/fail/ungraded from independent oracles.
2. Generation: empty answers, EOS, caps and consecutive repetition candidates.
   A correct fragment followed by a loop can pass a value oracle while failing
   complete-response success. A repetition candidate is a diagnostic, not IQ.
3. Teacher agreement: teacher-forced top-1/top-k on **reported positions**, with
   explicit position coverage; separately free-run positional matches, LCP, and
   exact first **5, 8, 16, 32 actual tokens**. Short EOS outputs are unavailable at
   longer horizons. Top-k next-token membership never implies consecutive rollout
   matching. Alternate correct answers may disagree with the teacher.

Teacher metrics require a task `teacher_reference` with exact token IDs, teacher
checkpoint SHA and teacher-forced context SHA. Optional `teacher_forced` response
records declare that exact context/checkpoint, `k` and unique positions with ranked
token IDs. Missing positions are not filled in as passes. Free-run token IDs must
come from the same pinned tokenizer. Neither teacher agreement nor finite logits
can substitute for semantic grading.

`compare --before before/report.json --after after/report.json --output NEWDIR`
refuses changed holdout/input/group/corpus/template/tokenizer/decoding/runtime or
grader pins. After recovery must name the baseline checkpoint and a real training
receipt SHA. The comparison publishes every before/after answer and outcome, both
count sets, and model identities. Report model correctness only after collecting
and grading those actual runs; preparation or synthetic test results are not a
baseline.
