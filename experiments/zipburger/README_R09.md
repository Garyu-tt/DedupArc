# ZipBurger R09: instrumented teacher traces and 24-layer agentic gate

## Research scope

This PR remains research-only. No production model, Mikoshi inference service,
mobile device or other deployed system has been modified.

- layer_trace_r09.py: verified 24-layer activation, top4 route and next-token
  prediction trace format with SHA256-linked manifest, bounded NPZ and no pickle.
- hf_capture_r09.py: opt-in local Transformers GPT-OSS-20B forward hooks for
  each decoder layer and MoE router; collects routed *up/gate* expert inputs.
  Down-projection inputs are not accessible with this hook.
- expert_trace_input_r09.py: checks separate calibration/validation archives,
  rejects missing experts and aligns 2880-row expert boundaries. Different
  corpus hashes alone do not prove the corpora are disjoint.
- bench_layerwise_r08.py: optionally uses those observed up/gate activations
  with real bounded GGUF weight slices, skips unavailable/down observations;
  never substitutes random vectors when actual traces were requested.
- agentic_gate_r09.py: paired executed tool/planning/recovery/long-context
  results. Tool-call sequences may differ, but task success and safety must
  not regress.
- trace_r09_cli.py: checks full 24-layer output drift, preserved top4
  routing, next-token predictions, and optionally paired executed tasks.
  Regression returns exit code 2.

The new offline tests use synthetic fixtures and **do not establish that
full-model GPT-OSS-20B activations or an agentic session were executed**.

## Opt-in local teacher instrumentation

Existing GGUF (~12 GB) is NOT directly supported by the HF hook sampler.
Running it requires an already-local Transformers-formatted checkpoint on
an owner-approved host with sufficient memory. It loads with
local_files_only=True and trust_remote_code=False. Model download and
hardware/service access are never performed in CI.

Prepare two distinct private JSON arrays of non-sensitive prompts,
for example /private/calibration.json and /private/holdout.json.
Use the **actual owner-verified checkpoint SHA256 identifier**, not an
invented hash. Capture them separately:

~~~bash
python -m experiments.zipburger.hf_capture_r09 \
  --local-model-dir /private/gpt-oss-20b-hf \
  --prompts-json /private/calibration.json \
  --trace-npz /private/r09_cal.npz \
  --checkpoint-sha256 ACTUAL_64_HEX_DIGEST \
  --source instrumented_teacher --device cpu

python -m experiments.zipburger.hf_capture_r09 \
  --local-model-dir /private/gpt-oss-20b-hf \
  --prompts-json /private/holdout.json \
  --trace-npz /private/r09_holdout.npz \
  --checkpoint-sha256 THE_SAME_64_HEX_DIGEST \
  --source instrumented_teacher --device cpu

python -m experiments.zipburger.bench_layerwise_r08 --real \
  --teacher-calibration /private/r09_cal_expert_upgate.npz \
  --teacher-validation /private/r09_holdout_expert_upgate.npz \
  --output /private/r09_teacher_weight_probe.json
~~~

The capture is bounded to 8 prompts x 32 tokens, and at most 256
observations per expert. It stores numeric activations, routes and
top1 predictions, not raw prompt text, yet activation traces may still
contain sensitive information and must stay private.

The expert-input manifests are operator declarations, not proof that
a given HF checkpoint is numerically identical to the pinned GGUF.
Full provenance parity must be verified separately. Kernelized MoE
implementations may bypass hooks; this collector fails if any layer's
router/expert/hidden hook is missing.

## Paired 24-layer and agentic regression evaluation

This step requires a separately implemented candidate runtime and
actual paired executions, which are NOT part of R09.

~~~bash
python -m experiments.zipburger.trace_r09_cli \
  --teacher-trace /private/teacher.npz \
  --candidate-trace /private/candidate.npz \
  --teacher-tasks /private/teacher_tasks.jsonl \
  --candidate-tasks /private/candidate_tasks.jsonl \
  --output /private/quality_report.json
~~~

Each task record permits only: case_id, family (planning, tool_use,
error_recovery, long_context), environment_sha256, completed (bool),
unsafe_actions, recovered_failures, tool_calls (list of tool names,
no raw tool arguments). Minimum 32 paired cases with all four families.

A matching output does not prove general intelligence or agenticity.
Even a clean report explicitly has model_agenticity_proven=false
and production_release_approved=false.

## Offline CPU fixtures

~~~bash
python -m unittest discover -s tests -p 'test_zipburger*r09.py' -v
python -m unittest discover -s tests -p 'test_zipburger*.py' -v
python -m experiments.zipburger.bench_layerwise_r08
~~~

Production gates remain: same-runtime original/candidate 20B logits,
per-layer drift and router behavior, real multi-turn agent outcomes,
full-checkpoint disk size, peak CPU RAM, speed and energy/power.
