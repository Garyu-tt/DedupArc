# ZipBurger R10: direct GPT-OSS-20B GGUF activation probe

## Why this exists

R09 collected true layer activations only from a Transformers checkpoint.
R10 is an opt-in, CPU-first standalone llama.cpp experiment that targets
an **existing local GGUF checkpoint** without converting it.

R10 uses the upstream public llama_context_params.cb_eval hook and the
ggml_backend_sched_eval_callback two-phase contract:

- ask=true: subscribe only to the chosen named graph nodes
- ask=false: copy the selected completed node tensor into a small host buffer;
  returning false aborts graph execution on an invalid/missing sample

Names verified in the upstream GPT-OSS graph include l_out, attn_post_norm,
ffn_moe_topk, ffn_moe_logits_biased and ffn_moe_weights_softmax, with
layer suffixes 0 through 23. All original model weights, attention,
expert routers and residual connections are unchanged.

Upstream reference is ggml-org/llama.cpp pinned at
f505c30b92b5579e00532c9e75baa3aa74b70432 on 2026-11-22.

## Data collected

For ONE local prompt, bounded to 8 token IDs, the native binary stores:

- 24 x l_out: final residual activation [tokens, hidden], float32
- 24 x attn_post_norm: input to MoE gate/up [tokens, hidden], float32
- 24 x ffn_moe_topk: selected expert IDs [tokens, 4], int32
- 24 x ffn_moe_logits_biased: [tokens, 32], float32
- 24 x ffn_moe_weights_softmax: [tokens, 4], float32
- Top1 token predictions and the actual input token IDs

The binary magic is ZBR10BIN. It is bounded, little-endian and strictly
validated by gguf_capture_r10.py before export. Unsupported tensor types,
non-contiguous buffers, missing hooks, nonfinite activations and invalid
routing cause fail-closed rejection. The native probe does not capture
down-projection input vectors, full KV tensors or model-level tool use.

Token IDs and activations **can reveal prompt contents**. Never commit
capture files to GitHub. Run only with permission on an isolated host,
and keep source inputs and generated traces private. Avoid running multiple
copies of a 12-GB GGUF on a machine with restricted memory.

## Reproducible developer checks

The ZipBurger R10 GitHub Actions workflow performs two narrow checks:

1. C++17 -fsyntax-only against the exact pinned llama.cpp public headers.
2. Offline Python fixtures for binary checksums, 24-layer trace shape,
   MoE routing, output predictions, token overlap and R09 compatibility.

A successful syntax check is **not** a linked executable, real GGUF run,
performance measurement, or proof that GPT-OSS agenticity survived compression.

## Local experiment

Use a fresh read-only clone of llama.cpp at the pinned commit, not your
running inference service. Build with CPU backend enabled. Link the R10
source in DedupArc at experiments/zipburger/gguf_eval_r10.cpp to that same
llama.cpp + GGML build. The exact linker and CMake build recipe depends
on the locally built static/shared library arrangement.

Example invocation once the executable has actually been compiled:

    gguf_eval_r10 --model /private/models/gpt-oss-20b-MXFP4.gguf \
      --prompt-file /private/prompts/sample.txt \
      --out /private/captures/cal_01.zbr10 --threads 2

A second disjoint prompt corpus must be captured separately.
To convert completed private binaries into R09 and R08-compatible archives:

    python -m experiments.zipburger.gguf_capture_r10 \
      --capture /private/captures/cal_01.zbr10 \
      --trace-npz /private/cal.npz \
      --checkpoint-sha256 VERIFIED_GGUF_SHA256 \
      --source instrumented_teacher

    python -m experiments.zipburger.gguf_capture_r10 \
      --capture /private/captures/holdout_01.zbr10 \
      --trace-npz /private/holdout.npz \
      --checkpoint-sha256 THE_SAME_VERIFIED_GGUF_SHA256 \
      --source instrumented_teacher

    python -m experiments.zipburger.bench_layerwise_r08 --real \
      --teacher-calibration /private/cal_expert_upgate.npz \
      --teacher-validation /private/holdout_expert_upgate.npz

The imported manifest records source binary checksums, exact token-sequence
SHA256 fingerprints and an operator-supplied checkpoint digest.
The model checksum is not independently verified by the importer. Archive
provenance MUST NOT be interpreted as a cryptographic runtime attestation.

The first 512 coordinates of the routed up/gate input are exported per
expert as a bounded calibration tile. Missing experts cannot be assigned
random data; the existing R08 benchmark skips them and the candidate
must use lossless R04 for those weights.

## Non-negotiable release requirements

A real 20B GGUF CPU session must first complete and demonstrate all 24
captured layers and full router fidelity. Compare native teacher and
candidate layer outputs, accumulate errors and evaluate next-token
predictions. Then run real paired multi-turn planning, tool-use,
error-recovery and long-context tasks.

Only after that measure full GGUF file bytes, peak RAM, tokens/sec and
joules/token on the SAME processor or phone. The desired 8-10 GB phone
target, near-idle energy draw and preservation of agenticity remain
**unproven**. Draft research must not ship to production without QA.
