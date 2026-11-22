# ZipBurger R11: 64 independent probes per GGUF model load

## Goal

Avoid loading the ~12.1 GB GPT-OSS-20B checkpoint again for every
short calibration prompt. R11 does not replace model computation or
prove a compressed model is agentically equivalent.

The new gguf_eval_batch_r11.cpp shares the existing R10 callback and
native ZBR10BIN format. It loads one local model on the CPU, executes
1..64 small prompt prefills sequentially, clears the model KV state
between prompts and writes one private trace file per prompt. A
completion marker is emitted only when every prompt finishes.

## Build in an isolated pinned llama.cpp checkout

    cmake -S experiments/zipburger/r10_native -B /private/build-r11 \
      -DLLAMA_CPP_ROOT=/private/llama-r10 -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF
    cmake --build /private/build-r11 --target zipburger_gguf_batch_r11 -j2

## Run on an authorized host

Prepare calibration and validation prompt directories, separately.
Each contains up to 64 regular .txt files. Every prompt must tokenize
to 1..8 tokens and have at most 2 KiB of text. Do not include secrets.

    /private/build-r11/zipburger_gguf_batch_r11 \
      --model /private/models/gpt-oss-20b-MXFP4.gguf \
      --prompt-dir /private/prompts/calibration \
      --out-dir /private/captures/cal --threads 2

    /private/build-r11/zipburger_gguf_batch_r11 \
      --model /private/models/gpt-oss-20b-MXFP4.gguf \
      --prompt-dir /private/prompts/validation \
      --out-dir /private/captures/val --threads 2

Output directories must be NEW. The program uses private Unix
file permissions and refuses symlink prompt files. The importer
rejects incomplete runs without complete.marker or with missing,
out-of-order or modified numbered capture files.

Import the completed batches into R09/R08:

    python -m experiments.zipburger.gguf_capture_r10 \
      --capture-dir /private/captures/cal \
      --trace-npz /private/r11-cal.npz \
      --checkpoint-sha256 ACTUAL_VERIFIED_GGUF_SHA256 \
      --source instrumented_teacher

    python -m experiments.zipburger.gguf_capture_r10 \
      --capture-dir /private/captures/val \
      --trace-npz /private/r11-val.npz \
      --checkpoint-sha256 THE_SAME_ACTUAL_SHA256 \
      --source instrumented_teacher

    python -m experiments.zipburger.bench_layerwise_r08 --real \
      --teacher-calibration /private/r11-cal_expert_upgate.npz \
      --teacher-validation /private/r11-val_expert_upgate.npz

The importer checks exact token-sequence digests, calibrate/holdout
disjointness and the checkpoint identifier supplied by the operator.
The model file checksum is NOT recalculated by the importer.
It exports first-512-coordinate up/gate inputs for matching expert
slices only. Down-projection data remains unavailable.

## Limits

CI builds and tests the executable and rejects malformed/incomplete
fixture traces. No live 20B session has been performed or measured.
This approach removes redundant disk loads but does not reduce
inference compute requirements, guarantee an 8-10 GB phone RAM
budget, or demonstrate low battery power. R08 quality gating must
continue to reject lossy candidates until REAL activation and
full-model agentic benchmarks prove them acceptable.
