# ZipBurger R03: read-only GGUF MXFP4 probe

This adds a real-checkpoint reader to the research-only R03. It does not
modify a checkpoint, run model inference, use GPU resources, or change ALT.

## GGUF vs safetensors

The original R03 safetensors reader expects separate .blocks and .scales
tensors. The llama.cpp GGUF instead stores GGML_TYPE_MXFP4 (39) in
**17-byte physical blocks**, each containing a **single E8M0 scale byte
followed by 16 packed E2M1 bytes** (32 logical weights, low nibble first).

The bounded probe extracts only selected row/block ranges and turns
scale-first GGUF blocks into R03's separate scale/mantissa arrays.
The stride is (first GGUF dimension / 32) × 17 bytes per tensor row.
No entire model, tensor, or decompressed matrix is memory-mapped.

## Real model index observed via JediControl on Mikoshi VM200

- File: /srv/alt/gpu-legion-stage/gpt-oss-20b-MXFP4.gguf
- Repository: ggml-org/gpt-oss-20b-GGUF
- Immutable revision: 41e8a94d15d62070b36cf4553a0250717ecf46b8
- Verified file length: 12,109,566,624 bytes
- Previously validated whole-file SHA-256:
  27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901
- GGUF v3: 459 tensor entries, 36 metadata entries
- 72 tensors of GGML MXFP4 (type 39), 289 of F32, 98 of F16
- Example tensor: blk.0.ffn_down_exps.weight, dimensions [2880,2880,32]

These are real index observations, **not R03 compression measurements**.
The two active GPU Legion llama-server processes remain unchanged.

## Read-only command

From the DedupArc PR #15 research checkout, with NumPy installed:

~~~sh
OPENBLAS_NUM_THREADS=1 python -m experiments.zipburger.gguf_probe_r03 \
  --gguf /srv/alt/gpu-legion-stage/gpt-oss-20b-MXFP4.gguf \
  --tensor blk.0.ffn_down_exps.weight \
  --rows 48 --groups 4 --row-start 0
~~~

Additional tensor candidates include blk.0.ffn_gate_exps.weight,
blk.12.ffn_down_exps.weight, and blk.23.ffn_up_exps.weight.
The reader limits any request to at most 128 rows × 16 consecutive blocks.

It evaluates exact GGUF byte-for-byte reconstruction (including signed zero
and scale bytes), direct float64 R03 GEMV against a dense small-sample oracle,
and ZipBurger frame size against raw/zlib-9/XZ-6, each using the real GGUF
scale-first byte order. Frame sizes exclude neural adapters and runtime RSS.
The direct experimental Python implementation is not a fused GPU kernel.

## Current evidence and release boundary

The current model source, GGUF tensor layout, numerical format and VM200 model
availability are confirmed. A remote attempt to execute the **new real-tensor
R03 compression benchmark** was blocked by the remote-execution security gate.
Consequently **real R03 compression ratios, packing latency and direct-dot
latency have not yet been observed for these tensors**. Do not derive them from
synthetic benchmarks.

Only fake GGUF fixtures are exercised by the new automated unit tests.
The real checkpoint and GPU Legion services were not changed.
Keep PR #15 Draft pending exact-HEAD CI, independent QA and authorized
real tensor-slice measurements. Related research: issue #14.
