# NeuroCrunch ZipBurger R06: GPT-OSS-120B real MXFP4 study

## Provenance and safety

Only **public immutable** Hugging Face GGUF data were accessed by the
credential-free GitHub Actions research runner, not by an inference engine
on Mikoshi. No private data, model service configuration or production
checkpoint was modified.

- Source: `ggml-org/gpt-oss-120b-GGUF`
- Revision: `f59990363fbbb67f01bce03e5e7bf0929e3e320b`
- GGUF: `gpt-oss-120b-MXFP4.gguf`
- File size: **63,387,346,208 bytes**
- SHA-256 from upstream `x-linked-etag`:
  `582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d`
- A file with the same exact size exists in the owner-owned VM200
  `/srv/alt/models/`, but **the local file hash was not recomputed**
  in this run, so file identity is not established by size alone.

The R06 reader fetches a maximum 24 MiB of GGUF header metadata in
4 MiB verified HTTPS `Range` requests, parses v3 with hard bounds,
then reads only a few 128 KiB-or-smaller selected ranges from public
MXFP4 tensor data. No full checkpoint download; no model inference.

## Index findings

- GGUF v3 tensor index: **687 tensors**
- **108 MXFP4** (type 39), 433 type 0, 146 type 8
- Indexed MXFP4 physical payload: **60,914,073,600 bytes**
- MXFP4 fraction of whole GGUF: **96.098%**
- Expert tensor shape example:
  `blk.0.ffn_down_exps.weight`: **[2880, 2880, 128]**
- Four sampled layers: **0, 12, 24, 35**
- Down, gate and up experts for each layer, sampled at two row
  positions: **24 samples**, 48 rows × 16 MXFP4 blocks each

## Real sample results

| Format | Sum of 24 slices | Relative to raw |
|---|---:|---:|
| Original GGUF MXFP4 | **313,344 B** | 100.000% |
| ZipBurger R04 E8M0 delta-bitpack | **301,560 B** | 96.239% |
| zlib-9 | **300,354 B** | 95.854% |
| XZ-6 | **306,856 B** | 97.930% |

- **24/24** exact physical GGUF bit restoration and direct float64
  dot vs dense small-sample numerical oracle
- **24/24** selected R04's `delta-bitpack` mode
- R04 saved **3.761%**, zlib saved **4.146%** on these samples.
  **zlib still outperforms R04 by 1,206 bytes** across this sample.
- No checkpoint-wide compression ratio, actual quantization fidelity,
  llama.cpp tokens/sec or memory traffic inference was measured.

All exact byte offsets, sample SHA-256, sizes, tensor shapes and
samples are stored in
[benchmark_alt120b_r06.json](benchmark_alt120b_r06.json).
The range fetch and corpus scan are independently reproduced in
`.github/workflows/zipburger-alt120.yml`.

## Hardware reality and agent replication

VM200 currently has seven NVIDIA P106-100 6 GiB cards, **42 GiB**
total physical VRAM distributed across the GPUs. One 120B MXFP4 GGUF
is about **59.04 GiB** on disk, exceeding total raw VRAM by about
**17.03 GiB**, before allocating runtime buffers or KV cache.

A pure GPU 120B replica would require **at least 28.9%** total size
reduction, before overhead. Even if the tiny sample's R04 rate
applied unchanged across all indexed MXFP4 bytes, the approximate
model saving would be only **2.29 GB**. This is explicitly a weak
illustrative extrapolation, **not a measured whole-model result**.

The CPU-first option remains viable because VM200 has around 94 GiB
RAM: offload selected layers or paged MoE experts into VRAM while
holding other model weights in host RAM or storage. Its inference
speed and reliability are NOT yet verified. Multiple logical agents
should first share one physical model's weights using **separate KV
contexts and bounded scheduler slots** rather than duplicating a
63 GB checkpoint per agent.

For current GPT-OSS-20B deployment, two separate GPU replicas were
observed; one 4-slot × 2048-token test passed on the 4-card cluster.
The 8-slot rollout suffered a watchdog/startup sequencing conflict and
rolled back to the original two 1-slot model services. No increased
production concurrency has been verified.

## Next experiments

1. **Broader sampling:** at least 4 spatially different MoE expert
   positions per layer, more layers, compare R04/zlib/Zstd, and
   compute nonparametric sample spread before extrapolating.
2. **Structural diagnostics:** entropy of scale bytes and nibble
   streams, exact/reversed/strided block repeat rates, cross-expert
   delta sharing; choose new codec candidates from measured evidence.
3. **Direct CPU runtime:** compare R05 C11 against same-host native
   GGML MXFP4 with cache and thread parity, then evaluate AVX2.
4. **ALT agent fanout:** use existing 300-entry queue and physically
   measured `/slots` counts for an admission-controlled inference
   router. Duplicated agent identities must not imply duplicated GPU
   weight allocations. Continue to reserve VRAM headroom for KV and
   CUDA scratch, and coordinate any restart with the health watchdog.
5. **Production gate:** no 120B checkpoint rewrite, full-model
   equivalence claim, or service cutover without independent QA.

R06 remains an experimental Draft PR; it is not ALT 120B compressed
or deployed.
