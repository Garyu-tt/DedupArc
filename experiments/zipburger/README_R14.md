# ZipBurger R14: expert-aware CPU memory paging feasibility

R12 proved the **storage** limitation: a naive 17-byte MXFP4 block
replaced by a 13-byte R08 3-bit block cannot bring the entire
12.110 GB GGUF under 8 decimal GB even in the impossible ideal case.
However model file bytes need not all be **resident** simultaneously.
A memory-mapped 12 GB file can be larger than RAM, at the cost of
storage reads/page faults and potentially severe latency/energy.

R14 models one narrow alternative: **MoE expert paging** while preserving
all original expert weights and router decisions. This is only an
offline LRU simulator, not a working GGUF paged execution backend.

## Weight size assumptions from official 20B architecture

Each of 24 layers has 32 experts with three 2880 x 2880 MXFP4
projection matrices (gate, up, down). At 17 bytes per 32 weights:

- Full expert: 13,219,200 bytes (12.61 MiB)
- 24 x 32 expert projections: 10,152,345,600 bytes (approx 10.15 GB)
- Other bytes in the pinned 12,109,566,624-byte GGUF:
  1,957,221,024 bytes. These include attention, embedding,
  routers, headers, etc. (estimate from model layout, not RSS).
- One decode token selects four experts per layer, 96 expert
  activations total. If none are cached, up to 1,269,043,200
  bytes (1.269 GB) of full expert weights would have to be read
  to perform those accesses before actual CPU math.

R14 keeps an LRU cache of whole, original, uncompressed expert pages.
It tracks cache hits, misses, evictions, worst-token storage bytes
and peak modeled resident expert bytes.

## Offline reproducible experiments

Synthetic routing, clearly marked and **not** real model output:

    python -m experiments.zipburger.expert_cache_r14 \
      --synthetic-tokens 64 --target-gb 8 \
      --reserve-runtime-gb 1 --output /tmp/r14-synthetic-8gb.json

Real routing **only if** R11 GGUF captures have been completed on an
owner-authorized host:

    python -m experiments.zipburger.expert_cache_r14 \
      --capture-dir /private/captures/cal \
      --target-gb 8 --reserve-runtime-gb 1 \
      --output /private/r14-route-cache.json

R14 reads 24-layer top4 expert selections from native R10/R11 binary
captures; no private prompt text is needed in its JSON report.
The resident estimate deducts all nonexpert GGUF file bytes plus
the specified additional OS/KV/runtime reserve from the target RAM
budget. It does not allocate any 20B tensors or launch inference.

## What this cannot prove

- It models *one-token-at-a-time decode* accesses, not the exact GGML
  prefill scheduling order, nor the physical Linux page cache.
- It assumes entire expert weights move atomically, but real kernels
  may use smaller chunks and OS read-ahead; misses are not measured I/O.
- Uncompressed expert storage is physically 12.1 GB on disk; this
  model reduces **resident RAM**, not checkpoint storage bytes.
- No measured tokens/sec, energy, battery drain, phone compatibility,
  inference correctness, real user tasks or preserved agenticity.
- A cache hit rate with synthetic uniform routing says nothing about
  actual GPT-OSS-20B routing; real teacher R11 captures are required.
- Heavy flash access on a phone may waste more energy than the
  compressed-domain inference approach. Measure device I/O first.

The best candidate may ultimately combine true per-layer weight
compression, small expert cache, and learned decoding, with the
original attention/router paths untouched until quality verified.
Do not merge or deploy this as a solution to 8 GB until real
GGUF/candidate and multi-step agentic performance are benchmarked.
