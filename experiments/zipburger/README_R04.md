# NeuroCrunch ZipBurger R04: directly usable lossless MXFP4 scale packing

**Research-only.** This is a new exact, compact representation of real
GPT-OSS-20B MXFP4 samples, not a new full model checkpoint, fast inference
engine, or proof of general compression superiority.

## Design

A GGML MXFP4 block stores one E8M0 scale byte then 16 packed E2M1 bytes
for 32 weights. R04 keeps each 16-byte E2M1 block in its original packed
form and replaces the E8M0 scale stream with the smallest of:

- **Delta-bitpack:** one scale base plus fixed-width bit-packed deltas.
  Exact O(1) scale access; works especially well when scales occupy a
  narrow range, even with many repeated values.
- **Palette-bitpack:** dictionary of actual scale bytes plus fixed-width
  indexes. Useful when the active scales are sparse but far apart.
- **Raw fallback:** complete original 17-byte interleaved blocks, if neither
  compact encoding wins after all format overhead is included.

Each complete ZBP4 frame includes dimensions, mode, bit width, metadata,
packed scales, packed E2M1 payload and a SHA-256 digest of original
interleaved GGUF bytes. Both `read_code(row,col)` and
`read_scale(row,group)` use O(1) reads, and `dot(x)` operates a single
group of 32 weights at a time, without constructing a full dequantized
weight matrix. `restore()` is only a test/export function.

R04 is a scale/bitpacking baseline, **not** the same neuron-graph
factorization as R03. Neural adapters are not part of its compressed
frame, and their memory cost is not included.

## Actual pinned GPT-OSS-20B GGUF experiments

Benchmark provenance:
- `ggml-org/gpt-oss-20b-GGUF` revision
  `41e8a94d15d62070b36cf4553a0250717ecf46b8`
- Model: `gpt-oss-20b-MXFP4.gguf`, 12,109,566,624 bytes
- The whole-file SHA-256 is published upstream and was previously
  independently verified on VM200, but it was **not recomputed by
  this GitHub Actions range-reader**.
- 9 MoE matrices: `blk.{0,1,2}.ffn_{down,gate,up}_exps.weight`
- Two starting rows per tensor: 0 and 4096
- Each sample: 48 rows × 16 blocks × 17 bytes = **13,056 bytes**
- 18 samples, 235,008 bytes in total, fetched in 18 bounded HTTP 206
  range requests (up to ~72 KB per range), no 12 GB checkpoint download
- Exact per-sample range offsets, SHA-256, mode and timings recorded in
  [benchmark_real_gguf_r04_palette.json](benchmark_real_gguf_r04_palette.json).

| Format | Total size | Fraction of original |
| --- | ---: | ---: |
| Original GGUF MXFP4 samples | 235,008 B | 100.000% |
| R03 neural graph / raw fallback | 235,800 B | 100.337% |
| **R04 delta/palette scale-bitpack** | **226,386 B** | **96.331%** |
| zlib-9 on original interleaved bytes | 226,407 B | 96.340% |
| XZ-6 on original interleaved bytes | 231,172 B | 98.368% |

**18/18 samples selected delta-bitpack, not palette or raw.**
Every sample recovered exactly the original packed nibbles and scale
bytes and matched a small dense float64 matvec within tolerance.

R04 reduces the sampled bytes by **3.669%** and beats zlib-9 on the
combined sample by **21 bytes**, a difference of under **0.01 percentage
points**. This narrow result does **not** establish that R04 beats zlib
across the full checkpoint, different layers, larger blocks, or different
quantized models. The Python dot path is not an optimized kernel and can
be slower than a dense NumPy matvec.

This particular R04 frame also retains uncompressed nibble payload;
its measured footprint includes the complete frame header and scale
metadata but excludes Python/NumPy runtime allocation and any separately
trained adapters.

## Reproduction

Run the offline unit and fake GGUF tests from repository root:

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_zipburger*.py' -v
```

For bounded **public real** GGUF samples, this workflow runs the
immutable URL listed above and rejects incorrect HTTP 206 ranges,
untrusted redirects, unexpected file sizes, and oversized responses:

```sh
OPENBLAS_NUM_THREADS=1 python -m experiments.zipburger.real_mxfp4_ranges_r04 \
  --row-start 0 --row-start 4096 --rows 48 --groups 16
```

The linked GitHub Actions workflow
`.github/workflows/zipburger-real-weights.yml` runs the same command
independently of Mikoshi / ALT. The experiment requires outbound HTTPS
and verifies the *sample byte hash* it prints; the entire model's hash
cannot be recomputed from small ranges.

## Next gates

1. Measure additional middle/end layers and non-contiguous expert
   regions before projecting a whole-checkpoint reduction.
2. Implement a compiled CPU SIMD kernel with inlined E8M0 decoding,
   E2M1 lookup and direct scale access. Measure apples-to-apples against
   GGML MXFP4 GEMV and account for CPU/RAM traffic and engine overhead.
3. Compare R04 at multiple tile/group sizes and investigate compressed
   residual nibble structure; report the literal/bitpacked fallback rate.
4. Prove model-level numerical tolerance at real logits and next-token
   quality with full BF16/FP32 accumulation semantics, not just float64
   local GEMV. Full checkpoint exact reconstruction remains unverified.
5. Do not merge or deploy into ALT / GPU Legion until independent QA and
   real inference acceptance.

The two existing GPT-OSS-20B GPU Legion serving processes on Mikoshi
were not modified to obtain these real ranges.
