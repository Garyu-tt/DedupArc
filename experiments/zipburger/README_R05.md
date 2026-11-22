# NeuroCrunch ZipBurger R05: compiled CPU direct GEMV for R04 MXFP4

**Research prototype, not a replacement for llama.cpp/GGML kernels.**

R05 adds the stateless C11 implementation
`experiments/zipburger/native_r04.c`, which consumes the exact ZBP4
scale-bitpacked R04 frame created and SHA-256 verified by Python.

It reads one E8M0 scale and 32 E2M1 nibble codes per physical block
directly from the compact frame, performs floating-point multiplication
and accumulation, and returns a row vector. **No full dequantized weight
matrix, heap allocation or GPU memory is needed inside the kernel.**
Scale lookup remains O(1) for the raw, delta and palette modes. The
kernel checks frame boundaries and numeric input sizes, but relies on
the R04 loader for the SHA-256 validation of original GGUF bits.

## C11 build

```sh
cc -O3 -fPIC -shared -std=c11 -Wall -Wextra -Werror \
  experiments/zipburger/native_r04.c \
  -o /tmp/libzipburger_r04.so -lm
```

The research Python test harness uses ctypes and a temporary compiled
library; the core can later be linked directly from an application.

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests \
  -p 'test_zipburger_native_r05.py' -v
```

The native code is tested against exact reconstructed R04 frames from
the three modes (literal, scale delta and palette), checks malformed
frames, and verifies float64 output against a dense reference. C11
`-Werror` is active. It is currently a sequential, scalar CPU loop,
not explicit AVX2/AVX-512 SIMD and not a patched GGML operator.

## Independently repeated real MXFP4 samples

GitHub Actions run:
`44966725567`, job `137037317570`, head commit
`74d4f30e49bafce0f00d26e50f028b855834e3ef`.

The benchmark used the same **18 real pinned GGUF MXFP4 slices** from
layers 0–2, each 48 rows × 16 groups (512 input weights per row).
The output matrix has 48 elements. Each native timing used 40 direct
calls, measured after compiling/loading the shared library and
packing the R04 frame, with allocations excluded from the timed loop.

| Metric | Value |
|---|---:|
| Real sample count | 18 |
| Exact C11 output vs float64 dense oracle | 18/18 |
| Native fused dot, median | **9.299 microseconds** |
| Native fused dot, range | 8.789–9.633 microseconds |
| Python R04 direct dot, mean | about 3.999 ms |
| Original GGUF sample total | 235,008 bytes |
| R04 frame total, including headers | **226,386 bytes** |
| zlib-9 baseline total | 226,407 bytes |

Full reproducible result with per-sample byte-range, SHA-256,
compression and timing data:
[benchmark_real_gguf_r05_native.json](benchmark_real_gguf_r05_native.json).

**Interpretation:** The direct compiled path is roughly 430× faster than
the Python reference on this particular benchmark. This is a
Python-to-C comparison, not a demonstrated 430× improvement over
optimized ggml MXFP4 kernels. Overall LLM throughput, token quality,
BF16/FP32 accumulation semantics, model peak RSS and full checkpoint
storage gains have not been measured. The data were read from the
immutable public Hugging Face GGUF using bounded HTTP 206 ranges, with
no changes to the working GPU Legion on Mikoshi.

## Next acceptance gates

1. Benchmark *identical* tiles against compiled GGML's own MXFP4
   dot/GEMV implementation using equal CPU thread counts and cache
   conditions; report cycles per weight, L1/L2 cache misses, resident
   bytes and actual throughput.
2. Add vectorized AVX2/SSE2 code paths with a portable scalar fallback,
   preserving numerical tolerance and bit-for-bit frame reconstruction.
3. Inspect more layers and expert ranges, compare zlib/Zstd/XZ and
   report all-in storage including decoder binary, indexing metadata,
   optional neural adapters and format overhead.
4. Evaluate end-to-end gpt-oss logits and full checkpoint roundtrips
   before proposing any live model swap. Preserve ALT rollback.
5. Keep PR #15 Draft for independent QA. Do not integrate this C
   proof-of-concept into live GPU Legion services yet.
