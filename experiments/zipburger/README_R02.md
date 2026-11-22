# NeuroCrunch ZipBurger R02: two-root exact neural carrier

**Status: exploratory Python proof of concept, not a compressed or executable GPT-OSS-20B checkpoint.** Production DedupArc archive formats and ALT serving are untouched.

## What is new

R01 uses at most one denominator-linked basis vector per row. R02 permits **up to two different shared base neurons per row**, plus an exact sparse residual matrix:

```
W[i, j] = A[i, u] * B[u, j] + A[i, v] * B[v, j] + R[i, j]
y       = A @ (B @ x) + R @ x
```

- `B` is a dictionary of *INT8 base-neuron vectors*. Their bytes really live in the representation.
- `A` has at most two nonzero signed coefficients (`±1`, `±2`, `±4`) per output row. Links are stored explicitly.
- `R` is the exact signed INT16 correction matrix, stored as sorted sparse per-row connections.
- Rows that are not worth factoring stay literal INT8. The complete stream also has an optional raw fallback; **it does not explode random bytes into a giant COO matrix** during loading.
- `read_cell(i, j)` retrieves an exact bit-pattern without restoring full `W`. `dot(x)` computes exact integer GEMV from root activations and edges without constructing `W`. `verify()` hashes streamed rows. `restore()` is explicit debugging/export only.
- `pack_bytes` can embed a **post-codec byte stream** as a signed-INT8 arithmetic matrix while preserving every original byte and its true length. This does **not** mean the stream is automatically compressible; a modern entropy-coded stream usually has high entropy.
- `make_torch_module` implements trainable rank-2 input and output residual neural adapters around the frozen, directly executed carrier. The adapters are identity-initialized, not trained to replace the language understanding of GPT-OSS-20B. The core never registers a complete decoded `W` tensor.

## Deterministic benchmark: 2026-11-22

All 2D matrices in the table have 208 rows × 256 columns and 53,248 INT8 source bytes. A separate control uses already-zlib-coded bytes. Size values include frame headers, root table, links and exact residuals, but **exclude the neural adapters**.

| Corpus | R01 | R02 | zlib-9 | Comment |
| --- | ---: | ---: | ---: | --- |
| Two-root structured synthetic | 7,320 | **5,454** | **1,812** | 8 shared roots / 414 links / 319 residuals; direct GEMV exact |
| Uniform random INT8 | 53,296 | 53,304 | **53,274** | Raw fallback; no compression |
| Zero-valued INT8 | 1,552 | 1,768 | **74** | Entropy coder substantially smaller |
| MXFP4-byte-*like* synthetic control | 53,296 | 53,304 | **23,858** | NOT actual GPT-OSS MXFP4 weights |
| Already zlib-coded structured stream (1,812 B) | n/a | 2,104 | 1,685 | Bit-exact, no new graph reuse |

R02 carrier improves on R01 by **25.5%** on the specifically favorable two-root matrix. It is **not** a better general-purpose compressor than zlib. These tests are not evidence of reducing an actual GPT-OSS-20B checkpoint.

On the favorable matrix, the rank-2 PyTorch input/output adapter parameters add 7,424 FP32 bytes. Thus the serialized carrier plus adapters takes at least **12,878 B** (not counting adapter container metadata), still not the minimum entropy-compressed representation. Torch's counted registered buffers plus parameters take **20,844 B**, excluding PyTorch/Python allocator overhead and temporary activations. A quick 500-iteration single-thread CPU NumPy microbenchmark measured ~0.050 ms for `carrier.dot` versus ~0.049 ms for dense INT8-to-INT64 multiply on this very small matrix. The times are illustrative, not stable throughput numbers; **no runtime speedup is established**.

All 10 R01+R02 tests pass with `OPENBLAS_NUM_THREADS=1` and Python/NumPy/PyTorch. The R02 tests include corrupt frames, post-codec exact bytes, random baseline, indexed read, direct GEMV, and backprop through both neural adapters.

## How to run

```bash
python -m unittest discover -s tests -p 'test_zipburger*.py' -v
OPENBLAS_NUM_THREADS=1 python -m experiments.zipburger.bench_r02
```

The full example JSON is in `experiments/zipburger/benchmark_r02.json`.

## Constraints and next milestone

1. Treat actual MXFP4 nibbles + per-block E8M0 scales, BF16 tensor bits, and packed checkpoint metadata as **different types**. Arbitrary signed-INT8 interpretation is a reversible *byte encoding* for data storage, not legitimate GPT-OSS-20B floating-point inference.
2. Train a shared learnable basis selector and sparse correction search on official GPT-OSS-20B tensor samples, while charging for **all** roots, link indexes, residuals, input/output adapter weights and workspace. Measure against equally complete zlib, XZ, Zstd, WeightBasis and RMG baselines.
3. Keep exact lossless reconstruction separate from model functional equivalence. Numerical floating-point operation order matters. A meaningful LLM execution benchmark requires correct MXFP4/BF16 math and a real inference backend.
4. For high-entropy post-codec data, neural bases have no guaranteed compression advantage. The information cannot be stored in the architecture for free; every learned parameter and residual must be counted.
5. Avoid production merge/deployment until independent QA, fuzz tests, memory profiling, actual checkpoint sampling and correct token-generation comparisons are complete.
