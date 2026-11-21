# NeuroCrunch ZipBurger R03: exact MXFP4 neural carrier

**Research-only, CPU-first prototype.** This is NOT a compressed GPT-OSS-20B checkpoint nor a production inference engine. Do not merge/deploy on ALT or existing DedupArc archives without independent QA.

## New in R03

GPT-OSS MoE checkpoints have **32 E2M1 FP4 values per block**, stored in 16 bytes (low nibble first), and a separate **E8M0 uint8 exponent** for every block. The official OpenAI reference uses `2**(scale_byte - 127)` and E2M1 LUT values `0,.5,1,1.5,2,3,4,6` with signed equivalents. Sources: [OpenAI gpt-oss torch weights.py](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py), [Hugging Face MXFP4](https://huggingface.co/docs/transformers/main/quantization/mxfp4).

R03 treats weights as **numerical integer mantissas in half-units**: `[-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12]`, and uses R02's *exact shared-neuron basis + coefficient links + sparse integer residuals* for the mantissa graph `M = A @ B + R`. Original 8-bit scales are kept in a **separate exact graph**, never erroneously divided into packed nibble bytes. The `-0` nibble (binary `1000`) is preserved by a zero-sign sideband (constant +0, constant -0, or packed mixed bits), so exact **packed bit patterns**, not just floating values, round-trip. The sectioned `ZBM3` frame includes per-row SHA-256 verification of the reconstructed original packed blocks and scales.

`direct.dot(input)` computes per-block values from the graph **without materializing dense W**, using:

```
y[i] = sum_g ldexp(0.5, scale[i,g] - 127) *
          (sum_root A[i,root] * dot(B[root,32*g:32*g+32], x[g])
           + sum_col R[i,col] * x[col])
```

The R02 core's literal-row fallbacks are supported. When the total graph container is larger than the original packed MXFP4 bytes, `pack()` returns a bounded `ZBR3` **raw fallback**, which computes group by group from original nibbles and scales, without creating a full float-weight matrix. No generic compression library is invoked by ZipBurger; zlib and XZ below are only baselines.

`make_torch_module(carrier, rank=2)` adds differentiable trainable input/output residual adapters around **graph-mode** direct execution, not around dense decompressed weights. These adapters are initialized as identity and **have not been trained** to preserve real GPT-OSS output behavior. Raw fallback has no learnable graph-mode adapter in R03. Adapter parameters are extra storage, *not included in carrier compressed sizes*.

## Reproducible local benchmark (2026-11-22)

Every corpus is **synthetic**, 48 rows by 4 blocks of 32 values, with **3,264 original bytes** (3,072 FP4 nibble bytes + 192 scale bytes). Complete frame sizes are:

| Corpus | R03 frame | Mode | zlib9 | XZ6 | Actual graph compression? |
|---|---:|---|---:|---:|---|
| Repeated E2M1 and E8M0 | 1,280 | neural graph | **326** | 372 | Yes, 2.55x vs raw |
| Repeated with sparse exceptions | 1,348 | neural graph | **365** | 412 | Yes, 2.42x vs raw |
| Random E2M1 and scales | 3,308 | raw fallback | **3,275** | 3,324 | No |
| All positive zero | 825 | neural graph | **40** | 96 | Yes, 3.96x vs raw |

All four cases round-trip **every original code nibble, including sign of zero and scale bytes**. All direct float64 matvec outputs match a dense float64 reference (`rtol=1e-12`, `atol=1e-10`) on the tested samples. This does **not** imply bit-identical BF16 LLM inference; numerical summation order, scaling, and activation precision can differ. The R03 Python implementation is a reference correctness path, not a fused GPU kernel or a speedup demonstration; dense NumPy GEMV can be considerably faster.

Both linked frames, all base neurons, edges, residuals, zero signs, raw fallback overhead and SHA-256 are charged to the frame size. **Input/output neural adapter weights and Python/PyTorch runtime RAM are not included**; all-in storage/RAM must include them before any claim about a smaller runnable LLM.

## Run synthetic tests and benchmark

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_zipburger*.py' -v
OPENBLAS_NUM_THREADS=1 python -m experiments.zipburger.bench_r03
```

## Run on a local GPT-OSS-20B checkpoint shard

This **does not download** 13+ GB of weights or alter the checkpoint. It uses `safetensors.safe_open(...).get_slice()` for small read-only aligned ranges. Put the official `model.safetensors.index.json` and referenced shard files together in a local directory:

```bash
OPENBLAS_NUM_THREADS=1 python -m experiments.zipburger.bench_r03 \
  --model-dir /path/to/gpt-oss-20b --rows 32 --groups 4
```

Optionally provide `--tensor block.0.mlp.mlp1_weight.blocks`. The benchmark locates matching `.scales` by tensor name and verifies SHA-256, packed-byte reconstruction and direct group-scoped float matvec. It refuses pairs spread across separate shard files and unexpected shapes. `safetensors` is required for this optional operation; NumPy is the only mandatory R03 dependency.

**Reality check:** The present sandbox has no network access to download official tensor bytes; it validated the checkpoint slicer against a locally generated safetensors fixture. Real GPT-OSS-20B results remain unmeasured here. Earlier DedupArc issue #14 contains prior real-byte baseline comparisons, but R03 must be evaluated separately on those identical tensors.

## Outstanding release gates

1. Repeat the aligned benchmark on actual official MXFP4 tensor slices from multiple experts/layers and publish SHA-256, exact frame ratios against zlib/XZ/Zstd/WeightBasis/RMG, and 95th-percentile latency and peak RSS. Previous synthetic gains cannot be extrapolated to a 20B model.
2. Implement compiled CPU SIMD direct execution with group scale handling, and compare apples-to-apples with existing optimized MXFP4 GEMV (including token-level floating-point tolerance).
3. Introduce BF16 graph-mode handling, neuron reuse across experts, trained adapters and controlled model-quality evaluation; separately demonstrate bit-exact checkpoint restoration.
4. Fuzz corrupt metadata, stress memory and enforce no extra global raw model copy; run full repository CI and independent review.
5. Keep this research branch isolated from production DedupArc, ALT, or live model servers.
