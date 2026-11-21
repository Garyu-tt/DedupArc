# NeuroCrunch ZipBurger R01 (experimental)

**This is a research prototype, not a compressed GPT-OSS-20B checkpoint.**
Its purpose is to test an exact, executable *neural carrier*: data are transformed into shared root neurons, denominator-linked row activations, and sparse residual synaptic edges. A separate tiny trainable neural network sits at both ends, initialized to identity so it does not alter the measured core.

## Exact integer representation

For a signed INT8 matrix `W` with row `i`:

```
W[i,j] = (8 / denom[i]) * B[root[i],j] + R[i,j]

y[i] = (8 / denom[i]) * dot(B[root[i]],x) + sum_j R[i,j] * x[j]
```

`denom` is one of `{8,4,2,1}`. `B` stores shared INT8 basis vectors (the root neurons). `R` is a sparse exact INT16 correction matrix; a row can fall back to literal INT8 if linking costs more. This is lossless for the original **INT8 bit patterns**. Signed integer GEMV gives exactly matching results for integer inputs as long as the INT64 accumulation does not overflow. The output dot does not reconstruct dense `W`.

The serialized format `ZBG1` contains a header, dimensions, SHA256 fingerprint, all root payloads, per-row link/denominator data and sparse residuals. `ZBR1` is a raw fallback. The complete frame size is measured, including metadata, root payload and residuals. `load()` verifies SHA256 with one row of temporary memory, without creating a full dense matrix. `restore()` is an explicit export/debug function.

## Neural wrappers

`make_torch_module(carrier)` constructs a model:

```
input -> ResidualAdapter(nn.Linear -> tanh -> nn.Linear, rank=2)
      -> root neuron dot products + denominator links + sparse correction edges
      -> ResidualAdapter(nn.Linear -> tanh -> nn.Linear, rank=2) -> output
```

Both adapters initialize as the identity; gradients reach the adapter weights. Their parameters **must be counted** in total storage. The carrier's mathematical representation is the network's computational core. It is not a claim of new zero-cost information or general lossless neural compression.

## Test

From repo root, with `numpy` (and optional `torch` for neural adapter tests):

```bash
python -m unittest discover -s tests -p 'test_zipburger.py' -v
python -m experiments.zipburger.bench
```

## Measured synthetic benchmark, November 22, 2026

All inputs were *deterministic synthetic INT8* arrays of **67,584 bytes** (`264 x 256`). Results are not estimates of GPT-OSS-20B compression.

| Corpus | ZipBurger frame | zlib-9 | Root neurons | Linked rows | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Shared roots + 1-4 edits | **6,240** | 9,275 | 8 | 264 | lossless, exact GEMV |
| Shared roots + 64 edits | 67,632 | **40,093** | 0 | 0 | raw fallback |
| Random INT8 | 67,632 | 67,615 | 0 | 0 | raw fallback |
| All zeros | 1,888 | **87** | 1 | 264 | lossless, exact GEMV |

The favorable corpus is **10.83x smaller than raw** as a carrier file and **1.49x smaller than zlib**. However, the two trainable FP32 adapters take **8,320 additional bytes**, giving **14,560 bytes** if they are serialized along with the 6,240-byte carrier. The evaluated PyTorch registered core buffers occupy about **24,160 bytes** on CPU because roots and sparse coordinates are expanded, with another **8,320 bytes** of adapter parameters. Therefore the tested in-memory PyTorch representation totals about **32,480 bytes** of tensor buffers/parameters versus 67,584 raw INT8 weight bytes, excluding Python/PyTorch object overhead and workspaces. This is not yet a fused GPU/CPU efficient inference kernel. Speed performance is not claimed.

**Negative controls matter**: random/densely edited data do not become compressible simply by converting them into a graph. All-zero data remain much better suited to an entropy coder. Cross-model/dataset generalization is unproven.

## Next for GPT-OSS-20B

1. Work on actual MXFP4 nibbles and scale tensors, not byte-wise pseudo-weights, with BF16 tensor bit preservation. Verify original tensor bytes and hashes, no dropped metadata.
2. Search reusable roots across MoE experts and layers; learn or optimize shared *basis connections* while retaining exact residuals, with an explicit storage and inference-cost penalty.
3. Compare against zlib/Zstd/XZ, WeightBasis and RMG on the **same official shard ranges** and later a full checkpoint. Count adapter weights, indexes, residuals, packed graph, cache and runtime overhead.
4. Implement memory-bounded streaming/fused execution, identical tensor decode, compare FP accumulation order and real end-to-end token generation. Test throughput and first-token latency.
5. Keep it separate from existing production archive formats, ALT serving and live hardware. Do not merge/deploy until independent checks pass.

A lossless data transformation does **not** guarantee functionally equivalent LLM execution. The neural adapters are currently small identity-initialized networks, *not trained to replicate GPT-OSS-20B behavior*. This R01 is a proof of exact shared-root representation and direct integer computation, nothing more.
