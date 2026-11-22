# ZipBurger R08: layer-scoped 3-bit MoE, held-out activation quality gate

## Scope and non-claims

This is a **research-only**, CPU-first experiment layered on the R01-R07
ZipBurger draft. It does **not** compress an entire GPT-OSS-20B GGUF, run
20B inference, preserve arbitrary model logits, or establish agentic parity.
No services or production checkpoints are modified. The experiment branch
targets the existing research branch, not `main`.

The previous R04/R05 baseline stores an exact lossless MXFP4 representation
with direct GEMV. R08 changes **only MXFP4 expert-tile mantissas**, keeping
original E8M0 scales. It is intentionally lossy. No attention, router, KV
cache, embeddings or layer residual connections are compressed by this
experiment. The name guard accepts only:

`blk.<layer>.ffn_{down,gate,up}_exps.weight`

## Storage and kernel

Original MXFP4: 32 E2M1 values packed in 16 bytes plus one E8M0 scale byte,
**17 bytes/block**. R08: choose one 8-entry E2M1 palette per tile, store 32
3-bit indices in **12 bytes**, with the original scale in **one byte**.
Frame header: **48 bytes** including 8 palette bytes and SHA-256 of packed
payload. On a 48-row x 16-group tile:

| Representation | Bytes |
|---|---:|
| Original GGUF packed MXFP4 | 13,056 |
| R08 3-bit frame | 10,032 |
| Difference | **-3,024 bytes (-23.16%)** |

Numbers are structural tile storage, **not useful model compression after
quality gating**. Bigger compression needs evidence that accuracy survives.
C11 scalar `native_palette3_r08.c` computes directly from this packed
format without allocating a dense weight matrix. The native entry point
validates dimensions, but the Python `load()` SHA-256 verifier must run
once **before** passing untrusted frames to C.

## Training / evaluation

Palette selection uses only the **calibration** activation set to weight
mantissa errors by input energy. The **separate held-out** activation set
is scored against direct original MXFP4 matvec. Candidate accepted only if:

- held-out relative L2 error <= 0.04
- held-out output cosine >= 0.999

Otherwise R08 chooses **R04 exact lossless fallback**. These are research
thresholds, not established whole-model or agentic criteria.

Current `bench_layerwise_r08.py` substitutes Gaussian activation vectors.
This is **not** an observation of real internal GPT-OSS activations, so even
a passing tile cannot establish quality on natural-language inference.
Real adapter training/finetuning and teacher-layer activation recording are
separate experiments.

## Run

```bash
python -m unittest discover -s tests -p 'test_zipburger*py' -v
python -m experiments.zipburger.bench_layerwise_r08
# Optional, bounded immutable public HF GGUF ranged reads:
python -m experiments.zipburger.bench_layerwise_r08 --real --output /tmp/r08_20b.json
```

The `--real` path reuses the R04 pinned GPT-OSS-20B reader and samples
18 **real** expert tensor fragments from layers 0, 1, 2, never a full model.
It is pure read-only research. GitHub Actions runs isolated regression
tests and optional controlled public-range experiments.

## Honest agenticity release gates

1. Capture genuine held-out baseline layer activations, full MoE routing
   indices/weights and logits with the original 20B on fixed prompts;
   never use synthetic activations for final tuning.
2. Train/calibrate at individual expert and layer scope; audit accumulated
   deviation after layers 0..23, including residual paths. Protect
   attention, routers, embeddings and KV-cache until isolated parity tests.
3. Compare original vs compressed model on **multi-step** instruction
   following, tool selection, tool-result incorporation, error recovery,
   long-context memory and robust self-correction. Record success rates
   against a fixed seed/scoring protocol.
4. Measure full GGUF disk bytes, peak CPU RSS/RAM, tokens/s, wall energy,
   cold start, CPU load and cache behavior on the **same hardware**.
5. Reject/rollback any compression that misses the quality gates. A block
   ratio extrapolated to 20B is not a deployment memory number.

Neither the 8-10 GB phone target nor low-power 20B CPU operation has been
achieved by R08. This implementation is a measurement foothold, not proof.

## Independent real-20B fragment result (GitHub Actions)

The first public pinned GGUF run (2026-11-22), [workflow run 45063130028](https://github.com/Garyu-tt/DedupArc/actions/runs/45063130028), completed successfully:

| 18 real MXFP4 expert fragments, 48x16 blocks | Bytes |
|---|---:|
| Source packed weights | 235,008 |
| Hypothetical 3-bit candidates | 180,576 |
| Actual chosen storage after 18/18 strict gate failures | 226,386 |

All 18 held-out **Gaussian input** tests failed the 4% relative L2 /
0.999 cosine gate, reporting approximately 20.9%–25.8% relative output
L2 error. The fallback chose exact R04 across all 18 fragments. The
resulting 3.67% aggregate saving is from **R04 lossless scale bitpacking**,
not useful deployment of the lossy R08 palette. This confirms that
naive 3-bit lossy replacement is not a safe drop-in.

## R08B low-rank correction

`residual_adapter_r08.py` adds an **offline** activation-energy-weighted
low-rank SVD correction of the 3-bit quantization error. For rank k, it
stores FP16 left/right factors and applies `left @ (right @ input)`
after direct 3-bit GEMV, without constructing the original full dense
weights at inference. Its 8-byte research metadata and every FP16
factor byte count in stored size. No rank can pass without meeting both
the held-out accuracy gate **and** source-storage budget.

The bounded 20B benchmark now compares rank 1 and rank 2 correction
alongside no correction. This remains **tile training on Gaussian inputs**
rather than real teacher-layer activation traces or full-model training.
