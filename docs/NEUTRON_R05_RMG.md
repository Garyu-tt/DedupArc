# Neutron R05 — Recursive Matrix Genome (RMG), bounded research prototype

**Status: experimental, CPU-first, deterministic and lossless.** This is not a
release. `create --neutron-rmg` is the only writer for **DARC v7**.
Previous modes continue to write their old versions; new reader supports v1–v7.

## What is genuinely implemented

For byte-aligned, row-major matrices with a discoverable width
32, 64, 128, 256, 512 or 1024, the codec tries the integer-only model:

```
A[row, col] = (U[col] + V[row]) mod 256 + exact sparse edits
```

The variable `U` is the first row of the original matrix, and `V[row]`
is the modal byte difference for each subsequent row. **Both vectors
are independently encoded** with the existing R02/R04 math generators
(A0 constant, A1 affine, A2 periodic, A3 quadratic) if their bytecode is
shorter than their literal bytes. This is a two-level mathematical
description: the model's own parameters can be stored as formulas.

The encoder uses only bounded samples to reject unlikely shapes.
**Every byte is checked** and all mismatches stored as exact patches
before accepting. The decoded output is verified with DARC's existing
SHA-256 block and file checks.

A useful case is a block where the column shape is complicated but
the rows differ by simple row-dependent shifts. A wholly random matrix
is not expected to shrink.

## CHNK method 5 (v7 only)

The normal CHNK header supplies `rawSize` and SHA-256; payload:

```text
u8       wire version = 1
uvarint  matrix row width (power of two, 32..1024)
vector U:
    u8      encoding (0 = raw, 1 = existing math bytecode)
    uvarint encoded byte length
    bytes   encoded bytes
vector V:
    u8      encoding
    uvarint encoded byte length
    bytes   encoded bytes
uvarint  exact patch count
repeat   uvarint gap from previous patch end, exact byte value
```

`rawSize % width == 0`, `rows >= 16` and `V[0] == 0`. No user-provided
code execution, network reads, external dictionaries or runtime model
dependencies. Nested codecs are restricted to the preexisting finite
math-VM opcodes, not another RMG stream. Therefore there are **no nested
RMG graphs or unbounded decoding recursion**.

Bounds: max 8 MiB decoded chunk, no more than six width hypotheses,
a 16-row probe, and a minimum 17-byte improvement against the best
available packed CHNK candidate. CRC is not assumed; both full SHA-256
file verification and malformed input rejection are tested.

Fallback: all R04 modes (inline whole-file formulas, compact manifest,
raw, zlib, CDC, existing math bytecode, delta and shallow math references)
remain eligible. The new mathematical matrix method wins only by measured
packed byte count, with no guaranteed reduction on arbitrary input.

## Mathematical caveats

This is **not** a generic low-rank matrix solver, trained neural
compressor, or exact program search. Nor can quantization create new
information. Future versions may propose quantized multi-rank factorization,
but every quantization error must be carried as a bit-exact residual.
A general arbitrary-depth program grammar should be introduced only with
strict depth, CPU, RAM, byte-budget and deterministic-verification limits.

As always, measure entire archives including catalog data, headers,
references, model parameters, residuals and decoder format requirements.
Only a decoder already installed or included with the archive is usable:
external model/code dependencies must not be counted as free storage.

## Verified results (2026-11-19)

[Linux full-archive CI and SHA-256 evidence](https://github.com/Garyu-tt/DedupArc/actions/runs/44490558380)
ran `go test ./...`, `go vet ./...`, RMG exactness/corruption
tests, and verified every extracted tree by SHA-256.

| Corpus | Raw bytes | R04 v6 | R05 v7 | tar.zstd -19 |
| --- | ---: | ---: | ---: | ---: |
| Matrix predictor test | 2,097,152 | 1,120,926 B | **1,983 B** | 135,605 B |
| Math sequences | 8,388,608 | 1,251 B | 1,252 B | 1,779 B |
| Shared math roots | 4,194,304 | 1,064 B | 1,064 B | **913 B** |
| Quadratic | 4,194,304 | 951 B | 952 B | 1,928 B |
| Versioned binaries | 41,943,040 | 4,253,367 B | 4,253,369 B | **4,206,586 B** |
| Random | 4,194,304 | 4,196,175 B | 4,196,175 B | **4,194,933 B** |
| Text configs | 1,837,782 | 24,822 B | 24,824 B | **22,624 B** |
| Real repo source tree | 256,454 (67 files) | 74,462 B | 74,464 B | **65,082 B** |

One fixed-size matrix file unit test reports 524,288 B original,
280,617 B in R04, and **835 B in R05**, including all archive overhead.

**Interpretation:** The structured matrix corpus is deliberately an
extreme favorable case with a noisy 128-byte column vector and a
quadratic row-offset sequence. Its R05 archive is roughly 565x
smaller than R04 and ~68x smaller than Zstandard. This does not
generalize to training weights, images, random data or arbitrary
matrices. The source-tree control is worse than Zstandard and XZ:
R05 does not yet improve ordinary code compression.

## Reproducibility

```sh
go test ./...
go vet ./...
go build -o deduparc-core ./cmd/deduparc
./deduparc-core create --neutron-rmg rmg.darc input/
./deduparc-core verify rmg.darc
./deduparc-core extract rmg.darc restored/
python3 scripts/benchmark_extreme.py --binary ./deduparc-core --workdir /tmp/rmg-benchmark
```

Validation categories: versioned binaries, random negative control, text
configuration, mathematical formulas, shared formulas, quadratic sequences,
matrix-generated binary data, and an actual source-code tree.

**Research gate:** no merge/release until Go/Windows CI, corpus SHA-256
roundtrips, hostile archive fuzzing and realistic VM/tensor data benchmarks.
