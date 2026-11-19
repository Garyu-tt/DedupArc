# Recursive Matrix Genome R01 (experimental only)

Issue: https://github.com/Garyu-tt/DedupArc/issues/7. Compression ladder: #4.

**Status:** Standalone, deterministic, lossless Go research codec in `internal/rmg`. It is **not enabled by `deduparc-core`**, not a production `.darc` v1/v2/v3 codec, and not a runtime memory mapper. Existing archive CLI, format and installed apps are unchanged.

## Model

For byte matrix A of width W and rows H, use the additive (modulo 256) predictor

`prediction[r,c] = base[c] + quantize(shift[r], step) (mod 256)`

`base` is the first row. The estimated row shift is the most frequent modular difference from that row; `step` candidates are 1/2/4/8/16/32. Deviations are stored as exact ordered `(index_delta, literal_byte)` corrections. Decoder evaluates the model and overwrites deviations with original bytes. Quantizing a rule therefore does not make the output lossy.

Both parameter arrays `base` and `shifts` are encoded with the **same RMG codec recursively**, with a maximum depth of 3. A candidate is selected only if its actual complete serialization, including child streams, headers, quantization metadata, residuals and length fields, costs fewer bytes than the current winner.

The competitors are raw bytes with RMG envelope and Go standard-library zlib level 9. No fixed gain is promised. Random input incurs only the small raw envelope overhead.

### Explicit limits and security

- `RMG1` standalone framing, NOT `.darc` reader-compatible.
- Maximum uncompressed bytes per stream or nested substream: 2 MiB. Maximum depth: 3. Width: 16/32/64/128/256/512/1024, only on compatible dimensions.
- Decoder rejects invalid magic/mode, lengths, malformed varints, excessive uncompressed size or depth, invalid steps, truncated or out-of-range residuals.
- No standalone cryptographic integrity tag. A trusted outer container must check a digest of the **original** bytes; future `.darc` integration requires its own versioned reader, checksum, and untrusted-input sandbox.
- CPU-first, no GPU/driver/kernel configuration, no production integration. Bounds alone are not full adversarial safety proof.

## Reproduce

`go test -count=1 ./internal/rmg`

`go test -race -count=1 ./internal/rmg`

`go vet ./internal/rmg ./cmd/rmgbench`

`go run ./cmd/rmgbench`

The benchmark emits JSON lines with fixed seed, input SHA-256, input size, zlib **payload-only** size, complete RMG1 stream size, winning mode, timing and bit-for-bit reconstruction result.

## Initial local results (Go 1.23.2, fixed-seed artificial corpus)

| Corpus | Original bytes | zlib level 9 payload | RMG1 total | Mode | Roundtrip |
|---|---:|---:|---:|---|---|
| structured_rows | 65,536 | 45,789 | **1,330** | matrix | byte-identical |
| nested_matrix_rules | 262,144 | 32,029 | **175** | matrix | byte-identical |
| similar_binary_versions | 65,536 | 3,893 | **2,339** | matrix | byte-identical |
| repetitive_text | 139,264 | **463** | 473 | zlib fallback | byte-identical |
| config_text | 119,808 | **417** | 427 | zlib fallback | byte-identical |
| random_uniform | 131,072 | 131,123 | 131,080 | raw fallback | byte-identical |

Nested matrix fixture deliberately contains a compact mathematical construction, an extreme best-case. It is **not** evidence of >1000× savings on real binaries, language models, random or previously compressed data. A separate test constructed a 262,144-byte example encoded to 180 bytes with **three** nested matrix nodes. Both RMG and zlib results are compressor- and machine-dependent; never convert this into a universal claim.

Comparisons against DedupArc Extreme, xz/zstd, 7-Zip and realistic files have NOT been run for this experiment. Next acceptance depends on equal-input benchmarks, immutable corpus hashes, independent QA and versioned decode.

## DRAM / DVRAM compatibility boundary

An RMG-compressed representation is **not** an ordinary MMU- or GPU-addressable array without reconstruction. Keep three mechanisms distinct:

1. **DedupArc archive:** encode RMG as a future optional, versioned .darc method; decompress only requested chunks, check checksums, preserve older format readers.
2. **DRAM:** exact dedup already materialized immutable pages for zero-decode reads; RMG pages may be hydrated into caches, counting materialized memory. Encoding is not free transparent ordinary CPU loads.
3. **DVRAM:** special compute-on-RMG kernels can potentially operate on compact descriptions, but arbitrary GPU kernels cannot; verify numerical equivalence, memory residency and extra instructions. CUDA VMM aliasing is an independent exact-data path.

Remain optional MAS providers: no hard dependency of Toolbox Core, Forge, DRAM, DVRAM or DedupArc itself. Future research: type-aware transforms, stronger factors, residual index for partial reads, bounded candidate search, real corpora and optional accelerator backends.
