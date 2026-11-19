# Neutron R04: Inline mathematical storage (experimental v6)

**Purpose:** increase full-archive density beyond R03 by removing the 45-byte
CHNK record and repeated per-chunk hashes for suitable whole-file mathematical
data. This is not a new physical storage medium or compression of arbitrary
random information. This is an **explicit experimental mode**; older modes and
formats remain available.

## Design

`create --neutron-dense` writes **DARC manifest v6**. For files of
64 KiB–8 MiB that can be exactly generated from a very short mathematical
bytecode or from a previously seen pure formula plus sparse replacements:

1. Add each unique *pure* mathematical root to the manifest
   `math_roots`: `{"s": rawSize, "c": base64(bytecode)}`.
2. Add `entry.inline_math = base64(uvarint(root_index) +
   uvarint(number_of_exact_patches) + sorted(delta,value) pairs)`.
3. Do **not** write a physical CHNK for this file. Its full SHA-256 is
   already present in the file entry.
4. Other files referring to the same root only store exact byte differences.

All compact expressions and roots are part of the existing zlib-compressed,
SHA-256-protected manifest. The decoder checks root bounds, file size, raw
size <=8 MiB, code size <=4096 B, valid deterministic opcodes, varints, patch
bounds, no competing CHNK/PACK links, and file SHA-256 on verification and
extraction. Data is never approximated.

If no eligible whole-file formula is found, use the R03 physical CHNK path
(raw / zlib / math / delta / mathematical root refs). This retains CDC for
arbitrary large data, streaming extraction and backwards support.

Maximum 8192 roots per archive, bounded cache of 32 MiB and 64 candidates
retained by the existing R03 matcher. The whole-file check is limited to
8 MiB to avoid unbounded memory growth. Memory allocation and throughput
are not yet production-hardened.

## Further density improvements in this branch

### Compact manifest table

Large sets of inline files can use an alternate binary `inline_table`
inside the same zlib-compressed MNF2 catalog, with original file paths,
sizes, modes, mtimes, full 32-byte SHA-256 digests and exact reference
bytecode. The writer compares the compressed standard and compact manifests
and keeps the smaller one. The reader reconstructs ordinary entries and
strictly checks varints, counts, bounds, SHA lengths and extra data.

### Quadratic integer-only bytecode (v6 only)

New opcode `A3 a b c count patches...` generates

```
byte[i] = (a + b*i + c*(i*(i-1)/2)) mod 256
```

using only exact integer arithmetic. The pure quadratic formula with no
patches takes **5 bytes** of bytecode, regardless of decoded length.
Dominant second/first differences are used to estimate `a,b,c` from
a bounded 4096-byte sample, and every source byte is checked before
accepting the packed result. Sparse deviations are encoded exactly.
The existing v4/v5 writers do **not** emit A3 and the new reader rejects
A3 in legacy-v4/v5 physical math chunks.

This is useful for integer sequences, deterministic synthetic data and some
structured numeric arrays. It does not help arbitrary/random data.

## Measured outcomes (Linux CI, 2026-11-19)

[Seven-corpus reproducible validation](https://github.com/Garyu-tt/DedupArc/actions/runs/44488215672)
passed `go test ./...`, `go vet ./...`, dedicated v6 mathematical
and archive tests, and SHA-256 roundtrip checks on all seven data sets.
The *source-tree* corpus is real text from the pinned DedupArc checkout;
the other six are explicitly synthetic.

| Corpus | Original | R03 v5 | R04 v6 | tar.zstd -19 |
| --- | ---: | ---: | ---: | ---: |
| Mathematical | 8,388,608 B | 1,546 B | **1,251 B** | 1,796 B |
| Shared math roots | 4,194,304 B | 1,660 B | **1,066 B** | **887 B** |
| Quadratic | 4,194,304 B | 16,575 B | **955 B** | 1,941 B |
| Versioned binaries | 41,943,040 B | 4,253,369 B | 4,253,369 B | 4,206,651 B |
| Random | 4,194,304 B | 4,196,174 B | 4,196,175 B | 4,194,933 B |
| Repeated text | 1,837,782 B | 24,820 B | 24,818 B | 22,699 B |
| Real DedupArc source tree | 236,809 B (63 files) | 68,396 B | 68,397 B | **59,925 B** |

The source-tree negative control is important: **R04 does not beat
Zstandard or XZ on ordinary source code** (XZ -9 was 58,264 B).
Quad data is deliberately constructed for polynomial coding, and repeated
patterns are not representative of arbitrary files.

R04 cuts the quadratic corpus by ~17.36x vs R03 and has full-archive
source-to-output ratio ~4391.94x there. The 8 MiB mathematical corpus
reaches ~6705x, but random data are not losslessly compressible
in general.

## Compatibility

| Mode | Writer manifest |
| --- | ---: |
| default / maximum | v2 |
| `--extreme` | v3 |
| `--neutron` | v4 |
| `--neutron-dag` | v5 |
| `--neutron-dense` | v6 |

The new reader can open v1–v6; older readers cannot open v6. The mode is
**never auto-selected**, merged or shipped without independent approval.

## Reproduce

```sh
go test ./...
go vet ./...
go build -o deduparc-core ./cmd/deduparc
./deduparc-core create --neutron-dense dense.darc ./data
./deduparc-core verify dense.darc
./deduparc-core extract dense.darc ./out
python3 scripts/benchmark_extreme.py --binary ./deduparc-core --workdir /tmp/dense-bench
```

Readiness gate: SHA-256 roundtrip tests, Go vet, Windows CI, comparison with
Zstd -19 and XZ -9 on the five deterministic corpora. Then test actual
repositories, VM images, model tensors and malformed/hostile archives.

**Important:** Do not claim that formula corpora represent arbitrary data.
Some corpora and modes will remain larger than Zstandard. Total archive bytes
are the only meaningful density measurement.
