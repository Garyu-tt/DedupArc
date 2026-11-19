# Neutron R03 — shallow mathematical reference graph (experimental)

**Status:** opt-in research. Create using `deduparc create --neutron-dag file.darc input...`.
This writes **DARC v5**. The earlier `--neutron` keeps writing v4;
`--extreme` remains v3; default mode remains v2. Do not release or merge
without independent real-world benchmarks, fuzzing and compatibility checks.

## Rationale

R02 encodes each chunk as a short deterministic math formula with exact sparse
corrections. R03 adds a compact **backward reference to an earlier mathematical
root chunk** and exact replacements, allowing distinct files to share the same
computed pattern. This is a **shallow, acyclic reference graph**, not a general
symbolic AST optimizer, not a learned neural codec, and not a guarantee of
compression on arbitrary data.

## V5 chunk record method=4 (math-ref)

Standard CHNK header remains unchanged; payload bytes:

```text
uvarint  root physical CHNK ordinal (0-based, archive order)
uvarint  number of exact-byte patches
repeat   uvarint gap from previous patch end, 1 byte exact replacement
```

The initial previous-patch-end is zero; after writing at position i, its new
value becomes i+1. An index is valid only when:

- It points **backward** to a previously scanned CHNK record.
- Root method is exactly `methodMath=3`, not another reference or delta.
- Root uncompressed size equals the referencing chunk.
- The entire patch stream parses without overflow, missing/trailing bytes
  or indexes outside the declared uncompressed size.
- The final output passes the existing SHA-256 CHNK integrity check.

The unpacker refuses reference cycles by construction; depth is one. Archive
format v1–v4 reading is retained; a prior binary will reject version 5.

## Encoder bounds

- Maximum decoded math chunk: 32 MiB, with existing CHNK limits.
- Root RAM cache up to **32 MiB** and **64 roots**, retaining recent math blocks.
- At most **16** equal-size roots considered per candidate.
- Capped **256-position sample** for candidate rejection; the accepted
  candidate is compared across **all bytes**.
- Only select the complete reference payload if it is at least 17 bytes
  smaller than the currently best raw/zlib/math/delta encoding.
- CPU-first and deterministic; no GPU, external dictionary or hidden state.
- Original math generator and delta encoders remain available as fallbacks.
- Writer never promotes a method-4 reference to a root or a delta base.

These are implementation ceilings, not guaranteed performance or a theorem
about optimal compression. Tiny improvements may be outweighed by format or
manifest metadata; measure total archive size.

## Test commands

```sh
go test ./...
go vet ./...
go test ./internal/archive -run 'TestNeutronDAG|TestMathReference' -v
go build -o deduparc-core ./cmd/deduparc
deduparc-core create --neutron-dag sample.darc ./samples
deduparc-core verify sample.darc
deduparc-core extract sample.darc ./restored
```

R03 is intentionally not used for all `--neutron` archives. Readiness
requires corrupt-archive fuzzing and measurements on code trees, VM images,
neural weights and real filesystem snapshots, including wall time and peak RAM.

## Further directions

1. Move from chunk-root references to an explicit *content-addressed formula
   dictionary* with independently authenticated entries and canonical IDs.
2. Explore reversible filters (byte-shuffle, bit-plane, integer predictors)
   and patch-cost models, with an unbiased negative control.
3. Consider periodic formulas shared across different raw lengths; enforce
   bounded decoding and explicit dependency budgets.
4. Compare against Zstd/LZMA and R02 on real corpora, not only formula data.
