# WeightBasis R01: shared denominators as reusable coefficients

**Status: isolated research codec; not a DARC writer/reader mode.** The default
DedupArc archive format, public API, flags and on-disk files are unchanged.
This branch adds a standalone Go lab (`internal/weightbasis`) and deterministic
benchmark (`cmd/weightbench`) that can later compete against RMG and other codecs.

## Exact transform

The shared-denominator observation motivates storing a common scale per block.
We do **not** assume that algebraic simplification itself reduces information content.
For each 256-byte group, choose an integer `w` from a fixed codebook. Represent:

```
x[i] = n[i]*w + r[i], exactly, where
n[i] = x[i]/w, r[i] = 0 if divisible by w;
n[i] = 0, r[i] = x[i] otherwise.
```

All quotients `n[i]` are bit-packed using the minimum bits needed for the selected
scale. Nonzero exact residuals use positive delta-coded indices and original bytes.
`w=0` means a raw group. There is no floating-point rounding, lossy operation,
approximate value match, or external model dependency.

The per-group `w` vector is recursively passed through the **same codec**, up
to three weighted levels. At each level the encoder compares raw, zlib-9, and
weighted candidates, and selects only the smallest *whole serialized frame*,
including all headers, coefficient frames, quotient bits and residuals.

The analogy with neural-network weights is reusable numerical coefficients,
not an assertion that these are learned or trainable model weights.

## Standalone WBS1 research format

All metadata uses canonical unsigned varints. Quotient bits are packed
least-significant-bit first. WBS1 is **not** a `.darc` CHNK method.

```text
magic[4] = "WBS1"
method[1] = 0 raw | 1 zlib | 2 weighted
original_size = uvarint (max 8 MiB)
payload:
  method 0: exact raw bytes
  method 1: zlib stream
  method 2:
    nested_weight_frame_size = uvarint
    nested_weight_frame = WBS1 frame with ceil(original_size/256) weights
    for each 256-byte (or shorter final) group:
      w = 0: raw group bytes
      w != 0:
        packed_quotients[ceil(group_size * quotient_bits/8)]
        patches = uvarint
        patches times: positive index delta uvarint; original byte
```

Decoder guards reject huge declared allocations, recursion deeper than 3
weighted levels, invalid coefficients, quotient overflow, unordered patches,
extra bytes, noncanonical varints and broken/truncated zlib streams.
Every selected frame is byte-exact upon decoding.

## Reproduce

```sh
go test -count=1 ./internal/weightbasis
go test -race ./internal/weightbasis
go vet ./internal/weightbasis ./cmd/weightbench
go run ./cmd/weightbench
go run ./cmd/weightbench -json
go run ./cmd/weightbench -file /path/to/real/checkpoint.bin
```

The deterministic set includes specially favorable multiples of 16,
sparse residuals, heterogeneous per-group multipliers, all zeros, and random
negative control. Benchmarks verify both exact bytes and SHA-256.
`raw_bytes`, `zlib_bytes` and `weighted_bytes` include framing/metadata.
Zero `weighted_bytes` means no weighted candidate was attempted.

## Adoption gate

- Benchmark real int8/float16/bfloat16 model *byte representations*, VM snapshots,
  and general files, against zlib-9, Zstd, XZ, RMG and DedupArc Extreme on the
  same source datasets and with the same accounting rules.
- Capture throughput, worst-case decode cost, peak memory and actual container size.
- Run aggressive malformed-frame fuzzing, Windows builds, and independent QA.
- Only then consider an explicit opt-in DARC version/method with old-reader
  compatibility tests. Do not silently deploy it to ALT, GENESIS, NeuroCMS,
  DRAM, DVRAM, or other users of DedupArc.
- WBS1 is not directly addressable CPU/GPU memory, and savings in archive bytes
  do not imply faster inference or runtime RAM reduction.

Random data do not become compressible by rearranging formulas; raw fallback
prevents expansion beyond WBS1 raw framing, but cannot promise to beat
Zstandard, XZ or a stronger codec on real datasets.
