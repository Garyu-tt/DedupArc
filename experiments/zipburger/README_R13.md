# ZipBurger R13: is inter-expert sharing worth the engineering cost?

R08 showed 0/18 quality passes for naive 3-bit 20B expert compression.
R12 showed even ideal R08 3-bit alone cannot reach an 8 GB phone
target. R13 investigates a **different** hypothesis: are separate
MoE experts redundant enough to benefit from shared reference bases
and exact compressed deltas?

This is a **research-only** survey, not a runnable codec or a claim
of model functionality. It does not replace R04/R08 or deploy code
into production.

## Public, bounded, reproducible study

The R13 job samples the **same 48 x 512 MXFP4 weight submatrix from
each expert**, unlike R08 which sampled row 4096 in expert #1
(its internal row 1216). Identical within-expert row positions are
necessary for a fair cross-expert comparison.

For each of the 9 pinned public layer 0..2 up/gate/down expert
tensors, compare four separate experts with row_start positions
0, 2880, 5760 and 8640. Each slice is 48 rows x 16 MXFP4 groups,
13,056 bytes, retrieved by the existing strict pinned HTTPS range
reader. At most 36 HTTP range requests of less than 120 KB each.

For each tensor, compute:

- Fraction of exact same 17-byte MXFP4 blocks at aligned positions.
- Cosine similarity of *decoded* paired expert weight samples.
- Fraction of total sampled expert weight energy explained by
  best shared rank-1 and rank-2 **expert-space** factors.
- zlib9 compressed bytes separately, concatenated across experts,
  and exact XOR-to-expert-0 deltas plus the reference expert.
- Best tested lossless size, including all reference/delta bytes.

The results do NOT prove that learned low-rank approximations will
preserve inference. High rank-1 energy in a tiny slice is only a
hypothesis for later full-projection tests. Low similarity or no
lossless delta gain is evidence against simplistic shared-expert
compression for the sampled region.

## Commands

Synthetic offline fixtures (no weights/network):

    python -m unittest discover -s tests -p 'test_zipburger_cross_expert_r13.py' -v
    python -m experiments.zipburger.cross_expert_r13 --experts 4

Actual bounded public 20B weight ranges:

    python -m experiments.zipburger.cross_expert_r13 --real --experts 4 > r13_real.json

An independent GitHub Actions workflow runs this public-range experiment
when the R13 research source or tests change and uploads the JSON report.

## Relationship to the 8 GB phone goal

The original GGUF is ~12.110 decimal GB. To fit 8 GB as storage
requires at least 33.94% reduction. For actual **RAM** the required
reduction is higher due to caches and runtime buffers.

Any R13-based representation must eventually pass all of these:
1. Full-size bytes including dictionary, decoding and adapter costs.
2. Same-input output logits and accumulated 24-layer hidden drift.
3. Independent top-4 routing and expert-weight parity.
4. Multi-step planning, tools and error recovery success.
5. Peak RSS, CPU tokens/sec and energy on the target device.

Without a real teacher trace and model-level candidate runtime,
the experiment is not eligible for agentic/production acceptance.
Keep ZipBurger PR #17 Draft until those gates are satisfied.

## Measured public 20B results (2026-11-22)

The full bounded [GitHub Actions R13 run 45069174115](https://github.com/Garyu-tt/DedupArc/actions/runs/45069174115)
completed successfully and uploaded JSON for **36 real GGUF MXFP4
expert tiles**, four aligned experts for each of nine tensors
from layers 0..2, 470,016 source bytes overall.

| Measurement across nine tensors | Result |
|---|---:|
| Aligned identical full 17-byte MXFP4 blocks | 0.00% |
| Mean decoded pairwise expert cosine | 0.00109665 |
| Mean shared expert rank-1 energy fraction | 31.28% |
| Mean rank-2 energy fraction | 57.95% |
| Best tested sample-only lossless compression | 452,940 / 470,016 bytes |
| Best sample-only saving | 3.63% |
| Independent zlib9 for each expert | 452,994 bytes |
| Combined zlib9 across experts | 453,134 bytes |
| XOR-to-first-expert + zlib9 | 462,160 bytes |

**Interpretation:** a raw shared expert dictionary/XOR-delta is NOT
promising on these sampled real weights. Exact MXFP4 block deduplication
is zero, pairwise cosines are near zero, and XOR is worse than independent
zlib. The rank-1 energy reflects the best possible *approximating*
factor across four samples and is not a quality-passing trained decoder.

These findings do NOT rule out nonlinear function-level distillation,
specialized per-layer learned codecs or different unexamined experts.
They do rule out claiming that the current simple expert dedup scheme
will reach a phone-ready 20B. Do not extrapolate 3.63% from a tiny
sampling window to the complete GGUF.

The next meaningful experiment must test a qualitatively different
compression mechanism with real activation traces (R11/R12) and
end-to-end agentic quality gating; simply trying more XOR dictionaries
would not solve the established bottleneck.
