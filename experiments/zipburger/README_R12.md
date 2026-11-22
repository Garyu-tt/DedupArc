# ZipBurger R12: real-expert coverage and phone feasibility

R12 adds a necessary data-quality gate, not a working smaller LLM.

## Per-expert activation coverage

Run after R11 calibration and holdout archives are complete:

    python -m experiments.zipburger.coverage_r12 \
      --cal-trace /private/r11-cal.npz \
      --val-trace /private/r11-val.npz \
      --cal-experts /private/r11-cal_expert_upgate.npz \
      --val-experts /private/r11-val_expert_upgate.npz \
      --output /private/r12_coverage.json

It validates the trace and archive prompt/checkpoint fingerprints,
compares actual router selections to recorded per-expert sample counts,
and reports coverage for all 24 x 32 = 768 layer/expert pairs.
Each considered expert needs at least 8 distinct calibration and 8
heldout observations to be eligible for **further isolated measurement**.
This is not enough to prove agentic or whole-layer equivalence. Anything
unseen or underrepresented remains exact R04. The results explicitly
set lossy_compression_authorized=false.

The reported figures describe only first-512-coordinate gate/up input
submatrices. No down projection capture or full 2880-feature equivalence
is claimed. Coverage is necessary, not sufficient.

## Absolute R08 bound on phone size

The public pinned GGUF is 12,109,566,624 bytes (12.110 decimal GB,
11.278 GiB). MXFP4 normally stores 32 weights as 16 mantissa bytes
plus 1 scale byte, or 17 bytes/block.

Plain R08 3-bit replaces that with 12 mantissa bytes plus 1 scale,
or 13 bytes/block. Even if **every byte** of the entire GGUF could
receive this improvement with no headers or uncompressible weights
(an impossible best case), its minimum whole-file size would be
around **9.260 decimal GB (8.624 GiB)**. This exceeds 8 GB storage
even before KV cache, other runtime buffers and Android OS memory.

Proof: maximum possible reduction is only 4/17 = 23.5294%.
To reach 8.0 decimal GB from 12.110 GB would require a reduction of
33.94% of original total model bytes even at zero additional RAM.
The **real** GPT-OSS has protected attention/routers and non-expert
weights, so the achievable R08-only upper bound is even worse.

For a decimal 10 GB model-storage target, 17.42% reduction is needed.
That is not categorically ruled out by the ideal 3-bit bound, but
quality tests on 18 real 20B fragments accepted 0/18 lossy replacements.
Model runtime memory can still make 10 GB infeasible.

Recalculate with an explicit eligible expert fraction and overhead:

    python -m experiments.zipburger.phone_budget_r12 \
      --target-gb 8 --expert-fraction 1.0 --extra-runtime-bytes 0

    python -m experiments.zipburger.phone_budget_r12 \
      --target-gb 10 --expert-fraction 0.7 \
      --extra-runtime-bytes 1073741824

An 8 GB / 8 GiB boundary is **not the same**. This calculator uses
decimal GB (1e9 bytes); always report your device's usable physical
RAM, not only marketing capacity.

## Next compression architecture

To approach the 8 GB device goal while maintaining agentic behavior,
R08 3-bit alone is structurally insufficient. Next experiment must
seek additional representation savings, such as:
- shared per-layer expert bases plus entropy-coded exceptions;
- learned higher-capacity compressed-domain computation;
- variable precision with exact fallback on sensitive experts;
- paged expert residency with explicit peak RSS/latency/energy tests.

These are hypotheses; **none** has yet passed whole-model quality or
agentic evaluations. Do not equate a best-case storage bound with real
phone inference. No changes to production, installed model or main.
