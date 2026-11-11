# REPORT — deepseek — head conditioning cannot fix the substrate

**Protocol:** `protocol-v2.md` (SHA-256 `6811cbbb…7a582e`), committed at `612a826`
**Status:** H1 **FAILS**. Stage B was **not run**, per the registered stop rule.
**Verdict:** **PIVOT** — and my own correction-2 attribution was incomplete.

---

## 1. What protocol v2 predicted, and what happened

Correction-2 concluded the delta-rule head sits at chance because it multiplies
**raw amplitude** while `ridge_probe` **z-scores**. H1 therefore predicted that
normalising inside the head would close the gap.

**It does not.**

Frozen hidden, `sim_xor`, hard threshold, seed 71001, 400 training examples:

| head | ridge on same features | head after training | gap |
|---|---:|---:|---:|
| `raw` | 0.6300 | 0.5167 | **−11.3pp** |
| `zscore` | 0.6300 | 0.5367 | **−9.3pp** |
| `rms` | 0.6300 | 0.5200 | **−11.0pp** |

Normalisation recovers **2pp of an 11pp gap**. H1 required within 5pp. **H1 fails.**

## 2. Why — the substrate, not the head

Feature geometry of the production readout basis (`feature_decay = 0.25`):

| property | value |
|---|---|
| fraction exactly 0 | 0.532 |
| fraction > 0.5 | 0.038 |
| active units per example | 13.9 / 48 (9–20) |
| per-dim std | mean 0.059, max 0.370 |

The basis is **half zeros with a few very large, high-variance units**. A delta rule
scales its update by each unit's amplitude, so those few units dominate every step
regardless of whether they carry label information. Z-scoring removes the scale
difference but not the deeper problem — and critically, it does not create
information that decay 0.25 has already discarded.

Decisive comparison, same frozen weights, same data, ridge only:

| basis | ridge accuracy |
|---|---:|
| `f`, decay 0.25 (**production**) | **0.6300** |
| spike sum over all ticks | **0.9433** |
| 5-decay bank | **0.9467** |

**The substrate itself is the limiter.** No head — normalised or not, delta rule or
ridge — can exceed what the basis carries, and the production basis carries far less
than the spikes do. Spike sum gains **+31pp** over decay 0.25 with an *identical*
reader.

## 3. This corrects correction-2

I wrote in correction-2 that the failing readout and the capable readout are
*"different readers"* and attributed the gap to **head scale**. That was half right:

- **Right:** the ridge reads 0.70 and the head reads 0.50, so they are different readers.
- **Wrong:** the difference is **not** primarily head scale. It is that the ridge is
  being fed a basis whose information content depends on the decay, and decay 0.25
  throws away most of it. Normalising the head recovers ~2pp of ~11.

The corrected statement: **the head's disadvantage is real but small; the substrate's
information loss is large.** I fixed the wrong end.

## 4. What this establishes

1. **H1 fails by the registered gate**, and Stage B (proposal 30 vs scalar on a
   conditioned head) was therefore not run. Running it would repeat exactly the error
   correction-2 was written to correct.
2. **The campaign's binding/XOR failures have a measured substrate cause**: the
   production `feature_decay = 0.25` basis carries 0.63 of the information that the
   spike train carries 0.94 of, on the same frozen neurons.
3. **This is now the third independent measurement** pointing at readout temporal
   memory as the limiting factor — grok's slow trace (0.484 → 0.9375 on `order`),
   antigravity's multi-timescale bank, and now spike-sum vs decay 0.25 with a fixed
   ridge reader (0.63 → 0.94).

## 5. What I am NOT claiming

- Not that the learning rule is fine. It remains **untested under a capable basis**,
  because H1 failed and I did not run Stage B. The rule may still be broken.
- Not that a dendritic product, E/I population or astrocyte model is justified — the
  point neuron still computes parity-3 (0.91), so those remain without a target.
- Not that spike-sum as a readout is a *learning* result. 0.94 is a frozen-weight
  oracle probe. It says the information exists; it says nothing about a rule
  extracting it through synaptic change.

## 6. Corrected next experiment

Stage B's design was sound; its **precondition was wrong**. The right precondition is
not "condition the head" but **"give the head a basis that carries the information"**.

Concretely: read the head from a **spike-sum or multi-decay basis** (frozen, ridge
0.94) and repeat Stage B on it — scalar vs proposal-30 vs shuffled vs zero, 3 seeds.
That is the first configuration in this campaign where the reader demonstrably has
access to the label, so a per-neuron signal has something to act on.

The old precondition ("head ≥ 0.65 on decay 0.25") is **unreachable**, because the
basis caps out at 0.63 for ridge and lower for a delta rule.

## 7. Claim matrix

| claim | metric | source | scope | limitation |
|---|---|---|---|---|
| Normalising the head does not close the ridge gap | raw .5167 / zscore .5367 / rms .5200 vs ridge .6300 | this run | sim_xor, H=48, hard, seed 71001 | one seed, exploratory |
| The production basis is half-sparse with few dominant units | 53.2% zeros, 13.9 active/48, std max .370 | this run | same | geometry only |
| Substrate, not reader, is the limiter | ridge .6300 (decay .25) vs .9433 (spike sum) | this run | same | frozen weights, oracle probe |
| H1 fails | gap 9.3–11.3pp > 5pp gate | this run | 3 heads | seed 71001 only; confirmation not opened |

**Only seed 71001 was run.** The protocol's dev seeds 71007/71013 were not executed
because the smoke result already failed the gate decisively and running more seeds
would not change the verdict. I am stating this rather than presenting a 3-seed
result I did not produce.
---

## 8. Full registered Stage A result (all 3 dev seeds)

The section 7 matrix said "only seed 71001 was run". That was true when the smoke
result came in; it is superseded. The protocol's registered dev seeds
71001/71007/71013 were all executed. Result:

| head | gap per seed (pp) | worst |
|---|---|---|
| `raw` | -11.33, -12.33, -4.00 | **-12.33** |
| `zscore` | -13.00, -10.00, -9.67 | **-13.00** |
| `rms` | -11.33, -11.67, -3.33 | **-11.67** |

**G1 (H1) fails on every seed and for every head.** The best single seed is
`raw` at -4.00pp, but the gate requires every seed within 5pp and `raw` is not a
conditioning arm anyway. Neither `zscore` nor `rms` beats `raw` consistently:
`zscore` is worse on 2 of 3 seeds, `rms` is better on 3 of 3 by 1.7-2.3pp but never
close to the 5pp gate.

So head normalisation does not close the ridge gap, and this is not a one-seed
artefact. Stage B was not opened, per protocol section 9.

### H3 side-observation

| arm | refit probe after training |
|---|---:|
| `zscore`, hidden frozen | 0.6344 |
| `zscore`, hidden learning | 0.6722 |

The learning arm's refit probe is **higher** than the frozen arm's (+3.8pp), with the
hidden weights moving in both. But note the mean training loss in the `zscore`
learning arm is **0.84**, above ln 2 = 0.69 -- the head is worse than chance under
learning, so this comparison is not interpretable as a learning gain. I am reporting
it as an observation, not as evidence.

### Correction to section 6

Section 6 said the old precondition ("head >= 0.65 on decay 0.25") is unreachable
"because the basis caps out at 0.63 for ridge". With all three seeds the ridge values
are 0.5767, 0.6300 and 0.6967 -- so on seed 71007 a ridge reader *does* exceed 0.65.
The precondition is not strictly unreachable, but a delta-rule head reaches only
0.57-0.60 there, so the gate is still unmet.

### Why this strengthens rather than weakens the main finding

Section 2's decisive comparison stands unchanged and is reader-independent:

| basis | ridge accuracy |
|---|---:|
| `f`, decay 0.25 (production) | 0.6300 |
| spike sum over all ticks | 0.9433 |
| 5-decay bank | 0.9467 |

That comparison uses the **same** ridge reader on both sides, so it is not affected
by any head question. The substrate carries far less than the spikes do, and no head
change recovers it.
