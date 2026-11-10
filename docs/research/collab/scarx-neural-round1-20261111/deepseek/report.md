# REPORT — deepseek — `pair_binding`: the gate was wrong twice, and the neuron already carries the signal

**Campaign:** `scarx-neural-round1-20261111`
**Agent:** deepseek
**Environment:** DeepSeek Harness (DSH) Web, Windows 11
**Model actually used:** `deepseek-ai/deepseek-v4.1-flash-1017`
**BASE_SHA:** `1735ad66ea9e7b416f5df15df99d9c13cea46b4e`
**Hypothesis commit:** `82f4a5fc131d3647cb23f970f917127b19449c1f`
**Protocol commit:** `85bf81ea251e02028968a1f85810b24908543405`
**Protocol SHA-256:** `465f67898c24ba0c6b9bbd8495262e104a781cd4670a00cc55f165b51b3e5cf3`
**Screen run:** `20261110T203340Z-deepseek-pair-binding-screen-v1-9da1fc`
**Self verdict:** **PIVOT** — my own primary claim C1 was falsified; a larger finding replaced it.

> **Independence disclosure (repeated deliberately):** this branch was authored by
> the same agent session as `research/antigravity`, at the user's direction because
> no independent deepseek session existed. Agreement between the two branches is one
> agent agreeing with itself and must not be counted as convergent evidence. Reviewers
> should weight this branch's **evidence** normally and its **independence** at zero.

---

## 1. What I set out to show, and what actually happened

| Claim | Registered as | Outcome |
|---|---|---|
| **C1** `pair_binding` is solvable by an additive point neuron + linear readout | order-1 probe ≤ 0.60 **and** order-2 probe ≥ 0.90 | **FALSIFIED** — see §3 |
| **C2** frozen hidden does not already clear the 65% gate | frozen < 0.65 | **HELD** (frozen 51.43%) |
| **C3** hidden learning is measurable on `pair_binding` | `score − frozen` CI95% lower > 0 | **FAILED** (−2.47pp [−2.73, −1.95]) |

`screen_pass = False`. Per the locked stop rule: no retune, no extra seeds, no
confirmation. The 8 fresh confirmation seeds were **never opened**.

## 2. The screen result — 108 trials, 36 scored

Direct accuracy, mean of 3 dev seeds, each arm at its dev-locked `readout_lr`:

| arm | `pair_binding` | `strict_binding` | `order` |
|---|---:|---:|---:|
| `score` (candidate) | 48.96 | 48.70 | **55.86** |
| `frozen` | 51.43 | 49.61 | 51.04 |
| `readout` | 48.57 | 49.09 | 51.69 |
| `shuffle` | 49.48 | 51.30 | 52.60 |

`pair_binding` contrasts:

| contrast | mean [95% CI] pp |
|---|---|
| `score − frozen` | **−2.47 [−2.73, −1.95]** |
| `score − readout` | +0.39 [−1.56, +1.56] |
| `score − shuffle` | −0.52 [−1.95, +0.39] |
| `gain: score − frozen` (probe) | +1.82 [−0.39, +5.86] |

Nothing reaches the 65% gate on either binding task from **any** arm. The learning
arm is *below* the frozen arm on `pair_binding`, which is a clear negative.

## 3. My primary claim was wrong, and the reason matters

I claimed `pair_binding` was solvable because its label is an **order-2** function
of the inputs. I conflated *"an order-2 feature exists in the input"* with
*"this architecture can express an order-2 feature"*.

Measured directly:

| basis | accuracy |
|---|---:|
| oracle order-2 **product** features | **1.0000** ← information is there |
| raw tape, linear (what a summing neuron can form) | 0.5283 |
| actual untrained ALIF features (what the readout sees) | 0.5217 |

An additive point neuron computes a **sum** of weighted inputs. `key × value` is a
**product**. Order-2 is *not* reachable by summing just because the function is
"order-2" in a polynomial sense. So `pair_binding` is not the valid order-2
replacement I promised — it is merely a *different* unreachable task.

**This is the same class of error as the discarded first generator draft**, where
independent value draws let a bag readout reach 0.70. I caught that one before
running anything. I did not catch this one until the screen came back at chance,
because I never probed the *product basis* against the *architecture's* basis
before registering the claim.

## 4. The finding that replaces it — and it is better

The screen result raised a sharper question than the one I started with: **is the
binding information reachable by this neuron at all, or is it destroyed upstream?**

Fixed-feature ridge probes on the same untrained ALIF, same 12 ticks, same data:

| substrate | `pair_binding` accuracy |
|---|---:|
| readout substrate (single leaky integral, `feature_decay=0.25`) | 0.5160 |
| full spike train, all ticks | 0.6740 |
| **spike co-occurrence across neuron pairs (Σₜ zᵢzⱼ)** | **0.9940** |
| spike co-occurrence including the time dimension | 0.9580 |

**The binding information is present in the spike train, almost perfectly
linearly decodable from pairwise neural co-occurrence (0.994), and the production
readout throws essentially all of it away (0.516).**

This is a strictly stronger and more useful statement than either of the two prior
branches reached:

- `grok` showed a slow *single-neuron* trace beats a fast one (0.484 → 0.9375) on `order`.
- `antigravity` showed a multi-timescale *readout bank* beats a single decay on `order`.
- **This branch shows the binding signal survives in the neuron's own spike
  co-occurrence and is destroyed by a readout that only integrates each neuron
  independently.**

The readout is the bottleneck, and specifically it is the **absence of any
quadratic/pairwise term** — not the number of timescales. That reframes all three
branches' results consistently: every readout tried so far (single decay, slow
trace, 5-decay bank) is a **first-order** function of spikes, and every one of them
therefore discards the pairwise structure that carries the conjunction.

## 5. Claim matrix

| Claim | Metric | Source | Scope | Limitation |
|---|---|---|---|---|
| `strict_binding` needs order-3; an additive neuron cannot express it | order-1 .47–.52, order-2 .48–.52, order-3 **1.00**, shuffled .45–.49 | P0 receipt | 12-tick, 8-channel, H=48 | proves the gate is invalid; does not prove *no* architecture could |
| `pair_binding` has no order-1 bag shortcut | order-1 ≤ 0.56 across 5 seeds | P0 receipt | same | — |
| `pair_binding` is **not** expressible by the architecture | raw-linear .528, ALIF .522 vs oracle .1.000 | §3 | same | **falsifies my own C1** |
| Frozen hidden does not clear 65% | frozen 51.43% | screen result | same | C2 held |
| Hidden learning produces no measurable gain | `score − frozen` −2.47pp [−2.73, −1.95] | screen result | same | C3 failed |
| **Binding information lives in spike co-occurrence** | co-occurrence probe **0.9940**, readout substrate 0.5160 | §4 probe | untrained ALIF, H=48 | **diagnostic only** — proves information exists, not that learning can use it |
| Memory / technical integrity | 14/14 P0, peak tree commit 700,858,368 B | P0 receipt | this prototype | — |

## 6. What is NOT claimed

No chat, no vision, no reasoning, no retention (not measured — the screen failed
before any retention arm), no lifelong learning, and **no claim that SCAR-X can
bind**. In particular, §4 is an *information-path diagnostic on frozen weights*: it
says the signal is present and linearly decodable by an external probe. It does
**not** say the network learns to use it. An oracle probe is not a skill.

## 7. Honest accounting of my own errors

1. **`pair_binding` v1 leaked** (order-1 reached 0.70–0.77) because values were
   drawn independently. Caught by P0 before any learning run; version discarded and
   documented in the protocol.
2. **Claim C1 was wrong** — order-2 *function* ≠ order-2 *expressive capacity* for
   an additive neuron. Caught only by the screen returning chance. This is the
   more serious mistake and it is the reason I registered C1 as falsifiable.
3. **No confirmation run**, because the screen failed. Not an omission; the
   registered stop rule.

Both errors are recorded because a reviewer needs to know that this branch's
positive finding (§4) came from a probe run *after* a failed hypothesis, not from
the hypothesis itself. That is a weaker provenance than a pre-registered
prediction and should be weighted accordingly.

## 8. Cross-review status

- **grok** and **antigravity** need to review this branch. It is pushed as a Draft PR.
- I (this session) reviewed grok's branch as `antigravity` and grok has not yet
  reviewed it. Because of the independence disclosure, **an `antigravity` review of
  this `deepseek` branch would be self-review** and I will not perform or sign one.

## 9. Suggested next step (not executed, not claimed)

If the §4 diagnostic is real, the testable follow-up is a **readout with an explicit
quadratic term** — e.g. a fixed random projection of `z_i·z_j` into the readout —
measured against the `frozen` control exactly as this suite does. That is a new
registered hypothesis for a new round. **It is not implemented here**, because
doing so after seeing §4 would be fitting the instrument to the result.
