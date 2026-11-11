# R7 ROUND 15 — REPORT: the threshold gain was a benchmark artefact (FP4), caught on myself

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: REJECT the threshold finding as a general improvement. It is task-specific
tuning, and it does not touch the retention problem.**

---

## 1. What I had, and why I did not trust it

Rounds 13–14 produced the largest single effect of this session: **+9.5pp on parity3** from
raising the threshold 0.50 → 1.00, and **+9.24pp** from `hta_scale` via the same mechanism.

Before reporting it as a finding I registered the test that could kill it:

> a threshold change is only "real" if it helps on **≥3 of 4 tasks** and hurts none by more
> than 3pp. If it helps 1–2 tasks, it is a benchmark artefact and I will say so.

## 2. Result — it fails the gate

| threshold | sim_xor | parity3 | order_short | delayed_short | helps | worst |
|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | +2.60pp | +0.78pp | −0.13pp | **−8.98pp** | 1/4 | −8.98pp |
| 1.00 | +2.73pp | **+9.51pp** | **−5.99pp** | −1.43pp | 2/4 | −5.99pp |
| 1.30 | +2.73pp | **+12.11pp** | **−8.33pp** | −5.86pp | 2/4 | −8.33pp |

**Threshold 1.30 buys +12.11pp on `parity3` and pays −8.33pp on `order_short` and
−5.86pp on `delayed_short`.** That is not an improvement; it is a trade across tasks.

This is **False Progress 4** in the mission's own list — "one benchmark reaches 100% while
a related task falls" — and I produced it myself by looking at a single task for two
rounds. The gate caught it because I wrote the gate down before running.

## 3. It does not help retention either

The second mission goal is unsolved at ~31pp mean forgetting. If a sparser code were more
separable, interference should drop. Same R-matrix protocol, honest metric (T−1 tasks), 3
seeds:

| threshold | acquisition | final | meanF honest | worstF |
|---:|---:|---:|---:|---:|
| **0.50 (production)** | 0.8928 | 0.6836 | **31.38pp** | 42.06pp |
| 0.70 | 0.9054 | 0.6641 | 36.20pp | 45.96pp |
| 1.00 | 0.8872 | 0.6567 | 34.57pp | 39.84pp |
| 1.30 | 0.9145 | 0.6532 | 39.19pp | 42.71pp |

**The production threshold is already the best of the four for retention**, and every
threshold fails the 3pp gate by an order of magnitude. Raising the threshold does **not**
reduce interference — mean forgetting gets *worse*.

So the one effect large enough to look like a finding is (a) task-specific and (b) useless
for the problem that actually needs solving.

## 4. What this closes

Round 14 left **sparsity** as the strongest remaining correlate (`corr(rate, accuracy) =
−0.755`). Round 15 shows that acting on that correlation by raising the threshold trades
one task against another and does nothing for retention.

**The sparsity correlation was also not causal** — it is a restatement of "each task has a
different optimal sparsity", which is a property of the benchmark suite, not a lever.

That was the last of the eight candidate causes in the brief:

| # | candidate | status | round |
|---|---|---|---|
| 1 | hidden lacks information | refuted — frozen linear 0.83–0.99 | 1, 12 |
| 2 | hidden has it, readout cannot fetch | **partly true** — channel loses 13–21pp | 1 |
| 3 | hidden can learn, signal is wrong | **refuted** — +0.945 alignment, zero gain | 10, 11 |
| 4 | eligibility loses causal influence | refuted once made exact | 10 |
| 5 | synapses learn then get overwritten | **true but readout-side** (−1.17pp hidden) | 4 |
| 6 | representation unstable across tasks | **true, unsolved** — 31pp | 4, 6, 7 |
| 7 | shortcut inflating accuracy | **FP4 found and rejected here** | 15 |
| 8 | benchmark/encoding creates a fake ceiling | **confirmed in round 1** (no headroom on the inherited suite) | 1 |

## 5. Honest position after fifteen rounds

**Established and replicated:**
- memory horizon 3 ticks at `leak = .65`; cross-tick XOR never linearly available (0.5527)
  while both bits read 1.0000 individually;
- the production decay readout discards **13–21pp** on same-tick conjunctions, hidden frozen;
- a joint linear head holds three retention tasks at 0.8620 — **capacity is not the blocker**;
- sequential training loses it: **31.4pp honest mean forgetting**, and hidden plasticity
  contributes **−1.17pp** — the head is what forgets;
- the production eligibility trace is at random-null alignment; the exact-derivative trace
  reaches **+0.945** and still produces **zero** representational gain;
- effective rank ~4.7 of 48 is real, but correlates only +0.169 with accuracy once
  separated from sparsity, and neither is a usable lever.

**Not established:**
- any mechanism that improves hidden representation or retention;
- retention within the mission's gates (best honest 31.38pp vs a 3pp gate).

## 6. What I am not going to do next

I have now tested six learning mechanisms and four hyperparameter directions. The result is
consistent: **the readout-side loss is where the measurable damage is, and the hidden
learning rule does not control it.**

The two remaining levers are the ones I have flagged twice and will not take unilaterally:

1. **A nonlinear or wider readout as a candidate** — round 5b measured 4.30pp mean /
   9.38pp worst forgetting with acquisition 0.8381. Still fails worstF, requires a scope
   decision against "no capacity expansion to mask a learning-rule failure".
2. **A readout codec change** — +13–21pp on same-tick conjunctions with the hidden frozen,
   which is a real, large, controlless effect and needs no learning at all.

Neither is a hidden-learning mechanism. Asking the mission owner is the honest move rather
than running a seventh mechanism of the same family.

## 7. Declared exposures

| risk | exposure |
|---|---|
| **FP4 — one benchmark up, related task down** | found on myself: parity3 +12.11pp, order_short −8.33pp. The pre-registered gate is what caught it |
| **sparsity reported as causal in round 14** | refuted here; acting on it trades tasks |
| **threshold could be read as a legitimate tuning result** | §2–3 show it fails both the cross-task gate and the retention gate |
| **no mechanism this round** | this round is a falsification, reported as such |

## 8. Message for Gemini

A test worth applying to your own best number, because it killed mine:

**Does your improvement survive on ≥3 of 4 related tasks, or only on the one you optimised?**

| threshold | parity3 | order_short | delayed_short |
|---:|---:|---:|---:|
| 1.30 versus 0.50 | **+12.11pp** | **−8.33pp** | −5.86pp |

If CKCER's gain is concentrated on one task family, it may be the same artefact. And the
second check: **does it move mean forgetting at all?** Mine went 31.38pp → 39.19pp, i.e.
*worse*, at every threshold above production.

## 9. Artifacts

`scripts/r7_r15_threshold_robust.py` ·
`artifacts/deepseek-r7/threshold-robustness.json` ·
`report-round14.md` (the retractions this round builds on)
