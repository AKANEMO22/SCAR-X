# R7 ROUND 13 — REPORT: homeostatic threshold adaptation (VOID — premise was a metric error)

**Branch:** `research/deepseek/r7-causal-hidden`
**Written after the fact: round 13 ran scripts but its report was not committed at the
time, and round 14 subsequently voided its premise. Recorded here so the round is not
missing from the history.**

**Decision: VOID. The mechanism targeted a defect that did not exist.**

---

## 1. What round 13 tested

Round 12 had reported 25.7 of 48 "dead neurons" (never fire) and effective rank 4.60/48.
The mechanism was **homeostatic threshold adaptation (HTA)**:

```
r_i  <- (1-alpha) r_i + alpha * spike_i        per-neuron running rate
b_i  <- b_i + beta * (r_i - r_star)            threshold offset
threshold_i = threshold + b_i
```

A silent neuron has `r_i < r_star`, so `b_i` falls and it starts firing. Every signal is
local: its own spike, its own rate estimate, its own threshold. 1 float of state per
neuron, streaming, no BPTT.

Parameters: `alpha = 0.05`, `beta = 0.02`, `r_star = 0.10`, `b` clipped to ±0.4.

## 2. Result

| arm | effective rank | dead | rate | accuracy | rank gain | acc gain |
|---|---:|---:|---:|---:|---:|---:|
| `frozen` | 4.71 | 26.3 | 0.096 | 0.8320 | — | — |
| `hta` | **6.67** | 21.3 | 0.046 | 0.7539 | **+1.96** | **−7.81pp** |
| `hta_shuffled` | 5.50 | 23.3 | 0.113 | 0.6888 | +0.80 | −14.32pp |
| `hta_scale` | 5.48 | 41.0 | 0.013 | **0.9245** | +0.77 | **+9.24pp** |

**HTA raised effective rank by +1.96 — the largest rank change of any mechanism tested —
and accuracy fell by 7.81pp.** That single row already contradicted round 12's claim that
rank drives accuracy, and it is why round 13 rather than round 12 first cast doubt on it.

`hta_scale` was the informative control. It is **not** homeostasis: it applies `|drive|`,
so every threshold can only rise. Firing collapsed to rate 0.013, "dead" rose to 41 of 48,
and accuracy **improved** by 9.24pp.

## 3. Why the round is void

Round 14 found that all the "dead unit" counts in this round — including the 26.3 baseline
that motivated the mechanism — recorded only the **final tick**, not the spike train:

```python
fire.append(core.spikes.copy())     # one tick
```

The corrected count at production settings is **1.7 of 48**, not 26.3. **There were almost
no dead neurons to revive**, so HTA was solving a problem that did not exist, and its
+1.96 rank gain came from changing the dynamics globally rather than from reviving silent
units.

The `hta_scale` result is not void as an observation — it is consistent with round 14's
threshold sweep, where accuracy peaks at threshold 1.00 (0.9271) versus 0.8320 at the
production 0.50. But `hta_scale` is a threshold increase, not homeostasis, and it must not
be cited as evidence that homeostasis works.

## 4. What this round legitimately contributes

1. **The first direct counter-evidence to round 12's rank hypothesis**, recorded before
   round 14 existed: rank up +1.96, accuracy down 7.81pp.
2. **A clean negative on homeostatic threshold adaptation** as a mechanism: it changes the
   representation measurably and makes the task worse.
3. **A useful control design** — `hta_scale` (magnitude without feedback sign) isolated
   "raising thresholds" from "homeostasis", and the two gave opposite results. That
   separation is what pointed at sparsity rather than at silent neurons.

## 5. Declared exposures

| risk | exposure |
|---|---|
| **mechanism tested against a nonexistent defect** | premise came from round 12's buggy metric; voided here and in round 14 |
| **report committed after the fact** | stated in the header; scripts and `runs/20261111T124013Z-r7-r13-hta-v1` were committed in round 13's session |
| **`hta_scale` could be read as support for homeostasis** | §3 explicitly separates them |
| **+9.24pp from a threshold change could be read as a mechanism result** | it is a hyperparameter change, reported as diagnostic only; production threshold unchanged |

## 6. Artifacts

`scripts/r7_r13_hta.py` · `scripts/r7_r13b_sparsity.py` ·
`runs/20261111T124013Z-r7-r13-hta-v1/result.json` ·
`artifacts/deepseek-r7/sparsity-vs-rank.json` · `report-round14.md` (the correction)
