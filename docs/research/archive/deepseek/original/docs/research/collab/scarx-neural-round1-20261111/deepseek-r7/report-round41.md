# R7 ROUND 41 — REPORT: the registered prediction is REJECTED; replay allocation is exhausted

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: REJECT the round-40 mechanism reading. Favour-the-recent does not beat
favour-the-old. Replay weighting is now exhausted as a lever at T=6.**

---

## 1. The registered test

Round 40 registered, before this run:

> At matched task count, a task in a six-task run is 5.41pp more damaged than at T=3, so the
> damage comes from the **later** tasks trained before the measurement. **Prediction: favouring
> the recent tasks should reduce the earlier tasks' damage** — the inverse of round 38's ageing.
> **Falsifier:** if `recent_last` is not better than `old_first`, the interference is not
> directional.

Four weighting schemes, same tasks, same seeds, same budget, only the draw probability changes:

| arm | meanF | worstF | first-3 worst | acq | seeds passing |
|---|---:|---:|---:|---:|---:|
| **`old_first`** (round 38) | **3.39** | **8.54** | **7.71** | 0.9256 | 0/12 |
| `uniform` | 4.03 | 9.92 | 9.58 | 0.9216 | 0/12 |
| `recent_last` | 3.62 | 9.12 | 8.58 | 0.9247 | 0/12 |
| `both_ends` | 3.94 | 9.00 | 8.62 | 0.9269 | **1/12** |

**`recent_last` is −0.58pp WORSE than `old_first` on worstF and −0.88pp worse on first-3.**

**The registered prediction is rejected.** Round 40's reading — that the later tasks cause the
damage and protecting the vulnerable early ones is the wrong sign — is refuted.

## 2. Per-task table — the four schemes hardly differ

| task | old_first | uniform | recent_last | both_ends | spread |
|---|---:|---:|---:|---:|---:|
| `xnor_delay` | 7.21 | 8.50 | 8.08 | 8.54 | 1.33 |
| `maj3` | −9.88 | −9.46 | −9.21 | −9.33 | 0.67 |
| `and3` | 3.12 | 4.50 | 4.04 | 4.75 | 1.63 |
| `order_short` | −5.67 | −5.29 | −4.29 | −3.75 | 1.92 |
| `parity3` | 5.29 | 5.79 | 5.17 | 4.96 | 0.83 |
| `sim_xor` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**No task moves more than 1.9pp across four different replay allocations.** Round 38's −1.38pp
was therefore close to the ceiling of what weighting can achieve, not a step toward the gate.

**Replay allocation is exhausted.** Further tuning of decay or scheme will not reach `worstF ≤ 5`.

## 3. Two things are true at once, and round 39's verdict was half right

Cross-round consistency for the two extreme tasks:

| `xnor_delay` condition | delta |
|---|---:|
| r38 aged, **position 1** | **+7.21** |
| r41 old_first, **position 1** | **+7.21** |
| r39 xnor_last, **position 6** | **0.00** |
| r39 simx_first, **position 6** | **0.00** |

**`xnor_delay`'s own degradation is positional and reproduces exactly** (+7.21pp twice at
position 1, 0.00pp twice at position 6).

| `maj3` condition | delta |
|---|---:|
| r38 aged, position 2 | **−9.88** |
| r39 xnor_last, position 1 | **+1.17** |

**`maj3` flips sign with position**, so its "improvement" is not a task property.

**Most importantly, sign consistency across all four arms:**

```
xnor_delay  +8.08pp   sign consistent: True
maj3        -9.47pp   sign consistent: True
and3        +4.10pp   sign consistent: True
order_short -4.75pp   sign consistent: True
parity3     +5.30pp   sign consistent: True
sim_xor     +0.00pp   sign consistent: True
```

**Every task keeps its sign under every weighting.** The mechanism here is not allocational.

## 4. Where the T=6 gate actually stands

| | value | gate | status |
|---|---:|---:|---|
| meanF | **3.39pp** | 3 | 0.39 over |
| worstF | **8.54pp** | 5 | **3.54 over** |
| seeds passing all | **0/12** | all | **not met** |

Four replay schemes, three orders, two head forms, two budget settings — **`worstF` never goes
below 8.54pp.** I have no further lever inside the current mechanism and I am not going to invent
one by tuning against these same 12 seeds.

## 5. What the next round should do instead

The two levers that could still move `worstF` are both **out of the current scope**:

1. **A second head per task group** — an architectural change (capacity) and forbidden without
   authorisation.
2. **Deeper template separation** — changing the benchmark, which is not a mechanism result.

Inside the allowed scope the honest option is to **report the T=6 result as it is** and hand the
scope question over again with this round's evidence attached: a nonlinear readout took T=3 from
6.52pp to 0.36pp, and the same readout at T=6 is bounded near 8.5pp by a limit weighting cannot
move.

## 6. Declared exposures

| risk | exposure |
|---|---|
| **registered prediction rejected** | §1 states it and the direction of the miss |
| **round 40's mechanism reading refuted** | §1 says so explicitly, one round after proposing it |
| **round 39's verdict partially rehabilitated** | §3 states `xnor_delay` *is* positional for itself while the system is not |
| **"exhausted" overclaimed** | §5 names the two levers left and why each is out of scope |
| **FP3** | acquisition 0.9216–0.9269 |
| **no further tuning** | §4 commits to not tuning against these 12 seeds |

## 7. Message for Gemini

A result worth having in your notes because it constrains what replay tuning can buy:

**Four replay allocations, same everything else, per-task deltas move by at most 1.9pp.**

| task | best arm | worst arm | spread |
|---|---:|---:|---:|
| `xnor_delay` | 7.21 | 8.54 | **1.33** |
| `order_short` | −5.67 | −3.75 | 1.92 |
| `parity3` | 4.96 | 5.79 | 0.83 |

And every task keeps its **sign** under all four schemes. If your mechanism's reported gain comes
from a replay-allocation change, check the per-task vector: in my setting allocation reshuffles
±1.5pp while task identity fixes the sign.

The second observation is about signed per-task data: `maj3` reads **−9.88pp** at position 2 and
**+1.17pp** at position 1 in the same suite. A task that "improves" in one order can lose in
another, so a single-order per-task claim needs the order attached.

## 8. Artifacts

`scripts/r7_r41_invert.py` · `artifacts/deepseek-r7/t6-inverted-decay.json` (four arms × 12
seeds, per-task deltas) · `local/audit_r41.py` (the cross-round consistency check) ·
`report-round40.md` (the prediction this rejects) · `report-round39.md` (whose positional finding
this partially restores)
