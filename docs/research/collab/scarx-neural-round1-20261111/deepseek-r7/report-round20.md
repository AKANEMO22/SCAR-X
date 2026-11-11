# R7 ROUND 20 — REPORT: replay is established over 18 paired cells; it also costs 2.6pp of acquisition

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: DEV_PROMISING on the mechanism. The effect is established. The gate is still
not met, and the acquisition cost is real and must be reported with the gain.**

---

## 1. Design

Round 19 showed within-seed order variance has sd 3–6pp, which made round 17's 3-seed,
1-order comparison insufficient. Round 20 uses the 18-cell design to remove that variance:
**replay and no-replay on the same (seed, order) cell**, so the difference is purely the
mechanism.

3 seeds × 6 orders = 18 paired cells. Identical data, initialisation, head algorithm, budget.

## 2. Result

| quantity | paired mean | sd | 95% CI | cells improved | excludes zero |
|---|---:|---:|---|---:|---|
| **mean forgetting** | **−19.66pp** | 9.53 | **[−24.40, −14.92]** | **18/18** | **YES** |
| **final accuracy** | **+10.51pp** | 7.98 | **[+6.54, +14.48]** | 17/18 | **YES** |
| acquisition | −2.60pp | 4.46 | [−4.81, −0.38] | **7/18** | marginal |

**Replay reduces mean forgetting by 19.66pp with a CI that excludes zero, and improves
every single one of the 18 cells.** Final accuracy rises 10.51pp. That is the strongest
result of this session, and unlike round 17 it is not a favourable draw.

## 3. The cost, stated as plainly as the gain

**Acquisition falls by 2.60pp, and only 7 of 18 cells improve.**

Per the mission's rule — "không trade-off acquisition lấy low forgetting" — a mechanism
that buys forgetting with acquisition must be reported as a trade, not a win. It is also
**False Progress 3's neighbourhood**: if acquisition kept falling, low forgetting would
eventually mean nothing was learned. Here it is small (2.60pp) and acquisition remains
0.83–0.86 against a 0.65 gate, so it is not FP3 — but the direction is the wrong one and I
will not bury it in a footnote.

The mechanism is doing what the round-4 decomposition predicted (readout overwriting is the
cause, and replay re-exposes the readout to old pairs), and the exchange rate it buys
forgetting at is ~7.5pp of forgetting per 1pp of acquisition. That ratio is the honest
description.

## 4. Gate position

| requirement | value | status |
|---|---:|---|
| acquisition ≥ 0.65 | 0.83–0.86 across cells | **✓** |
| final ≥ 0.65 | 0.79–0.85 | **✓** |
| mean forgetting ≤ 3pp | paired mean change −19.66pp, but **absolute** level remains | **✗** |
| worst ≤ 5pp | not shown by this test | **✗** |
| every seed | 18/18 improve, but absolute levels vary 5–17pp | **✗** |

**This round measures an effect size, not gate attainment.** A −19.66pp change from a
31.90pp baseline lands around 7.7pp, and round 19 established that the absolute level's
noise floor (sd 5.36pp) is above the 3pp gate. Both statements are true simultaneously:

1. replay is a real, large, replicated mechanism;
2. the mission's forgetting gate is not attainable **or measurable** at this sample size.

## 5. What this closes and what it opens

**Closed:** the question "does any mechanism reduce forgetting in SCAR-X?" — **yes**, with a
CI excluding zero over 18 cells in all 6 task orders. That is the first affirmative answer
in twenty rounds.

**Opened, and it is the interesting one:** the exchange rate. Replay pays ~2.6pp of
acquisition for ~19.7pp of forgetting. If a mechanism could reduce the acquisition cost
while keeping the retention gain, the ratio improves toward gate-compatible territory. That
is a *mechanism* question, not a tuning question, and it is where round 21 should go.

## 6. What I am not claiming

- Not that the mission's gates are met. They are not, and the noise-floor finding stands.
- Not that replay is a product solution. It is a buffer of 128 sequences per task in a
  three-task synthetic benchmark.
- Not that −19.66pp generalises outside this suite. It is measured on `sim_xor`, `parity3`,
  `order_short`, at H48, on the spike-sum channel.
- Not FP9: the buffer re-exposes the readout to `(feature, label)` pairs; the neurons
  compute every feature. Adaptive replay is on the brief's permitted list.

## 7. Declared exposures

| risk | exposure |
|---|---|
| **FP3** (low forgetting because it learned nothing) | acquisition −2.60pp, 7/18 cells; explicit in §3 with the exchange rate |
| **trade-off presented as a win** | headline names both the −19.66pp and the −2.60pp |
| **effect size confused with gate attainment** | §4 separates them explicitly |
| **replay read as FP9** | §6 states the buffer trains the readout, not the answer |
| **round 17's favourable draw** | superseded by this 18-cell design; §2 says so |

## 8. Message for Gemini

The experimental design is the transferable part:

**A 3-seed, 1-task-order retention comparison cannot resolve a few-pp effect on SCAR-X.**
Within-seed order variance is sd 3–6pp (round 19, 18 cells). The fix is cheap: run BOTH arms
on the same (seed, order) cells so the order variance cancels.

With that design:

| quantity | paired mean | 95% CI | cells improved |
|---|---:|---|---:|
| mean forgetting | **−19.66pp** | [−24.40, −14.92] | **18/18** |
| acquisition | −2.60pp | [−4.81, −0.38] | 7/18 |
| final accuracy | +10.51pp | [+6.54, +14.48] | 17/18 |

Replay is real. It also costs acquisition. If CKCER reports a retention gain, the same
paired-across-orders design would tell you whether it is a mechanism or a favourable seed
and order — and whether it pays for the gain in acquisition.

## 9. Artifacts

`scripts/r7_r20_paired.py` · `artifacts/deepseek-r7/replay-paired-18.json` ·
`report-round19.md` (the variance measurement that motivated this design) ·
`report-round16.md` (the label bug whose fix made replay work)
