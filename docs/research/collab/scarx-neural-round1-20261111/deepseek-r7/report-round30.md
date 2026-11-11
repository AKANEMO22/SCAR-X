# R7 ROUND 30 — REPORT: 14/15 seeds pass all three gates with a nonlinear readout

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no
**Seeds:** same 15 fresh seeds as rounds 24/26/27/28/29.

**Decision: DEV_PROMISING. The registered prediction is mostly confirmed: 14/15 seeds meet
meanF ≤ 3pp AND worstF ≤ 5pp AND acquisition ≥ 0.65 simultaneously. One seed misses by
0.12pp.**

---

## 1. The registered test

Round 29 registered, before running:

> **Prediction:** with a nonlinear readout, sequential training + interleaved replay reaches
> meanF ≤ 3pp AND worstF ≤ 5pp on **every** seed, acquisition ≥ 0.65.
> **Falsifier:** if the nonlinear head's sequential forgetting exceeds its own joint floor by
> more than ~3pp, capacity fixes co-representation but not sequential interference.

## 2. Result

| arm | meanF | n≤3 | worstF | n≤5 | acq | n≥.65 | **all three** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 5.94 | 2/15 | 10.23 | 0/15 | 0.8378 | 15/15 | **0/15** |
| **`mlp32`** | **1.02** | **14/15** | **1.64** | **14/15** | **0.9635** | **15/15** | **14/15** |
| `mlp32_noreplay` | 4.69 | 5/15 | 7.40 | 7/15 | 0.9642 | 15/15 | 5/15 |

**14 of 15 seeds meet all three gates at once.** The linear head meets zero.

**The falsifier does not fire.** Sequential forgetting exceeds the nonlinear head's own joint
floor by only **+0.66pp** (floor 0.36pp → total 1.02pp), so capacity fixed both
co-representation *and* sequential interference.

**Replay still matters at this capacity**: removing it takes meanF from 1.02 to 4.69pp
(−3.67pp effect, improving all 15 seeds). So this is not "a bigger head makes the mechanism
irrelevant" — the mechanism and the head form are both doing work.

## 3. The one failing seed, named

| seed | meanF | worstF | acq | verdict |
|---|---:|---:|---:|---|
| **240099** | **3.12** | **5.08** | 0.9844 | **FAIL** by 0.12pp / 0.08pp |

Its R matrix and per-task delta:
```
R = [[0.965, 0.422, 0.590],
     [0.953, 0.992, 0.996],
     [0.953, 0.941, 0.996]]
per-task delta = [1.17, 5.08, 0.00]
```
`parity3` is the task that degrades — 0.08pp over the worst-case gate.

I report it rather than the 14/15 headline alone, because "every seed" is the mission's
requirement and 14/15 is not it.

## 4. The audit I ran on my own result

I audited this before accepting it, because acquisition 0.9635 is high enough to be suspicious.

**4a. Is high acquisition inflating the result?** No — it is **conservative**. Forgetting is
`acq_after_learning − accuracy_at_end`, so a higher acquisition sets a **higher bar** for the
end state. The MLP is measured more harshly than the linear head, not less.

**4b. Are the two heads comparable?** **Not exactly** — linear acquisition is 0.8378, MLP is
0.9635, a 12.6pp difference. So the heads learn to different levels and their forgetting
numbers are not a pure "same task, different interference" comparison. **The MLP simply fits
better.** That is a real limitation of the head-form comparison and I state it rather than
using the gap as evidence.

**4c. Is the 14/15 a selection?** No — it is a count over all 15 seeds, and the failing seed is
named with its R matrix.

## 5. What this establishes, precisely

| claim | status |
|---|---|
| the linear floor (6.52pp) is **not** a property of SCAR-X's features | **CONFIRMED** |
| a nonlinear readout can co-represent the three tasks (floor 0.36pp) | **CONFIRMED** |
| sequential interference is also absent at this capacity (+0.66pp) | **CONFIRMED** |
| interleaved replay is still required (+3.67pp without it, 15/15 seeds) | **CONFIRMED** |
| **all three mission gates met on 14/15 seeds** | **CONFIRMED** |
| all three gates met on **every** seed | **NOT MET** — 240099 fails by 0.12pp |

## 6. What this does NOT establish — and it is the important part

**6a. `mlp32` was designated a DIAGNOSTIC, not a candidate.** The brief forbids expanding
capacity to mask a learning-rule failure. This round used the nonlinear head to *test whether
capacity was the constraint*, which is a legitimate diagnostic (the role BPTT plays). **It did
not become a candidate, no production code was changed, no production weight was produced, and
nothing here is proposed for Hermes.**

Promoting it to a candidate is **the scope decision** I have raised in rounds 5, 26, 27, 28 and
29, and it is now backed by the strongest evidence in this session. I will not take it alone.

**6b. The acquisition gate is met at 0.9635, but the mission's no-trade-off rule is about the
*mechanism's* cost on a fixed head.** Round 24 measured −4.33pp of acquisition cost for replay
on the linear head. This round does not measure that cost on the nonlinear head, because the
`mlp32_noreplay` arm has essentially the same acquisition (0.9642 vs 0.9635). **So on a
nonlinear head, interleaved replay appears to cost nothing in acquisition** — but I have not
tested that with the paired 18-cell design that round 22 taught me is necessary for a 1–2pp
effect. That is the next round's job.

**6c. Three synthetic tasks, H48, 12 ticks, spike-sum channel, frozen hidden.** Rounds 4–11
established the hidden learning rule contributes nothing to retention, so the hidden was held
frozen throughout. **The mission's first goal — hidden synapses learning a useful general
representation — remains unproven**, and none of these 30 rounds changed that.

## 7. Honest position after thirty rounds

**On the mission's second goal (skill accumulation without severe forgetting):**

| requirement | best achieved | status |
|---|---|---:|
| acquisition ≥ 0.65 | 0.9635, 15/15 | **met** |
| final ≥ 0.65 | 0.94–0.99 | **met** |
| mean forgetting ≤ 3pp | 1.02pp mean, 14/15 | **met on 14/15** |
| worst ≤ 5pp | 1.64pp mean, 14/15 | **met on 14/15** |
| every seed | **14/15** | **not met** |
| no acquisition trade-off | replay cost ≈ 0 on this head; **not yet properly tested** | open |

That is the closest this programme has come, and it required overturning my own round-27
"unreachable" conclusion.

**On the mission's first goal:** unproven. No round produced evidence that hidden plasticity
improves a representation, and rounds 10–11 showed the exact-derivative trace is possible yet
insufficient.

## 8. Declared exposures

| risk | exposure |
|---|---|
| **using a forbidden capacity increase** | §6a states the diagnostic role, that it is not a candidate, and that promotion needs authorisation |
| **reporting 14/15 as a pass** | §3 names the failing seed and its shortfall |
| **acquisition gap between heads** | §4b states the heads are not directly comparable at +12.6pp |
| **FP2** (readout better, hidden unchanged) | the hidden is frozen throughout and this is labelled a readout result |
| **FP3** (low forgetting because it learned nothing) | acquisition 0.9635, the *highest* in the programme |
| **FP10** (tests pass ⇒ product ready) | §6c lists what remains unproven |

## 9. Message for Gemini

The full trajectory on this, because the wrong turns are the useful part:

| round | belief | what killed it |
|---|---|---|
| 4 | forgetting = readout overwriting (+31.9pp) | survived |
| 16 | replay works (my label bug had hidden it) | survived |
| 27 | **gate unreachable, floor 6.41pp irreducible** | round 29 |
| 28 | floor stable across 8× budget | true but for the *linear* head only |
| 29 | floor is head-expressiveness (linear 6.52, mlp32 0.36) | survived |
| **30** | **14/15 seeds pass all three gates** | current |

The single most transferable measurement: **`solo − joint` for at least two head forms.** A
wider *linear* head was **worse** (8.79pp) than a narrow one (6.52pp) with 16× the parameters,
while a tanh head was 0.36pp — so the constraint was expressiveness, not capacity, and not the
features.

If CKCER's mechanism is evaluated on a linear readout, its reported floor may be self-imposed
the same way mine was.

## 10. Artifacts

`scripts/r7_r30_nonlinear_seq.py` · `artifacts/deepseek-r7/nonlinear-sequential.json` (per-seed
R matrices, both aggregations, three arms) · `local/audit_r30.py` (independent audit) ·
`report-round29.md` (the floor decomposition this tests)
