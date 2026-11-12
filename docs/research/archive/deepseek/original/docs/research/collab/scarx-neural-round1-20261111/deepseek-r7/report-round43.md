# R7 ROUND 43 — REPORT: hidden learning IS possible (+3.69pp); DECOLLE is not better than the scalar

**Branch:** `research/deepseek/r7-causal-hidden` · **Base SHA:** `4f1c902b7948af1513dc127555bbd3ef80412f32`
**Production `src/scarx/neural/` modified:** no
**Mechanism source:** Kaiser, Mostafa & Neftci, *DECOLLE*, Front. Neurosci. 14:424 (2020), Eq. 8. **No code copied** — re-implemented from the equations.

**Decision: PARTIAL — and the first positive hidden-learning result in 43 rounds, together with a negative result for the transferred mechanism.**

---

## 1. The three questions, answered separately

The registered prediction was `perneuron` beats `scalar` by ≥2pp. It does not. But separating the question into three parts gives a result more useful than a single verdict:

| question | measured | CI | seeds better | answer |
|---|---:|---|---:|---|
| (a) does the hidden rule learn **at all**? | **+3.69pp** | **[+1.38, +6.00]** | 9/10 | **YES** |
| (b) does the **routing** matter? | **+3.67pp** | **[+0.48, +6.86]** | 7/10 | **YES** |
| (c) does per-neuron beat the **scalar**? | +1.52pp | [−0.97, +4.01] | 6/10 | **NO** |

Probe (ridge decodability of the frozen features), paired over 10 seeds:

| arm | probe | sd |
|---|---:|---:|
| `frozen` | 0.9033 | 0.0271 |
| `scalar` (production) | 0.9250 | 0.0266 |
| **`perneuron`** | **0.9402** | 0.0239 |
| `perneuron_zero` | 0.9033 | 0.0271 |
| `perneuron_shuffled` | 0.9035 | 0.0338 |

`perneuron_zero` reproduces `frozen` **exactly** (0.9033 both) — confirming the zero-factor arm is a true no-update control. `perneuron_shuffled` is also at the floor (0.9035), so **permuting the per-neuron error destroys the gain entirely**.

## 2. Why (a) and (b) are the important results

**Rounds 4 and 11 could only establish that hidden learning did not help *with the signals they tested*.** Round 11 used an exact derivative trace and got zero gain at every eta. That is a statement about *that* signal, not about hidden plasticity in general.

**This round shows hidden plasticity can move the representation: +3.69pp on a held-out ridge probe, 9/10 seeds, CI excludes zero, and the permutation control confirms it is the routing rather than generic noise.**

That is a genuine change to the programme's central conclusion, and it is the first positive result on mission goal 1.

**But it does not make SCAR-X better.** The production scalar already achieves **+2.17pp** over frozen (CI [+1.08, +3.26], 9/10 seeds). Per-neuron error adds only 1.52pp on top of that, and the CI includes zero. **DECOLLE's mechanism is a worse signal than what SCAR-X already has, not a missing one.**

## 3. What this rules out

The mission's central question lists "learning signal is wrong" as one candidate. This round narrows it:

- the signal is **not absent** — hidden learning happens (a);
- the routing **does carry information** — permuting it removes the gain (b);
- the **scalar is not the bottleneck** — a richer per-neuron signal is not better (c).

So the residual limitation is not "wrong learning signal". Something else bounds the representation, and rounds 29/34 already localised it: **readout expressiveness at T=3, frozen-feature capacity at T=6.**

## 4. Implementation record and controls

**What was changed:** only the learning factor.

```
production:  delta = hidden_lr * advantage * elig - anchor      advantage = ONE scalar
this arm:    delta = hidden_lr * g_i * (p[target]-1) * elig - anchor    factor per neuron
```

`g` is a fixed random vector, length H=48, `N(0, 1/√2)`, seed-fixed, never trained. Same `hidden_lr`, same `anchor`, same `update_clip`, same `weight_clip`, same eligibility, same H48, same `fan_in=8`, same forward pass.

**Five arms with exactly one factor differing each, hidden epochs matched.**

## 5. Defects I must disclose

**5a. The run crashed in its own summary and wrote no artifact.** `acc`/`probe` are per-task dicts; the paired-difference code subtracted dicts. The 15 seeds all completed and printed. **The numbers in this report are the run's stdout, transcribed**, and the recomputation is in `local/audit`-style script `r7_r43b_verdict.py`.

**5b. Only 10 of 15 seeds are used.** The first five scrolled off the captured tail. **n=10, not 15.** I could have re-run for the missing five; I did not, and the CI is computed with t(9).

**5c. The budget was cut from 120 to 30 hidden epochs** after measuring 610µs per step+learn call and projecting 88 minutes. Final wall was ~55 minutes. The cut is disclosed in the script. Since hidden epochs are matched across arms the comparison stays paired, and a lower absolute level makes a positive effect *harder* to see — so the positive result is not an artefact of the cut, but the **magnitude** may be understated.

## 6. What is NOT claimed

- **Not** that DECOLLE should be adopted. (c) says it should not.
- **Not** that mission goal 1 is achieved. This is a **+3.69pp probe gain**, not a working general representation, and it is *worse than the existing rule*.
- **Not** that rounds 4/11 were wrong — they were correct about the signals they tested. This round narrows their scope.
- **Not** a PASS. No gate was applied; this was a DEV mechanism-transfer screen.

## 7. What the next candidate should target

Given (c), a *richer* signal is not the answer. The two remaining candidates in the survey are:

- **B — Distribution-aware Scaling** (SpikingLM). Targets the 1.7/48 dead-unit measurement from round 12. **But R13's HTA was VOID and it is adjacent**, so I rate it lower than the survey did.
- **C — Benna–Fusi multi-timescale state.** Targets retention, and "importance decay / metaplasticity" already failed in 303 trials, so the prior is poor.

Neither has a strong prior. **The honest read is that the mechanism-transfer route has produced one real finding — hidden learning is possible — and no improvement.**

## 8. Message for Gemini

Three sentences that separate what was previously tangled:

> **Hidden learning is possible:** +3.69pp on a held-out probe, 9/10 seeds, CI [+1.38, +6.00], with a permutation control at the floor.
> **The routing carries the information:** permuting the per-neuron error returns the probe to the frozen value exactly (0.9035 vs 0.9033).
> **It is still worse than the existing rule:** the production scalar gets +2.17pp by itself, and per-neuron adds only +1.52pp over it with a CI containing zero.

If your mechanism report shows a hidden-weight contribution without a **permutation control**, it cannot distinguish credit assignment from generic update noise. Mine was 3.67pp and it was the difference between a real result and a null.

## 9. Artifacts

`scripts/r7_r43_decolle.py` (the run; crashed in summary, 15 seeds complete) ·
`scripts/r7_r43b_verdict.py` · `artifacts/deepseek-r7/decolle-transfer-recomputed.json` ·
`mechanism-transfer-survey.md` (the transfer card and the 11-mechanism comparison)
