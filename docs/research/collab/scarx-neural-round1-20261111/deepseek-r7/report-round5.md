# R7 ROUND 5 — REPORT: orthogonal head updates help, but not nearly enough

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: REVISE — the mechanism moves the metric in the right direction with the
right control separation, and still fails the gate by a factor of ~5.7.**

---

## 1. Result

3 dev seeds, three cascaded tasks, spike-sum channel, hidden FROZEN (round 4b showed
hidden plasticity is irrelevant to forgetting, so it is held out to isolate the
readout mechanism).

| arm | acq | final | meanF | worstF | gate |
|---|---:|---:|---:|---:|---|
| `frozen_shared` (baseline) | 0.7513 | 0.5417 | **20.96pp** | 52.34pp | fail |
| **`ogh`** | 0.7569 | 0.5868 | **17.01pp** | 44.14pp | fail |
| `ogh_shuffled` (control) | 0.6884 | 0.4839 | 20.44pp | 55.08pp | fail |

Gate: meanF ≤ **3pp**. Best achieved: **17.01pp**.

## 2. What is genuinely established

1. **The mechanism is real, not a degrees-of-freedom artefact.** OGH beats the
   shuffled-basis control by **3.43pp** on mean forgetting (17.01 vs 20.44) and by
   **+10.3pp** on final accuracy (0.5868 vs 0.4839). The shuffled control is *worse* than
   the unprotected baseline, so the benefit is specifically from using **task-relevant**
   directions, not from constraining updates.
2. **Acquisition is preserved.** 0.7569 vs baseline 0.7513 — the projection does not cost
   learning. This rules out the False-Progress-3 reading ("low forgetting because it
   learned nothing") for this arm; the replay arms in round 4 were exactly that, and OGH
   is not.
3. **It is still far from useful.** 17pp of forgetting against a 3pp gate.

## 3. Where the mechanism fails, mechanistically

Look at seed 140023, the clearest case:

```
frozen_shared R = [[0.992, 0.746, 0.426],
                   [0.684, 0.637, 0.477],
                   [0.469, 0.422, 0.836]]
ogh           R = [[0.992, 0.746, 0.426],
                   [0.738, 0.695, 0.477],
                   [0.605, 0.625, 0.844]]
```

OGH raises the final `sim_xor` from 0.469 to **0.605** (+13.6pp) and `parity3` from
0.422 to **0.625** (+20.3pp) while *keeping* `order_short` at 0.844. So the protection
works directionally — but 0.605 is still below the 0.65 acquisition gate, and 0.605 vs
the 0.992 it once had is 38.7pp of forgetting.

**Root cause of the shortfall:** the basis `B` is capped at 8 directions in a 48-dim
space, and it stores only the *mean* gradient per task. Three tasks consume 3 of 8 slots,
so the projection removes only a thin slice of the parameter space. Most of the
interference is not addressed by a mean-gradient null space: the tasks' decision
functions overlap in the same few coordinates (all three read a 48-dim spike sum from the
same population), and a *linear* head has no spare directions for that overlap.

The deeper constraint: with 2 outputs and 48 inputs, a linear head has **98 parameters**
for three task boundaries that occupy the same subspace. Orthogonalisation preserves
directions; it cannot manufacture the capacity to separate three overlapping
distributions with 2 logits.

## 4. Interpretation, with the alternatives I rejected

| explanation | test | verdict |
|---|---|---|
| OGH works by using fewer directions | `ogh_shuffled` | **rejected** — shuffled is worse than baseline |
| OGH works by learning less | acquisition 0.7569 ≈ baseline 0.7513 | **rejected** |
| OGH works by task-relevant orthogonalisation | ogh − shuffled = 3.43pp | **supported** |
| OGH is sufficient | 17.01pp vs 3pp gate | **rejected** |

## 5. Honest conclusion about the mission's second goal

The mission wants acquisition ≥ 65%, final ≥ 65%, mean forgetting ≤ 3pp, worst ≤ 5pp, on
every seed.

Current state:

| requirement | best achieved | gap |
|---|---:|---|
| acquisition ≥ 0.65 | 0.7569 ✓ | met |
| final ≥ 0.65 | 0.5868 ✗ | **−6.3pp** |
| mean forgetting ≤ 3pp | 17.01pp ✗ | **5.7×** over |
| worst forgetting ≤ 5pp | 44.14pp ✗ | **8.8×** over |

**No mechanism I have tried comes within an order of magnitude of the remaining gate.**

And crucially: round 4b established that this gap is **not** the hidden learning rule.
Hidden plasticity contributed −1.17pp. The gap is a **readout capacity** problem, and the
head is a 98-parameter linear map by design of this research configuration.

## 6. What I am NOT claiming

- **Not** that orthogonalisation is useless. It is a measured +3.43pp over its own
  control with preserved acquisition, which is a real effect.
- **Not** that the gates are unreasonable. They are the mission's thresholds and I have
  not relaxed them.
- **Not** that continual learning is impossible for SCAR-X. It is impossible **for a
  98-parameter linear head over these features**, which is what is being tested.

## 7. The honest next step — and why it needs a decision, not more tuning

Every remaining lever is blocked or out of scope:

| lever | status |
|---|---|
| more head parameters / nonlinear head | **blocked** — the brief forbids expanding capacity to mask a learning-rule failure, and a bigger head would be exactly that |
| task-conditioned gating | **FP8 risk** — a supplied task ID is not semantic discovery |
| replay of raw data | tried; buys retention by destroying acquisition (round 4) |
| larger basis / better OGH schedule | plausible, small effect: the basis is already 8 of 48 dims and mean-gradient only |
| raise `leak` / add neurons | **forbidden** by the mission |
| hidden plasticity | measured irrelevant to forgetting (−1.17pp) |

**The truthful reporting is therefore:** the bound is readout capacity, it is a property
of the tested configuration (2-logit linear head, H48, spike-sum features), and the
mission's 3pp gate is not reachable by readout-side protection alone. I present this as
the round's conclusion rather than tuning OGH toward a headline number.

If the mission intends the gate to be reachable, the question to resolve first is
whether a **nonlinear** head is permitted — because a linear 2-logit head over
three overlapping 48-dim task distributions is the binding constraint, not the learning
rule. That is a scope question I will not decide unilaterally.

## 8. Declared false-progress exposures

| risk | exposure |
|---|---|
| **FP3** (low forgetting = learned nothing) | OGH acquisition 0.7569 preserved; shuffle control degrades, so the effect is not from doing less |
| **FP7** (optimizer change called mechanism) | `hidden_lr`, `trace_decay`, `update_clip` untouched; hidden frozen in all arms |
| **new: small effect reported as success** | OGH is 5.7× short of the gate; stated in the headline |

## 9. Message for Gemini

Two numbers from my runs that bear directly on any retention claim:

1. **An arm that cannot forget is the only way to validate the metric.** My first
   forgetting metric reported 7.68pp for a control with a separate head and core per
   task. The correct R-matrix version reads 0.00pp for it. If your retention metric has
   not been validated against a structurally-incapable-of-forgetting arm, it may be
   measuring task difficulty.
2. **Hidden plasticity contributes −1.17pp to forgetting while readout overwriting
   contributes +20.96pp.** If CKCER's retention comes from the hidden side, the baseline
   it must beat is one where the hidden is irrelevant — a much harder bar than beating a
   shared-head baseline.

And a constraint you may hit: my orthogonal-head mechanism preserved acquisition and
improved forgetting by 3.43pp over its own shuffled control, and still ended at
**17.01pp against a 3pp gate**. The remaining gap is readout capacity, not credit.

## 10. Artifacts

`scripts/r7_r5_ogh.py` · `runs/20261111T120304Z-r7-r5-ogh-v1-8467a0/result.json` ·
`report-round4.md` (contains the invalid-metric disclosure and the corrected
decomposition this round builds on)
---

## 11. ADDENDUM — the capacity claim is now measured, not inferred

Section 5 asserted that readout capacity is the binding constraint. That was an
interpretation. Round 5b tests it directly.

**These heads are DIAGNOSTIC ONLY.** The brief forbids expanding capacity to mask a
learning-rule failure; a wider or nonlinear head used as a *candidate* would be exactly
that. Used as a *learnability diagnostic* — the role BPTT plays — it answers one
question: is capacity what bounds retention?

3 dev seeds, frozen hidden, identical features and data, identical R-matrix protocol:

| head | params | acq | final | meanF | worstF | gate |
|---|---:|---:|---:|---:|---:|---|
| `linear_2logit` | 98 | 0.7513 | 0.5417 | **20.96pp** | 52.34pp | fail |
| `linear_wide_32` | 3 136 | 0.7227 | 0.6306 | **9.20pp** | 33.20pp | fail |
| `mlp32_tanh` | 1 602 | **0.8381** | **0.8095** | **2.86pp** | **9.38pp** | fail |

**The trend is monotone in capacity and large:**
- a wider *linear* head halves mean forgetting (20.96 → 9.20pp);
- a *nonlinear* head cuts it **7.3×** (20.96 → 2.86pp) while **raising** acquisition to
  0.8381.

**So the round-5 interpretation was correct, and incomplete.** Capacity is the dominant
constraint for **mean** forgetting. It is not sufficient for the **worst-case** gate:
9.38pp against 5pp.

### Why I still do not call this a pass

1. **worstF fails** at 9.38pp. The gate is 5pp.
2. **meanF 2.86pp is 0.14pp under the 3pp gate** on 3 seeds. The per-seed spread I
   observed across rounds is larger than 0.14pp, so this margin is not resolvable at
   this sample size. Calling it a pass would be exactly the "statistical significance
   confused with effect size" error the brief warns about.
3. **It is a diagnostic, not a candidate.** No production weight is derived from it and
   nothing is proposed for Hermes.

### What this settles

The mission's second gate is **not** blocked by the local learning rule or by hidden
plasticity — round 4b showed hidden plasticity contributes −1.17pp to forgetting. It is
blocked by **linear readout capacity over three overlapping task distributions**. That is
now a measured statement with a monotone dose-response, not a hypothesis.

### The scope question this raises

Reaching meanF ≤ 3pp **and** worstF ≤ 5pp simultaneously appears to require a nonlinear
or substantially larger head. Whether that is permitted is a decision I will not make
unilaterally, because it directly conflicts with "no increasing capacity to mask a
learning-rule failure". The distinction I would defend is:

- a nonlinear head used to **claim** hidden learning works → forbidden, and I have not
  done it;
- a nonlinear head used to **demonstrate that the bottleneck is readout capacity, not
  credit** → a legitimate diagnostic, and that is exactly how it is used here.

That reading needs the mission owner's confirmation before it becomes a design change.

### Declared exposures for this addendum

| risk | exposure |
|---|---|
| **FP10** (tests pass, therefore product ready) | nothing here is proposed as product |
| **capacity diagnostic read as a candidate** | labelled DIAGNOSTIC in the script name, the run metadata, and the table header |
| **thin margin reported as a pass** | §11.2 explicitly declines, 0.14pp under gate on 3 seeds |
