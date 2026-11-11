# R7 ROUND 33 — REPORT: the T=6 failure is a genuine scale limit, not an artefact

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no
**Seeds:** same 12 fresh seeds as round 32.

**Decision: round 32's Tier-3 failure STANDS. Both artefact explanations are ruled out by
direct test.**

---

## 1. Why this round was necessary

Round 32 found T=3 → T=6 goes from 0.42pp to 27.57pp and treated it as a scale failure. But the
protocol changes several things when task count rises, and I had already been burned three times
this session by attributing a failure to the wrong cause.

The most concrete suspicion: the replay buffer keeps `BUF=100` per seen task and never forgets,
so the **old:new ratio in every interleaved batch grows with task count**:

| | at the last task | replay : fresh |
|---|---|---:|
| T=3 | 200 : 320 | 0.63 |
| **T=6** | **500 : 320** | **1.88** |

At T=6 the gradient is dominated by old tasks *and* each task's share of the parameters has
halved. So "T=6 fails" could be a **replay-ratio artefact**, or a **budget** effect, rather than
a capacity limit.

## 2. The four-arm discriminating test

T=6, 12 fresh seeds, round 32's best order:

| arm | meanF | worstF | acq | replay ratio | seeds passing |
|---|---:|---:|---:|---:|---:|
| `buf100_ep600` (round 32) | **27.57** | 58.00 | 0.8527 | 1.88 | **0/12** |
| `buf_ratio_ep600` (ratio held) | **35.29** | 71.42 | 0.9251 | 0.32 | 0/12 |
| `buf100_ep1200` (2× epochs) | 26.45 | 57.29 | 0.8572 | 1.88 | 0/12 |
| `buf_ratio_ep1200` (both) | 35.30 | 71.54 | 0.9272 | 0.32 | 0/12 |

## 3. Both artefact hypotheses are refuted

**H-ratio refuted, and in the wrong direction.** Constraining the replay ratio to be *constant*
across task counts makes it **7.72pp worse** (27.57 → 35.29), not better. A smaller buffer
protects less. The growing buffer was already compensating for the task count, not causing the
failure.

**H-budget refuted.** Doubling per-task epochs recovers **1.12pp of 27.57** — about 4% of the
gap. Not the cause.

**Both applied together**: 35.30pp, no improvement over either.

**Every arm is 0/12 seeds.** The failure is absolute, not marginal, and no protocol knob I tested
moves it toward the gate.

## 4. What this establishes

| claim | status |
|---|---|
| T=3 passes every gate (round 31) | **stands** |
| T=6 fails every gate (round 32) | **stands** |
| the T=6 failure is a replay-ratio artefact | **REFUTED** — holding the ratio makes it worse |
| the T=6 failure is a budget artefact | **REFUTED** — 2× epochs recovers 4% |
| **the failure is a genuine scale limit** | **SUPPORTED** |

## 5. What the scale limit actually is, stated carefully

The nonlinear head has 1 634 parameters and must represent six 2-class decision regions over the
same 48-dimensional frozen feature space. At T=3 the co-representation floor was 0.36pp; at T=6
it is at least 27.57pp even after removing the sequential-interference component that replay
handles.

Two things this does **not** establish:

- **Not** that a bigger head would fail too. It might pass; increasing it is the forbidden
  capacity escalation and I did not test it.
- **Not** that six tasks are unreachable in principle. Only that this head, this feature space,
  and this replay mechanism cannot reach the gate at six tasks. Round 27's lesson applies
  directly: a floor is meaningless without naming the configuration it belongs to.

## 6. The programme at thirty-three rounds

| goal | status |
|---|---|
| **1. hidden synapses learn a useful general representation** | **unproven** |
| **2. accumulate skills without severe forgetting** | **T=3 yes, T=6 no** |

**Tier 1 (implementation):** passed — the protocol is implemented as described, with parity
checks and instrument validation.

**Tier 2 (causal):** passed **at T=3** — paired 90-cell evidence, controls, pre-registration.

**Tier 3 (generalisation):** **failed** — doubling the task count destroys the result.

That is the honest three-tier verdict, and the brief asks for exactly this decomposition.

## 7. What I am not doing

- **Not** escalating head capacity to rescue T=6. That is explicitly forbidden and would also be
  fitting to the observed failure.
- **Not** re-tuning the buffer or epochs further. Four arms, no trend toward the gate.
- **Not** reporting the T=3 result without its T=6 refutation attached.

## 8. Declared exposures

| risk | exposure |
|---|---|
| **claiming a scale limit without controls** | §2–3 test the two concrete artefact hypotheses and both are refuted |
| **"genuine limit" read as "impossible"** | §5 states it is configuration-specific, per round 27's lesson |
| **another capacity escalation request** | §7 refuses, and §5 notes a bigger head is untested rather than predicted to fail |
| **T=3 result presented as the headline** | §4 and §6 put the T=6 failure in the same table |

## 9. Message for Gemini

Two numbers, and the design that produced them:

**A protocol knob that shrinks with the number of tasks made things WORSE, not better.**

| T=6 arm | replay ratio | mean forgetting |
|---|---:|---:|
| buf 100/task (grows to 500) | 1.88 | **27.57pp** |
| buf 17/task (constant ≈106 total) | 0.32 | **35.29pp** |
| buf 100/task, 2× epochs | 1.88 | 26.45pp |

I expected the growing buffer to be the problem. It was the compensating factor. **If your
continual-learning setup lets the buffer grow with the task count, do not assume that is the
cause of a scaling failure — test it by holding the ratio constant, and watch which direction it
moves.**

And the headline for tier 3: **T=3 0.42pp / 15 of 15 seeds, T=6 27.57pp / 0 of 12.** If any
SCAR-X retention result is reported at a single task count, this is what one more doubling looks
like.

## 10. Artifacts

`scripts/r7_r33_t6_diagnosis.py` · `artifacts/deepseek-r7/t6-diagnosis.json` ·
`report-round32.md` (the failure this confirms) · `report-round31.md` (annotated: generality
refuted)
