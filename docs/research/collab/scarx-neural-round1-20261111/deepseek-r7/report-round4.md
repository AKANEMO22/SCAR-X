# R7 ROUND 4 — REPORT: forgetting is readout interference, not hidden plasticity

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: REJECT the round-4a metric (my own bug), KEEP the round-4b decomposition.**

---

## 1. I published an invalid forgetting metric and caught it

Round 4a computed `forgetting = acc(task0) − acc(taskK)` — a difference between
**different tasks**. That measures task difficulty, not forgetting.

**The proof it was wrong, from my own data:** `isolated_pertask` gives every task its own
head and its own frozen core, so it is *structurally incapable* of forgetting. Round 4a
reported **7.68pp mean forgetting** for it. A control that cannot forget cannot have
forgetting.

Round 4a is committed and marked `INVALID_METRIC`. It is not cited as evidence.

## 2. Correct protocol

```
R[i][j]        = accuracy on task j after training through task i
acquisition_j  = R[j][j]                  (right after learning j)
final_j        = R[T-1][j]                (after the whole sequence)
forgetting_j   = acquisition_j - final_j
```

**Instrument check (must pass before any conclusion):** `isolated_pertask` now reports
**0.00pp** — exactly as required. The instrument is calibrated.

## 3. Result — 3 seeding, cascaded tasks

| arm | acq | final | meanF | worstF | gate |
|---|---:|---:|---:|---:|---|
| `isolated_pertask` (check) | 0.9661 | 0.9149 | **0.00pp** | **0.00pp** | instrument ✓ |
| `frozen_shared` | 0.7513 | 0.5417 | **20.96pp** | **52.34pp** | fail |
| `learned_shared` | 0.7448 | 0.5469 | **19.79pp** | **49.61pp** | fail |
| `frozen_replay` | 0.4818 | 0.4822 | −0.04pp | 16.02pp | fail |
| `learned_replay` | 0.4688 | 0.4661 | 0.26pp | 17.58pp | fail |

**No arm passes.** Acquisition ≥ 0.65 and final ≥ 0.65 are met only by the isolated
control (which trains three separate models, so it is not a continual learner and its
"fail" is a labelling artefact of my table, not a result).

## 4. The causal decomposition — this is the round's real finding

| source | arm contrast | contribution |
|---|---|---:|
| **S2 readout interference** | `frozen_shared` 20.96pp − `isolated` 0.00pp | **20.96pp** |
| **S1 hidden plasticity** | `learned_shared` 19.79pp − `frozen_shared` 20.96pp | **−1.17pp** |
| **S3 replay** | `frozen_replay` meanF −0.04pp, but acq 0.4818 | *traded acquisition away* |

**Hidden plasticity contributes essentially nothing to forgetting (−1.17pp, i.e. it is
very slightly *better* than frozen).** Essentially all of the 21pp mean forgetting is
**the output head being overwritten by the next task.**

This is a clean causal statement, and it is directly falsifiable and was tested: if
hidden plasticity were the cause, `learned_shared` would be worse than `frozen_shared`.
It is not.

### Replay is not a solution here

Replay flattens mean forgetting to ≈0pp — but its **acquisition collapses to 0.48**, below
the 0.65 gate. It bought retention by not learning. That is **False Progress 3** in the
mission's own list ("low forgetting because it learned nothing"), declared explicitly.

### Why the head is overwritten

Look at one R matrix (seed 140023, `frozen_shared`):

```
R = [[0.992, 0.746, 0.426],
     [0.684, 0.637, 0.477],
     [0.469, 0.422, 0.836]]
```

`sim_xor` drops **0.992 → 0.469** as the head is retrained on `parity3` then
`order_short`. Meanwhile `order_short` rises to 0.836 — so the same 48-dim feature space
holds multiple tasks, but **one linear head cannot hold them simultaneously**. That is
classic readout capacity/orthogonality conflict, not representation destruction.

Note also `R[0][1] = 0.746 > R[1][1] = 0.637`: `parity3` was *more* readable from a head
trained on `sim_xor` alone than from one trained on `parity3`. The two tasks' decision
boundaries interfere.

## 5. What this changes about the mission's second goal

The mission asks for a learner that accumulates skills without severe forgetting. The
evidence says the blocker is **not** hidden plasticity — it is that a single shared
linear readout cannot represent three task boundaries at once.

That reframes the search. The candidates that could actually help are readout-side:

- a head with more capacity (blocked: would need more neurons/params — **not proposed**),
- task-conditioned gating using only legal inputs (**danger: FP8** — a supplied task ID
  is not semantic discovery),
- **orthogonal/subspace-preserving head updates** — keep each task's update in a
  direction that does not disturb previous tasks' decision directions. This is local,
  needs no task ID, no replay, no extra neurons, and is testable.

I have **not** implemented that. It is the registered next round, and it must be tested
with the same R-matrix instrument plus the `isolated_pertask` control.

## 6. Honest limits of this round

1. **Three tasks, three seeds.** No confirmation.
2. **`learned_shared` vs `frozen_shared` differ by only −1.17pp**, which is inside
   seed noise for 3 seeds. The correct reading is **"hidden plasticity does not
   measurably worsen forgetting"**, NOT "hidden plasticity is proven neutral". I do not
   have the CI to claim the latter and the report does not.
3. **Replay buffer of 192 episodes** was chosen without a budget sweep; its acquisition
   collapse may be an artefact of that size rather than intrinsic.
4. **Task order was fixed** (sim_xor → parity3 → order_short). Order effects are known to
   matter in continual learning and I did not permute.

## 7. Declared false-progress exposures

| risk | exposure |
|---|---|
| **FP3** (low forgetting because nothing was learned) | replay arms: meanF ≈ 0 but acq 0.48 — declared |
| **FP2** (ridge better, hidden unchanged) | `learned_shared` ≈ `frozen_shared`; hidden delta reported |
| **new: invalid metric published** | round 4a superseded and marked; control exposed it |
| **FP7** (optimizer change called mechanism) | `hidden_lr`, `trace_decay`, `update_clip` unchanged |

## 8. Message for Gemini

If you are reporting continual-learning results, the one check that would have caught my
error: **include an arm that is structurally incapable of forgetting** (one head per task,
one frozen core per task). Under a correct metric it must read ~0pp. Under the metric I
first used, it read 7.68pp.

Second: my decomposition says hidden plasticity contributes **−1.17pp** to forgetting
while readout overwriting contributes **20.96pp**. If CKCER's retention gain comes from
the hidden side, it has to beat a baseline where the hidden is irrelevant to forgetting —
which is a much harder bar than beating a shared-head baseline.

## 9. Artifacts

`scripts/r7_r4_continual.py` (INVALID metric, retained) ·
`scripts/r7_r4b_rmatrix.py` ·
`runs/20261111T115956Z-r7-r4-continual-v1-53a2e6` (invalid) ·
`runs/20261111T120113Z-r7-r4b-continual-rmatrix-v1-a5a0b0` (valid)

---

> ## RETRACTION (round 16): the replay arms were trained on WRONG LABELS
>
> The replay conclusions in this report are **withdrawn**. The bug is at
> `r7_r4b_rmatrix.py` lines 170–177:
>
> ```python
> buf.append(trx[i].copy())                                   # sequence, NO label
> Ry = np.array([try_[k % N_TR] for k in range(len(buf))], np.int32)   # CURRENT task labels
> ```
>
> `try_` is the current task's label array, so sequences stored from **previous** tasks
> were trained against labels from the **current** task, applied positionally. The replay
> arms were learning near-random labels.
>
> **Corrected (round 16, labels stored with sequences):**
>
> | arm | acq | meanF | worstF |
> |---|---:|---:|---:|
> | frozen_shared | 0.8819 | 31.90pp | 54.30pp |
> | **frozen_replay** | **0.8607** | **9.38pp** | 18.75pp |
> | learned_replay | 0.8594 | 10.29pp | 24.22pp |
> | frozen_pertask (instrument) | 0.9123 | 0.00pp | 0.00pp |
>
> So the statements below — "replay flattens mean forgetting to ≈0pp but its acquisition
> collapses to 0.48", and the **FP3** declaration — are **artefacts of my bug**.
>
> Corrected replay **preserves acquisition (0.8607, not 0.48)** and **reduces mean
> forgetting 3.4× (31.90 → 9.38pp)**. It still fails the 3pp gate, but it is the best
> retention result of this session.
>
> The causal decomposition in this report (hidden −1.17pp, readout +20.96pp) used
> `frozen_shared` and `learned_shared`, which were **not** affected by the label bug, and
> it stands. See `report-round16.md`.
