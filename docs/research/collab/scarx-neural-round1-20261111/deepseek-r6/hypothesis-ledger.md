# HYPOTHESIS LEDGER — Round 6A

Written before the scored confirmation run, updated only with measured values. Every
hypothesis states the observation that would reject it.

---

## H1 — Data-limited learning

**Claim.** More UNIQUE examples improve held-out performance even when total update
count is held constant.

**Mathematical prediction.** If the local delta rule is fitting an approximately
separable structure, then at fixed `U` the held-out error scales with the *number of
distinct constraint rows*, not with the number of gradient steps. At fixed `U`, more
distinct rows ⇒ lower held-out error. Under heavy repetition the rule converges to the
empirical risk minimiser of the 256-row sample, so held-out error plateaus at that
sample's generalisation error regardless of `U`.

**Required evidence to reject H1.** `unique-16384` and `repeated-256` are
indistinguishable at `U = 16384`.

**Measured (confirmation, 8 untouched seeds).**
`repeated-256 both_A = 0.7920` vs `unique-16384 both_A = 0.9244`;
difference **+13.25pp [+6.99, +19.40]**. Repeated-256 also reaches **train 1.0000**,
i.e. it fits its 256 rows perfectly and still fails to transfer.

**Verdict: H1 SUPPORTED.**

---

## H2 — Update-limited learning

**Claim.** More weight updates help even without additional unique information.

**Mathematical prediction.** Under a convex surrogate the delta rule converges to the
empirical optimum; extra passes over a fixed sample reduce training loss monotonically
and held-out loss to that sample's limit. Once reached, additional passes cannot help.

**Required evidence to reject H2.** Repeated training plateaus or degrades while
unique-data training keeps improving.

**Measured (screen, n=1024, epochs 1→16).**
`both_A`: 0.7338 → 0.7945 → **0.8318 (4 ep)** → 0.8269 → **0.8036 (16 ep)**.
Train accuracy also degrades 0.9775 → 0.9473.

**Verdict: H2 REFUTED as a growth mechanism.** It helps to ~4 epochs, then degrades —
consistent with clip saturation drift, not with continued learning.

---

## H3 — Representation-limited learning

**Claim.** The representation, not the learner, is the constraint: an offline reader
solves held-out data that the online head cannot.

**Mathematical prediction.** A large persistent `ridge − direct` gap that does **not**
close as data grows. If the gap closes, the representation was sufficient and the
constraint was optimisation under data scarcity.

**Required evidence to reject H3.** The ridge gap shrinks materially with data.

**Measured (confirmation).** Gap `ridge − direct`:
0.1152 (n=256) → 0.0786 → 0.0426 → **0.0228 (n=16 384)**.

**Verdict: H3 REFUTED.** The information was always there; the learner needed data.

---

## H4 — A delayed learning transition

**Claim.** The curve contains a reproducible transition region rather than ordinary
gradual improvement or seed noise.

**Mathematical prediction.** An interior inflection: second differences positive in a
region, with a magnitude **exceeding between-seed dispersion**, and a saturating law
that cannot account for it.

**Required evidence to reject H4.** The apparent jump vanishes on fresh orders/seeds, or
is smaller than seed sd, or is well fit by a smooth saturating curve.

**Measured (fine sweep, 3 seeds × 2 orders × 8 budgets).**
The aggregated curve has 2 interior accelerations and the saturating fit leaves max
residual 0.0739 — superficially consistent with a transition. But:

| test | result |
|---|---|
| per-curve positive second differences | **18/36 = 0.500** (chance) |
| mean \|second difference\| | 0.0512 |
| between-seed sd | **0.0631** |
| curves containing a dip | **6/6** |

**Verdict: H4 REFUTED.** The bump is seed noise aggregated. Rejected by my own
registered criterion, which is why that criterion required the per-curve test.

---

## H5 — Hidden plasticity is necessary

**Claim.** A hidden-plasticity arm beats matched frozen-hidden learning after
substantial data.

**Status: BLOCKED, untested.** The existing local hidden rule lives behind
`PlasticCore`/`UnifiedCore` in `src/scarx/neural/`, which this round must not modify.
Reproducing it faithfully in research-only code requires reimplementing eligibility
traces and their state lifecycle; an approximate reimplementation would not be the
registered rule.

**I do not assume H5 is true and I do not assume it is false.** R5B found the hidden
rule unhelpful on a related task, but that is a different experiment and is not used
here as a substitute.

---

## Separation table — the brief's three causes

| cause | isolated by | measured effect | verdict |
|---|---|---|---|
| unique examples | `n_unique` at 1 epoch | 0.5407 → 0.9237 | **binding** |
| update count | epochs at fixed `n` | peaks 0.8318 then 0.8036 | saturates, harms |
| representation | ridge gap | 0.1152 → 0.0228 | not limiting |

## Counterexample required by the brief (section J)

**Appealing but incorrect reading:** "the head needs more training, so let it run more
epochs."

**Numerical counterexample from these runs:** at fixed `n = 1024`, going from 4 epochs
to 16 epochs — a **4× increase in updates with no new data** — moves `both_A` from
**0.8318 down to 0.8036**, while `train` also falls from 0.9775 to 0.9473. A
more-training explanation predicts monotone improvement toward the ridge limit of
0.9924; the observed direction is the opposite. Meanwhile adding 4× more *unique*
examples at the same 1-epoch budget moves `both_A` from 0.7176 to 0.9237. **Updates and
data are not interchangeable, and here they have opposite signs past a small budget.**

## Second counterexample: repetition reaching perfect fit without transfer

`repeated-256` at 16 384 updates reaches **train accuracy 1.0000** — it can reproduce
every one of its 256 training episodes exactly — while `both_A` is 0.7920. A reader who
reported training accuracy would call this "fully learned". It has learned nothing
transferable. This is the numerical case that makes "repeated training is evidence of
learning" plainly false here.
