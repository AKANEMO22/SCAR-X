# MATH ANALYSIS — Round 6A

Brief §J requires eight items. Each answered, with the claims that admit a counterexample
checked numerically.

---

## 1. Why XOR over four logical combinations cannot demonstrate "learning many concepts"

The label is `y = vA ⊕ (q·(vA ⊕ vB))` with `vA, vB, q ∈ {0,1}`. The function has
**8 input points** (4 value pairs × 2 queries) and takes **2 output values**. Its
description length is tiny; a lookup table has 8 entries, and the label distribution is
uniform so the entropy is exactly 1 bit.

Consequently any learned artefact represents **at most one bit of decision structure**
beyond the query. Reaching 0.92 on this task demonstrates that a *relational read
operation* works — it does **not** demonstrate general concept learning, because there
are no other concepts in the data. Four memory states × two queries is a single
generalisation problem, and the 0.75 shortcut ceiling exists precisely because two of
the four states are symmetric under the query.

## 2. Unique data count vs repeated update steps

Let `S = {(f_i, y_i)}_{i=1..N}` be distinct rows and `U = N·E` the update count for
`E` epochs. The per-example delta rule performs

```
W ← clip( W + clip(η·err(f_i,y_i) ⊗ f_i, −c, c), −3, 3 )
```

Repeating row `i` `E` times applies the same functional step `E` times from different
weight states, which converges to a **stationary point of the empirical risk on S**.
The generalisation error is then controlled by `|S|`, not by `U`.

**Measured:** at `U = 16 384`, `N = 256` gives `both_A = 0.7920` and train `1.0000`;
`N = 16 384` gives `0.9244`. Fixed `U`, different `N`, **+13.25pp**. So `U` and `N` are
not interchangeable and `N` is the binding variable.

## 3. When a fixed feature map is linearly decodable but an online delta rule underfits

Three necessary conditions, all observed here:

1. **Ill-conditioning.** Feature coordinates have wildly different scales. In R4B's
   geometry audit the decay-0.25 features had median sd 0.0042 and condition number
   ~1.5e7. A single learning rate then moves some coordinates far too slowly.
2. **A saturating clip.** With `c = 0.025` and `η = 0.08`, `η·|err|·|f|` exceeds `c` for
   most coordinates, so the step is **magnitude-blind**: every coordinate moves by ≤ c
   regardless of how much it needs to move. Measured clip fraction falls from **33.3%**
   at `n=256` to **7.4%** at `n=16 384` — as data grows, fewer updates saturate.
3. **Finite samples.** The empirical optimum of a small `S` is far from the population
   optimum, so even a converged rule underfits the population.

Ridge does not suffer (1) because it solves the normal equations in one step with an
explicit ridge penalty, and does not suffer (2) because it has no clip. Hence
`ridge ≫ head` at small `N` and they converge at large `N`: gap **0.1152 → 0.0228**.

## 4. Does the update rule admit a convergence argument for a convex readout?

Yes, under three assumptions that hold in a simplified setting:

Assume (a) the readout is linear with a **convex** loss (softmax cross-entropy is
convex in `W` for fixed features), (b) **no clipping** (`c = ∞`, `wmax = ∞`), (c)
features in a **bounded** set with finite second moment, and (d) a **fixed** step `η`
small enough that `η·sup|f|² < 1/L` where `L` is the loss's Lipschitz-smooth constant.

Under (a)–(d) the per-example update is stochastic gradient descent on a convex
objective with bounded gradient, so `E[loss]` converges to the optimum at rate
`O(1/√T)` in the stochastic case and `O(1/T)` in the full-batch case. Crucially the
bound depends on the **number of distinct samples** through the variance term: with
`N` distinct rows the gradient noise variance is `O(1/N)`. **That is the formal reason
data count, not step count, controls the final error.**

## 5. Which assumptions the real system breaks

| assumption | reality | consequence |
|---|---|---|
| (a) convex | holds for the readout; **not** for the frozen population downstream | the readout sub-problem is convex, so the argument applies to it |
| (b) no clip | **broken** — `c = 0.025` with `η = 0.08` saturates on 33% → 7% of updates | step is magnitude-blind; convergence guarantee lost |
| (c) bounded features | holds after z-scaling with a floor of 0.05, but the floor **compresses** small-variance coordinates | effective conditioning is worse than nominal |
| (d) fixed small η | `η = 0.08` fixed, but the clip makes the effective step data-dependent | no single `η` satisfies the condition for all coordinates |
| — | recurrence makes features depend on history, so "fixed features" is an approximation of a frozen snapshot | acceptable here because **hidden weights are frozen** |

So the convex argument explains the **shape** of the curve (error falling with `N`) but
cannot bound the **constant**, which is why the measured clip fraction matters.

## 6. Neural learning transition vs optimisation artefact

A **transition** would require the learning curve to have an interior inflection that
(a) exceeds seed dispersion, (b) is not explained by a saturating law, and (c)
reproduces under changed order and fresh seeds. An **optimisation artefact** is a bump
produced by the interaction of a fixed step size with a moving loss surface — it
reduces to noise once seeds are averaged.

**Measured.** Aggregated: 2 interior accelerations, saturating fit max residual 0.0739.
Per curve: **18/36 second differences positive = 0.500**, exactly chance; mean |d2| =
0.0512 versus seed sd 0.0631; **6/6** per-seed curves dip.

⇒ **optimisation/aggregation artefact, not a transition.**

I note explicitly that my *first* verdict read this as a transition, because the check
only looked at the aggregated curve. The registered criterion (per-curve test) rejected
it.

## 7. Are improvements reproducible across seeds, or a few lucky populations?

**Reproducible.** At `n = 16 384`, `both_A` per seed has **min 0.7984, mean 0.9237,
max 0.9921** across 8 untouched seeds. The worst seed still clears the 0.75 gate. At
`n = 256` the range is min 0.4274 / max 0.7778 — that is where seed luck dominates, and
it is exactly why the small-data screen was unreliable.

The one genuine seed-level failure is **the stress conditions**: seed **121017** falls
to 0.5557 (temporal) and 0.5713 (distractor). That is a real population-level fragility
and it is what fails gate G6.

## 8. Do hidden weight changes produce measurable benefits beyond readout learning?

**Untested in this round (H5 BLOCKED).** I will not answer this from R5B's related
result, because that experiment did not use this generator, these seeds or this budget
ladder.

---

## Numerical counterexamples to appealing readings

### Counterexample 1 — "more epochs will close the gap"

At `n = 1024`, 4 → 16 epochs (**4× updates, no new data**):
`both_A` **0.8318 → 0.8036**, `train` **0.9775 → 0.9473**.
Ridge on the same features is 0.9924 and does not move.
A "train longer" reading predicts monotone approach to 0.9924; the measured direction is
opposite. Meanwhile 1 epoch on **4× more unique data** gives 0.7176 → 0.9237.

### Counterexample 2 — "training accuracy proves learning"

`repeated-256` at 16 384 updates achieves **train accuracy 1.0000** and `both_A` 0.7920.
Perfect reproduction of 256 episodes coexists with a 13.25pp transfer deficit against
16 384 distinct episodes. Reporting training accuracy here would be actively misleading.

### Counterexample 3 — "the bump at 1024 is a phase transition"

The aggregated curve jumps +0.136 across one step. Per-seed, the same step gives
−0.0485 on one curve and +0.0847 on another; 18 of 36 second differences are positive,
which is chance. The bump is arithmetic on noisy points.

---

## Complexity

| quantity | value |
|---|---|
| features per example | 258 (192 state + 2 cue + 2×32 gated) |
| readout parameters | 518 |
| updates at largest budget | 2^(14) = 16 384 |
| FLOPs per update | ≈ 2·2·258 ≈ 1 032 |
| peak process-tree commit | 201 953 280 B (target 2 000 000 000) |
| wall clock, full confirmation | 26–40 s per stage |

## What is not claimed

- No exponential growth law: accuracy is bounded at 1.0 and the fitted law is
  saturating (`A − B·e^(−U/τ)`), not exponential.
- No claim that ALIF is incapable of anything, from a ridge result.
- No claim that reading a frozen state is learning.
