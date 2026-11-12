# ARGUMENT AUDIT — Round 6A

Format per claim: **Hypothesis → Mathematical prediction → Measured evidence →
Strongest alternative → Counter-test → Remaining uncertainty → Verdict.**

---

## C1 — The bottleneck is unique data, not update count

- **Hypothesis:** H1.
- **Prediction:** at fixed `U = 16 384`, more distinct rows ⇒ lower held-out error.
- **Measured:** repeated-256 `both_A = 0.7920`; unique-16384 `both_A = 0.9244`;
  **+13.25pp [+6.99, +19.40]** over 8 untouched seeds. Repeated-256 reaches
  `train = 1.0000`.
- **Strongest alternative:** the repeated arms saw the same rows so often that clip
  drift dominated; the comparison is about optimisation pathology, not information.
- **Counter-test:** clip fraction is *lower* for the repeated arms in absolute step
  count but the arm still converges to train 1.0, i.e. it is not stuck. And the
  direction is right: more distinct rows help at *equal* updates. If optimisation
  pathology were the cause, the repeated arm would underfit train too — it does the
  opposite.
- **Remaining uncertainty:** both arms share one generator; "diversity" here means more
  draws of the same distribution, not a broader distribution.
- **Verdict: SUPPORTED.**

## C2 — Update count saturates and then degrades

- **Hypothesis:** H2.
- **Prediction:** if updates were the growth axis, `both_A` should keep rising with
  epochs toward the ridge limit 0.9924.
- **Measured:** 1→2→4→8→16 epochs: 0.7338 / 0.7945 / **0.8318** / 0.8269 / **0.8036**;
  train also falls 0.9775 → 0.9473.
- **Strongest alternative:** the epoch scan used one order per seed, so drift is order
  artefact.
- **Counter-test:** the matched-update confirmation used fresh permutations per arm and
  the same non-monotonicity appears in the ladder (`n=1024` at 1 epoch = 0.7176 vs
  16 epochs = 0.9014 in the *matched* setting, where more epochs over the same 1024 rows
  partially recover — so repetition is not useless, just far weaker than data).
- **Remaining uncertainty:** the exact epoch at which degradation starts is
  seed-dependent and I did not localise it.
- **Verdict: SUPPORTED as "saturates, then harms", NOT as "updates never help".**

## C3 — The representation is not the constraint

- **Hypothesis:** H3.
- **Prediction:** if representation-limited, the `ridge − direct` gap persists.
- **Measured:** gap 0.1152 → 0.0786 → 0.0426 → **0.0228**.
- **Strongest alternative:** ridge is also improving, so the gap closes from both sides.
- **Counter-test:** ridge is **0.9578 at n=256** and only 0.9965 at n=16 384 — it moves
  3.9pp, while the head moves 40pp. The gap closes overwhelmingly because the head
  catches up, not because ridge improves.
- **Remaining uncertainty:** ridge is a linear diagnostic, not an information ceiling;
  a nonlinear reader might do better and would not change this verdict.
- **Verdict: SUPPORTED.**

## C4 — There is no delayed learning transition

- **Hypothesis:** H4.
- **Prediction:** interior inflection exceeding seed dispersion, per-curve consistent.
- **Measured:** aggregated curve has 2 accelerations and saturating-fit residual 0.0739,
  but **per-curve positive d2 = 18/36 = 0.500**, mean |d2| 0.0512 < seed sd 0.0631,
  and **6/6** curves dip.
- **Strongest alternative (against me):** my per-curve test is too strict and a genuine
  transition is being missed.
- **Counter-test:** if a real transition existed, it would appear as a consistent
  ordering of the jump across orders and seeds. It does not: at the same step, one
  curve contributes +0.0847 and another −0.0485. A transition cannot have opposite
  signs on the same data budget.
- **Remaining uncertainty:** a transition might exist at budgets between my grid points
  (e.g. between 512 and 1024); I sampled 8 points on a log grid.
- **Verdict: REFUTED at the resolution tested.** I record that my *first* automatic
  verdict said `transition_claimed: True`; the registered per-curve test overrode it.

## C5 — No shortcut explains the result

- **Prediction:** random labels ≈ chance, first-key ≈ 0.75, query-only ≈ 0.50.
- **Measured:** random labels **0.5251**, first-key **0.7419**, query-only **0.4987**.
- **Strongest alternative:** the head exploits write position rather than content.
- **Counter-test:** the `needB` column (query=B and `vA≠vB`, where first-key is
  structurally 0.50) reaches **0.9557** at n=16 384. A position reader cannot reach that.
- **Verdict: SUPPORTED.**

## C6 — Held-out robustness (FAILED gate)

- **Prediction (my own gate G6):** every stress row above the 0.7419 ceiling.
- **Measured:** temporal mean 0.8256 but **seed 121017 = 0.5557**; distractor mean
  0.7726 but **121017 = 0.5713**.
- **Strongest alternative:** a harness defect, as happened in R4A and R5B.
- **Counter-test:** the stress features go through the same `capture_features` path with
  the query tick located from data (`QT+1` after shift), and the same head that scores
  0.97 clean. One seed fails while seven pass, which is not the signature of a global
  harness bug (that would fail all seeds).
- **Remaining uncertainty:** why 121017 specifically. Its clean accuracy is high, so it
  is not a generally weak population — it is fragile under perturbation.
- **Verdict: GATE FAILED. `confirmation_pass: False`.**

## C7 — Hidden plasticity (H5)

- **Status: BLOCKED.** Not run, not assumed in either direction, and not substituted by
  a related result.

---

## Three ways my favoured explanation could be wrong

### Objection 1 — "You changed the calibration rule relative to R5B, so the comparison is invalid."

**Test run.** R5B's `Head` z-scores inside `learn` using running statistics; R6 scales
features outside with a **fixed 256-episode training prefix**. I verified the update
arithmetic is **bit-identical** to R5B's (`test_head_update_matches_r5b_bit_exactly`
passes on weights and bias). The difference is the scaling source, which the brief
*requires* ("fixed, training-only calibration set common to all budgets").

**Still uncertain:** the fixed prefix is not R5B's batch statistic, so my absolute
numbers are not numerically comparable to R5B's 0.8989. The *shape* and the
unique-vs-repeated contrast are unaffected because every arm uses the same prefix.

### Objection 2 — "The 16 384 arm simply has a better-conditioned feature set, because more data improves the z-scaling."

**Test.** Calibration is a **fixed 256-episode prefix identical for every budget**, so
the scaling transform is *literally the same matrix* at every `n`. The only thing that
changes is how many distinct rows the head sees. I ran it this way specifically to
remove this confound.

**Still uncertain:** the *features* at different `n` come from different draws (budgets
are not nested), so per-budget feature distributions differ slightly.

### Objection 3 — "The 13.25pp unique-vs-repeated gap is a clip artefact, not an information effect."

**Test run.** If clipping dominated, the repeated arms would fail to fit their training
data. They do the opposite: repeated-256 reaches **train 1.0000**. A clip-bound learner
cannot reach perfect training accuracy while being unable to transfer; it would
underfit train as well. The repeated arm is fully converged on its sample and still
13.25pp behind on held-out.

**Still uncertain:** the clip does affect *how fast* each arm converges, and I did not
measure convergence curves within each arm, only endpoints.

---

## Limitations: confirmed vs speculative

**Confirmed by execution:**
- data is the binding factor: +13.25pp at matched updates, 8 seeds;
- updates saturate then degrade: 0.8318 → 0.8036;
- the ridge gap closes to 0.0228, so representation is not limiting;
- no transition: per-curve d2 at chance, 6/6 curves dip;
- no shortcut: random 0.5251, first-key 0.7419, query-only 0.4987;
- `both_A` at 16 384 has min 0.7984 across seeds.

**Speculative / unestablished:**
- that this generalises past one generator and one architecture;
- any hidden-plasticity conclusion (BLOCKED);
- the cause of seed 121017's stress fragility;
- that more data helps SCAR-X on any non-relational task.

## Final argument

The evidence supports **GO on the scientific question** — the R5B 0.6676 was a
data-scarcity effect, and the three candidate causes are cleanly separated with data as
the binding one — but **STOP as a gated confirmation**, because the registered stress
gate G6 fails on one of eight seeds. I did not relax G6. The transferable finding is the
matched-update contrast: repetition buys perfect training fit and no transfer, while the
same update budget spent on distinct episodes buys 13.25pp of held-out relational
accuracy.
