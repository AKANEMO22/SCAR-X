# Lessons learned — antigravity — campaign `scarx-neural-round1-20261111`

Environment: DeepSeek Harness (DSH) Web, Windows 11.
Model actually used: `deepseek-ai/deepseek-v4.1-flash-1017`.

---

## 1. Which of my own hypotheses was wrong?

**My primary hypothesis was wrong in its learning claim and right only in its
diagnostic claim.** I predicted that giving the readout a multi-timescale basis
would raise acquisition *and* increase the measured hidden-representation gain,
because the single-decay readout was masking what the hidden weights learned.

The screen rejected that:

| prediction | measured | verdict |
|---|---|---|
| MTB beats the dimension-matched control | +3.60pp [−1.30, +8.77] | rejected |
| MTB beats same-width single-timescale | +4.90pp [−0.22, +10.42] | rejected |
| The learning signal matters | +0.65pp [−0.95, +2.30] | rejected |
| Hidden learning contributes | +0.43pp [−0.91, +1.87] | **rejected decisively** |

The last row is the important one. `mtb` (68.88 on `order`) is *identical* to
`mtb_frozen` (66.54) and `mtb_readout` (66.54) within noise, and `mtb_shuffle`
(66.15) shows the sign of the learning signal is irrelevant. So the mechanism I
proposed does exist — a slower readout basis does recover a delayed order bit —
but **it is not learning**, and I was wrong to frame it as a representation-learning
improvement. I also over-generalized: everything sits on `order`; `xor` and
`strict_binding` never move.

**Second, subtler error:** I built elaborate controls (dimension-matched,
single-timescale, frozen, readout-only, shuffled) and *still* nearly reported a
win. The `mtb − prod` contrast is +4.25pp and looks encouraging. Only the
`mtb − mtb_frozen` contrast reveals that nothing was learned. The lesson is that
**the incumbent comparison is the least informative one**; the control that
matters is the one that removes the mechanism you claim to have added.

## 2. What did the other branch find that was useful?

**grok (`d509878`) — genuinely useful, and it changed how I read my own numbers.**

1. Their information gate measured the productive thing *before* any training:
   fast feature probe 0.484 vs slow trace probe 0.9375 on the same task. That is
   the cleanest possible statement that the information exists upstream and the
   readout basis, not the learner, is what gates it. My own scratch probe found
   the same thing (single decay 0.59–0.68 vs 5-decay bank 0.94–0.96) and I had
   not thought to measure it before training until I saw their framing.
2. Their `slow-zero == slow-frozen` on all 8 seeds is a **stronger null design**
   than a statistically-null contrast: it proves the hidden update contributed
   exactly nothing, not merely that the difference was undetectable. I adopted
   the same logic when reading my own `mtb_frozen` result.
3. Their honesty was instructive: they had a large CI-secure contrast available
   (`slow_correct − fast_correct`, lower bound 0.43) and did not headline it as
   success, because it compares against a near-chance arm. That is the right call
   and it prompted me to check whether my own `mtb − prod` contrast had the same
   defect. It does.

**Where I disagree with grok — and this matters:** their NO-GO hinges on a
**t-interval**. On their same eight seed values, the repository's registered
bootstrap gives a lower bound of **+0.0195** (positive, would pass) while their
t-interval gives **−0.000098** (contains 0, fails). I reproduced both. Their
report does not state which method was used, and with n=8 both are defensible, so
I cannot call it wrong. But because the method determines the verdict, it needs to
be named and justified. I flagged it as `grok-003` and marked it for adjudication
rather than quietly "fixing" it in either direction — including in the direction
that would have passed them.

## 3. What reproduces and what does not

**Reproduces:**
- My P0 suite (30/30) on repeat runs, including the cold-process resume and the
  cross-arm spike-identity check.
- The `mtb`/`mtb_frozen`/`mtb_readout`/`mtb_shuffle` equivalence, and the
  `mtb_only_slow` ≈ 65.89 result that makes the whole multi-timescale story
  reduce to "one slow decay".
- grok's six arm means and headline CI, exactly, from their raw `result.json`.
- The order-3 requirement of `strict_binding` (order-1 0.52, order-2 0.54,
  order-3 1.00, label-shuffled 0.45) — deterministic and repeatable.

**Does not reproduce / was not attempted:**
- My confirmation phase: **never run**, because the screen failed. Not an
  omission — the registered stop rule.
- Retention/forgetting: **not measured**. A model that does not acquire cannot
  supply a meaningful forgetting number, and the master plan explicitly refuses
  "no learning therefore no forgetting" as a pass.
- grok's training script, their tests, and a cold resume on their branch: I did
  not execute them. Their report's numbers I verified arithmetically, not
  procedurally. That is recorded as NOT_VERIFIED in my review.

## 4. What could be combined on mechanism, and at what cost

Being strict about this, because "each part had one good number" is not a reason
to merge anything.

**Worth considering (low risk, diagnostic value):**
- A **pre-training information gate** of the form grok used and I independently
  reproduced. Cheap (seconds at H=48), and it would have saved both of us a
  learning run. It is a *measurement* addition, not a mechanism change.
- A **frozen/readout-only arm as a mandatory gate**, not just a control. Both
  branches pass the 65% acquisition bar with a frozen hidden, which means the
  acquisition gate as written does not discriminate learning from information
  access. Gating on `candidate − frozen` would have failed both hypotheses
  immediately and cheaply.

**Not worth combining:** neither branch's learning mechanism helps. Combining
slow eligibility (grok) with a multi-timescale readout (mine) would produce a
number around 65–70% on `order` that is attributable to neither mechanism's
learning component, and would be indefensible as a capability claim. Explicitly:
**I do not propose merging these.**

**Cost of any real next step:** every additional arm at 3 seeds × 3 tasks is
~1.5–4 s/trial; 297 trials took ~45 min wall. That is cheap. The expensive
resource is *discriminative experimental design*, not compute — which is the
actual finding of this round.

## 5. Risks

- **The dominant risk is statistical theatre.** Two independent branches each
  produced a large, CI-secure-looking number (`+0.506` and `+4.25pp`+) that does
  not survive a frozen-hidden control. If the campaign reports "two of three
  branches improved `order`", it will have learned the wrong thing.
- **`strict_binding` is an invalid gate for this architecture.** It needs a
  third-order feature; an additive point neuron with a linear readout cannot
  express it at any learning-rule quality. Every branch reporting a binding
  failure is partly measuring the task, not the learner. Registering a
  rank-2-expressible binding task is a prerequisite before any branch concludes
  anything from `strict_binding`.
- **Interval method determines verdicts** at n=8. This round had a concrete case
  where bootstrap and t disagree across the decision boundary. Protocol must name
  the interval and the campaign must not allow a post-hoc switch.
- **Vacuous-true gating.** My own rerun reported `screen_pass: True` because
  every gate was `None` and I wrote `all(v for v in ... if v is not None)`. This
  is a generic hazard in pass/fail harnesses; it must be impossible by
  construction.

## 6. Process notes worth passing on

1. **Compute the cheapest discriminating measurement first.** A fixed-feature
   ridge probe on frozen weights answered the central question in seconds; the
   297-trial learning run then only confirmed it. I did this partly by accident
   (design exploration) and grok did it deliberately.
2. **Assert grid integrity before summarising.** My recompute script refuses to
   summarise unless all 297 registered cells are present exactly once — which is
   how I caught that `progress.json` lagged the final batch.
3. **Distinguish model bugs from test bugs explicitly.** Five P0 "failures" were
   wrong references in my *tests* (wrong `feature_decay`, wrong zip member, RNG
   stream not shared, a ULP tolerance). Changing the model to satisfy a broken
   test would have been a much worse error than the original failure.
