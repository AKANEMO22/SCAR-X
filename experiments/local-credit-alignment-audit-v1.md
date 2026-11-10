# Audit 39 — local credit direction vs actual loss reduction (read-only)

Registered2026-11-10 BEFORE code/measurement. Directly continues
[audit38](word-training-objective-audit-v1.md), which closed with an explicit
open question and refused to answer it: its potential-directions are measured only
at a FINAL fixed checkpoint and are never compared against a real loss change.
Audit38 states verbatim that "Normlarge is NOT usefulcredit or proofcausal
responsibility", and its `DIRECTION_CONFLICT` flag is *descriptive only*.

This protocol answers exactly one mechanism question that the user raised, the
one audit38 left open: **is the local learning direction actually aligned with
reducing the objective, or is it merely large?**

NO neuraltraining/optimizer/headrefit/newlearner/encoderstateviewgrid. No
reopening of STOP35/36/37. No objective/topology/threshold change. No deployment
or scaling. Main ALIF, private weights and Hermes config stay untouched.

## Why this is not a re-run of audit38

Audit38 measured **direction norms and cosines between positions**. Magnitude and
internal self-consistency cannot distinguish "useful credit" from "consistently
large noise". This audit measures the only thing that can: whether applying the
local direction actually **lowers the held-out loss**, and whether that reduction
exceeds what random directions of identical norm achieve.

This is a counterfactual-alignment test, not a performance benchmark. It makes no
claim about language ability, chat quality or lifelong retention.

## Sources / invariants

All15 FINAL36 checkpoints (paired seeds36313/36319/36341 × correct/shuffled/zero/
frozen/BPTTcontrol) plus pinned reviewed 64train/64dev prompts and 8 owned style
labels from dataset SHA
585fd541cdc2651e554e7cd9b7a0ae6d04d0dd5fc0acc1104add773a92864d0b.

Existing actual AGY128 prompts and Qwen64 text references are reused as
**archived text only**; no new teacher/model/API call, no Qwen answer becomes a
label. Historical data, not fresh-final/general-chat/lifelong evidence.

Source checkpoints are read via checksummed `DirectWords`/`ReadoutEligibility`.
Every source file SHA256 is re-verified before and after, and every protected
field (core weights/stable/topology/head stats/Adam moments/counters/RNG) is
asserted unchanged. Any direction is applied only to a **deep-copied disposable
model**; the loaded source object is never mutated.

## The measurement

For a fixed checkpoint, a fixed prompt and the gold token sequence, at the
teacher-forced endpoints we already obtain from audit38:

1. `g` = the existing local potential direction from
   `word_training_audit.potential_direction` (unchanged rule: one-step readout
   error × own-neuron forward eligibility, interior clip/scale).
   This is the SAME quantity audit38 already computed; we add no new rule.

2. Apply `g` to the disposable copy's `w_in`/`w_rec` with the checkpoint's own
   existing Adam state and clip, exactly as `ReadoutEligibility._apply_direction`
   would, but **without** touching any counter of the source.
   Then re-measure teacher-forced NLL on the SAME (prompt, gold) pair.
   `Δloss_g = NLL_after(g) − NLL_before`.

3. Controls, all matched on applied norm and all using the same clip:
   * `random` — Gaussian directions rescaled to ‖g‖ (N=16 draws per endpoint).
   * `signflip` — `−g` (opposite direction, identical norm).
   * `shuffled` — the source's own `shuffled` condition direction, where present.
   * `zero` — no-op, verifies measurement determinism (`Δloss` must be exactly0).

4. Also report the **held-out** version: apply the direction accumulated over the
   TRAIN cases only, then measure dev NLL before/after. This is the only variant
   that can speak to generalization, and it is reported with its own seed-level
   interval. Dev labels are used **only for scoring**, never to build `g`.

## Pre-registered quantities

* `ALIGNMENT_FRACTION` — fraction of endpoints where `Δloss_g < 0`.
* `GAIN_VS_RANDOM` — paired `Δloss_random − Δloss_g` (positive favors `g`).
* `GAIN_VS_SIGNFLIP` — paired `Δloss_−g − Δloss_g` (positive favors `g`).
* `HELDOUT_GAIN` — dev NLL change from the train-aggregated direction.
* cosine(`g`, `gradient_of_loss_wrt_applied_params`) is NOT computed: it would
  require a backward pass and is out of scope by construction. This audit cannot
  and does not claim to approximate a gradient.

Every quantity is reported per split, macro over the 8 families, with seed-level
bootstrap CI95% via the existing `phase21.interval`, and Holm correction across
the registered contrasts.

## Pre-registered flags (descriptive, NOT project quality gates)

* `LOCAL_DIRECTION_ALIGNED` if median `ALIGNMENT_FRACTION` > 0.5 and
  `GAIN_VS_SIGNFLIP` CI95% excludes 0 on dev.
* `DIRECTION_UNINFORMATIVE` if `GAIN_VS_RANDOM` CI95% contains 0 on dev, i.e. the
  local direction is statistically indistinguishable from a random perturbation
  of the same size.
* `DIRECTION_CONFLICT_CONFIRMED` only if `GAIN_VS_SIGNFLIP` is negative with CI95%
  excluding 0, meaning the opposite direction would have been better — a much
  stronger and more alarming statement than audit38's cosmetic cosine<0 flag.

If unsupported, report `UNRESOLVED`. Do not tune step size, clip, subsets or
seeds to manufacture an effect; the clip and Adam settings are the checkpoint's
own, fixed before measurement.

**Alarm rule:** if `DIRECTION_CONFLICT_CONFIRMED` holds, the current local rule is
not merely weak but counterproductive, and that must be reported plainly and stop
further spend on this rule family. This would be a diagnosis, not a new learner.

## Scope limits, stated before results

* Only 15 existing checkpoints / 3 paired seeds. 15 conditions are NOT 15
  independent seeds. CIs are exploratory.
* One-step readout-error direction; this does not evaluate multi-step or
  full-trajectory credit.
* Local linear probe of a single update; repeated application may behave
  differently and is NOT tested here.
* CPU1thread, 2GBprocess, 300sround, no Torch/iGPU/offload. Abort on
  source/format/parity/nonfinite/protected-state/budget failure and preserve logs.
* Meaningful tests plus full pytest/native parity before the audit.
* Results publish to localhost as typed **diagnostic** metrics, clearly separated
  from free neural answers; no invented learning loss or ΔW entries.

This audit can only produce a mechanism diagnosis. Any selective-credit or
learning-rule change still needs its own separate protocol and explicit approval.
