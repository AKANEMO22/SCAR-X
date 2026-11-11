# R7 DRAFT — what do recurrent hidden synapses learn?

**PENDING USER APPROVAL. NOT IMPLEMENTED/TRAINED.** Retain one recurrent ALIF SNN,
H48/fan-in8 and existing input/codec. No Transformer, pretrained embedding/proxy,
global/time backward, RTRL, architecture growth or production migration.

## Separate the hypotheses

- R7-A READOUT_ONLY: fixed hidden state carries information and a readout learns
  to exploit it. This may be useful but is not a hidden-learning result.
- R7-B LEARNED_REPRESENTATION: approved local hidden plasticity makes a new feature
  map more useful under a common independent head/refit budget on unseen inputs.
- R7-C CONTINUAL_REUSE: the acquired representation/skill survives interference
  while remaining plastic enough to acquire another skill. This is a separate
  laterP2 stage, not opened automatically by R7-B.

## Source/candidate boundary

Start from approved proposal30 own-neuron eligibility + one-step readout-error
feedback. Identify a specific measured credit/geometry defect first; do not
reopen STOP35–38 solely by changing seeds/corpus/learning rate. A stopped-reset
surrogate, new auxiliary error, Oja inside hidden, new gating state or neuron
model is a new mechanism and needs its own explicit approval. No such candidate
is silently supplied by this draft. No scientific run starts with the candidate
algorithm left unspecified.

## Required design before approval

Tasks: audited independent two-key binding; temporal order/XOR controls; a
registered unseen-combination/overwrite variant. Use at least two valid tasks,
including XOR or strict binding, to retain the masterP1 criterion. Validate
shortcut/bag baselines and absence of target/task-ID inputs first.

Arms, same forward/input/budget: initial hidden, hidden frozen+head learning,
readout-only, correct local feedback, zero feedback, neuron-shuffled feedback.
Use common random consumption and optimizer state. If a matched magnitude arm
is required, match the **applied** step after clipping/optimizer, not raw direction.
An isolated approved BPTT control, if needed, stays in evaluation/controls.py;
never import its weights or use it as a candidate.

After acquisition, freeze each hidden snapshot. Fit the **same fresh head** on
the same independent labeled fit set with identical capacity/update budget and
initializations. Evaluate untouched held-out inputs; no target, source ID, answer
mapping, retrieval or previous gold prefix enters neural inference. A trained-
head lesion alone is coadaptation evidence, not a common-refit representation gain.

Interventions: return hidden weights to initial while preserving refit procedure;
shuffle/zero meaningful input; permute event order; reset state; compare new
combinations and query transformations. Record actual feature and weight hashes.
Separate working-state memory, readout optimization and persistent synaptic change.

## Proposed locked P1 decision

Keep masterP1: direct≥65%, lower95% refit-probe improvement≥5pp on at least two
valid tasks including XOR/strict binding; correct−frozen positive with registered
paired signflip/Holm.05. Additionally report correct−zero/shuffled, every seed,
all task regressions and whether hidden gain transfers to unseen combinations.
Do not let order-only100%recall substitute for this conjunction.

Provisional development seeds181301/181307/181313 and confirmation
181401/181407/181413/181419/181423/181429/181431/181437. Recheck full branch/sample
lineage before locking; none generated/scored by Codex. Candidate/fixtures/
statistical comparisons and cost must be committed before opening them.

## R7-C / P2, separate later approval

Only after useful P1representation has independent evidence:20stage boundary-free
changing mixture, no task marker given to learner, verified labels only. Compare
bounded replay, readout protection, importance/metaplasticity and consolidation
under matched memory, update count, examples and wall budget. Protecting a head
that never acquired skill is not retention. Do not let replay simply spend more
training compute than its comparator.

Keep existing joint gate on every registered seed: acquisition/final≥65%, mean
forgetting≤3pp, worst≤5pp; measure after restart and delayed task return. Also
measure new-skill acquisition to detect lost plasticity, hidden/head-only overwrite,
and unknown/noisy feedback handling. No P2pilot is authorized by this draft.

## Engineering gate / resource plan

Before training: finite-difference check of the explicitly stated local surrogate
graph, forward parity, no label in features, inactive-state codec checks, exact
partial-segment cold resume, malformed checkpoint rejection, fault-injected
rollback and no private/live writes. Memory includes stable/current weights,
traces, optimizer/importance, normalizer/gate, replay, serialization/rollback copies.

CPU one numerical thread,2GB process-tree cap. Proposed initial screen budget
≤45minutes, confirmation≤3hours after measured preflight projection; if over,
report resource-infeasible and seek a scoped revision before running. Maximum
two approved registered rounds per hypothesis; failed screen stops confirmation.
No scale, language corpus, iGPU/C++optimization or deploy in this proposal.
