# G1 capability + adaptive capacity product loop — 2026-11-09

User authorizes capability improvement, per-turn information load display, and
bounded extra neurons when work needs them.48 is a historical baseline, not a
permanent product maximum. Keep one core, local learning, CPU-first,2GB, legacy
checkpoints/history. No pretrained proxy, BPTT, automatic pruning or deletion of
rare skills. Rewiring/self-organization remain separate hypotheses.

## Registered small capability comparison

- Seeds17,29,43; same48-neuron baseline, same joint UCI digits split2701/2703.
- Conditions: original encoder/full, lexical-balanced encoder/full,
  lexical-balanced encoder/readout-only. The latter tests hidden contribution.
- Lexical encoder: separate normalized word and char-trigram hashes; train-only
  IDF, word80%/trigram20%, contextgain2.4, previouswordgain0.4. No answer lookup.
-60joint epochs/condition/seed, maximum600s/seed, one numericalCPUthread,
 2GBprocesscap. Same schedules/dataset/update counts; each sixth image gets
 one random authored dialogue lesson. Bounded product training, not pretraining.
- Selection: max(mean dev text exact, dev digit accuracy), tie lower dev
 teacher-forcing loss; select on dev only. Confirmation opened once after all
 configuration/seed selections fixed. Original18acceptanceprompts not reused.
- New confirmation:32project-authored paraphrases,4/family,8families. Families
 themselves known: this measures paraphrase generalization, not novel skills.
- Product target: >=15pp paired-seed mean gain in text exact over retrained
 original, no >3pp mean loss in digits; publish seed spread and intervals.
 Failure means retain previous release, no threshold changes. Historical
 forgetting≤3mean/≤5worst remains failed; this experiment cannot pass it.

## Adaptive runtime acceptance

- Baseline weights survive48→64→96→128 max expansion, stable neuron IDs,
  zero new readout columns initially so growth alone does not add skill.
- Actual input/recurrent/readout computation selects only allocated active
  neurons, not full dense compute followed by masking. Core anchors48protected
  neurons until usefulness tests support fewer; new neurons are experimental.
- Controller budget based on observable input workload/modality, with explicit
  per-request override and resource cap. This is a workload heuristic, not a
  scientifically validated difficulty/capacity detector. Never grow because an
  answer is wrong. Reserve neurons need subsequent local learning.
- Dormant states retained with lazy temporal decay on reactivation, no neuron
  pruning; eligibility rows selected from active neurons. Full topology,
  active/allocated/max, clocks,RNG and encoder settings persist/restart/rollback.
- Trace/telemetry: text chars/UTF8bytes, actualimagepixels/bytes, actual nonzero
  inputfeatures, selected/allocated neurons, neuralbufferbytes, resource cap.
  Never call a byte counter remembered knowledge, or capacity expansion a
  demonstrated learning/quality advantage. Compare active compute against
  all-neuron inference and report overhead when noCPUimprovement.
- Existing personal sessions get a checkpoint before any new candidate;
  incompatible data/corrupt checkpoint rejected, no silent newmodel init.

## Data decision

First diagnose encoder/teacher-forcing/interference. Prepare a generation brief
for Antigravity if broader language/relations/skills remain weak. Generated data
must have provenance, grounded/verifiable answers, taskfamily/split IDs, noise
and contradiction handling. Do not auto-submit another app/agent or accept a
`verified` flag as evidence. No fabricated claim of general chat from8families.
