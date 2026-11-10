# P1 optimizer/coverage factor audit v1 — isolated controls only

Registered2026-11-09 BEFORE measurements. Follow-up to local-classifier v2's
stopped family, NOT another local-classifier candidate or a third adjustment.
MasterP1 requires matched learnability controls. Prior control changed credit,
optimizer and updated-row coverage simultaneously; this audit isolates the
last two while keeping task loss/BPTT fixed. No candidate, deployment, larger
model, language corpus, changed success gate or new local rule.

## Locked design

Fresh development seeds13217,13219,13229. Same order/XOR/strict_binding ALIF48,
K8,R12,D8,O2 and Configdefaults(anchor0). Dataset offsets: unlabeledcalibration
+5200000(128), train+5300000(2048), refitfit+5400000(128), dev+5500000(256),
OOD+5600000(128order/XOR; binding notsupplied). No old test used for selection.
Shared initial weights, inputorder, split, calibration and normalized main
readout for all conditions. Main head uses existing local clipped delta(.08).
All hidden learning in this audit uses task loss through time and neuron
connections ONLY inside evaluation/controls.py. Never import into serving.

Five controls per seed/task =45trials:
1. Readout-only (hidden frozen).
2. Hidden BPTT + Adam.002 + all48rows.
3. Hidden BPTT + Adam.002 + rotating12rows.
4. Hidden BPTT + SGD.002(no momentum) + all48rows.
5. Hidden BPTT + SGD.002(no momentum) + rotating12rows.

Adam defaults same as installed Torch/v2control. Coverage starts at
`(episode*12)%48`, selects consecutive12rows, wraps. Compute full BPTT gradient,
global gradientnormclip1 BEFORE row masking. Clear inactive gradients; after
optimizerstep restore inactive weights AND vector moment states exactly.
Adam scalar step/bias-correction clock advances globally; disclose this calendar
time convention, not per-row Adam. Final perweightupdateclip.025,weightclip3.
Allrows branch is exactly previous control's update math. No extra inner steps,
norm-matched/LR searches or candidate retuning after opening data.

Same2048examples,1headupdate/episode and2048optimizer calls for unfrozen
conditions. Hidden synapse writes768/episode(all48) or192/episode(rot12).
Full graph/gradient is still computed for both; rotating is a weight-coverage
diagnostic, NOT evidence of the memory/time saving a local rule would achieve.
Equal numerical LR does not imply equal effective update magnitude between
SGD and Adam; report gradient norms, ΔW, clipping and optimizer bytes.

## Analysis fixed before data

Primary paired direct-accuracy contrasts per task:
Adam-all minus SGD-all; Adam-rot12 minus SGD-rot12;
Adam-all minus Adam-rot12; SGD-all minus SGD-rot12.
Also report same contrasts for refit-probe gain, main loss, compute and ΔW.
Seedbootstrap95%/exact signflip/Holm over12primary contrasts; only3development
seeds means coarse statistics, no confirmatory success claim. No selection of
a winner, test reopening or follow-on budget increase in this audit.

Interpretation categories predeclared: direct effect>=5pp is operationally
material in this budget, but NOT a research gate pass; effects<5pp called small
in this audit. Compare empiricalCI and multiplicity p, do not declare absence
of an effect from nonsignificance. Interaction/distributional differences
reported as limitations, not proof credit alone is the remaining bottleneck.
Negative/positive results apply to these optimizers/LR/forward/dataset only.

Frozen controls must match NumPy probabilities/hidden states. Rowmask tests
must prove inactive weights/moments do not drift; defaults preserve old tests.
Save final weights/head, every target/prediction and neural state sample. No
optimizer-resume or lifelong retention claim for control checkpoints.

Publish real loss/ΔW/counts every512examples and first3actual heldout tensor
predictions per trial to localhost, explicit BPTT-only scope and missingQwen.
Do not fabricate chat or teacher responses. No service restart/weight import
required during this audit, no Qwen concurrently, no Antigravity data dump to
hide the failed acquisition mechanism. Existing generated data/references kept.

CPU1thread, sequentialcontrols,2GBprocess-treeRAM monitored,180s/trial and
1800sround. Stop on RAM/NaN/forward mismatch/leakage or deadline. Whole-network
backward remains confined to evaluation/controls.py; all live/main code remains
local learning. At completion record DIAGNOSTIC_ONLY and next unresolved
questions, never GO for normalchat/vision or reopening the stopped family.
