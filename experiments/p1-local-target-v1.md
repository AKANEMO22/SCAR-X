# P1 local target-rate eligibility — protocol and signal review

2026-11-09, registered before development/confirmation. Within master P1/P2;
not deployed, not new framework, not a larger model or language pretraining.
The preceding input audit found exact order/color information loss in G1.
This trial feeds full temporal event sequences, which do preserve that information.
It does not claim the current serving text/image compiler has been repaired.

## Candidate, equations and review BEFORE running

Same deterministic ALIF forward as SpikingCore:48neurons,K8,R12,D8,O2.
For a synapse into neuron j maintain only its own forward sensitivities qv,qa.
At tick t, presynaptic activity x is input or the prior source spike:

```
qv_pre = leak*qv + x
psi = 0.3/threshold_j * max(0,1-abs((v_pre-threshold_j)/threshold_j))
e = psi*(qv_pre - adaptation_strength*qa)
qv = qv_pre - threshold_j*e - adaptation_strength*z_j*qa
qa = adaptation_decay*qa + e
elig = feature_decay*elig + (1-feature_decay)*e
```

Clip sensitivities/eligibilities to[-10,10]. Ignore derivatives through the
other neuron's recurrent spike; do not unroll/reverse time or other neurons.
This is an approximate synapse-local sensitivity, not exact network gradient.

At feedback, broadcast the VERIFIED target identity y through a fixed balanced
random target-rate code C[j,y] in{0.02,0.35}. Each neuron computes its OWN error
`m_j=C[j,y]-features[j]`. Update selected synapses by
`clip(hidden_lr*m_j*elig-anchor*(w-stable),+-update_clip)`.
Readout receives its normal local categorical delta. Target identity appears
only in learning, never input/inference. No W_out transpose/error feedback into
hidden, no whole-network backward/BPTT. The fixed code is supervised information,
not purely unsupervised/STDP; disclose that distinction.

Inspiration: target projection [DRTP](https://arxiv.org/abs/1909.01311), local
eligibility [e-prop](https://www.nature.com/articles/s41467-020-17236-y) and
local-error SNNs [DECOLLE](https://arxiv.org/abs/1811.10766). This custom combination
is NOT a faithful reproduction of those papers and has no novelty claim.

Review: signed label broadcast is an explicit third factor allowed by the plan;
all eligibility math stays at the postsynaptic neuron. Local pseudo-derivatives
are used; global gradient transport/autograd/time graphs are absent. Unit tests
must prove readout-weight independence of hidden updates and exact checkpoint
continuation including random-code/shuffled-signal state before experiments.
No inference helper/verifier may provide a target.

## Design

Tasks: existing order and XOR generators, plus strict-binding audit generator
(no bag-of-values shortcut). All binary,12ticks, train2048examples, probe-fit128,
development256, OOD128. Independent fresh network seeds1201,1213,1229.
Generator seeds = network seed +1,100,000train/+1,200,000probe/+1,300,000eval/
+1,400,000OOD; strict-binding OOD reports time-shift variant if supported,
otherwise not applicable (never fake an OOD split).

Fixed config: default Config ALIF except H48,K8,R12,anchor_strength0. Candidate
hidden_lr has TWO settings0.002/0.02. Baselines readout-only and scalar-STDP
use hidden_lr0.002. For each candidate include per-example random target-code
modulation with correct output label. Same forward/data/update budget; report
different compute costs. Frozen/initial hidden evaluated by the same refit head.
Ridge probe regularizer0.01, train-only centering/scaling; never deployed head.

Select candidate LR by mean development probe gain over initial across all
3tasks/3seeds (tie:smaller LR). Freeze selection before opening confirmation.
Screen passes only when gain>=5pp AND direct accuracy>=65% on at least2tasks.
Otherwise stop: PIVOT candidate, no confirmation or larger data/model.

If screen passes, eight NEW confirmation seeds1709,1723,1741,1759,1777,1789,1801,
1811; same data sizes but eval512. Compare selected target rule/readout/scalar/
shuffled at identical updates. Report seed-cluster bootstrap95%, exact signflip
and Holm across3tasks. Representation gate: lowerCI gain>=5pp and direct>=65%
on at least2valid tasks, positive full-vs-readout and adjusted significance.
Do not call a task learnable merely because a simple unrelated recall hits100%.
If learnability control is missing/fails, report that limitation, not impossibility.

Selected inference lesions: permute non-query time steps or reset state each
tick; refit-head and direct metrics reported separately, never choose settings
by lesions. No replay, retrieval, importance or consolidation in this P1 trial.
No retention claim: P2 continual acquisition/retention is required before deploy.

## Resources / outputs

CPU-only, BLAS1thread/worker, at most2workers, per-trial180s/whole round1200s,
aggregate sampled RAM<=2GB and per-process Windows committed cap2GB. Keep
foreground server running independently; no Qwen concurrently in this gate.
Persist target-code, membrane/adaptation, sensitivities, selected rows, synapse
state, learning counters and RNG. Own codec rejects incompatible old readers.

Unique run/source snapshot, per-trial raw outputs, development selection,
optional confirmation, artifact checkpoints and GO/PIVOT/NO-GO. Preserve all
historical runs. At most one later preregistered adjustment after this v1;
two failed rounds stop this candidate family. Main model remains untouched.
