# Stopped-reset eligibility — independent confirmation proposal V1

**PENDING EXPLICIT APPROVAL. DO NOT RUN.** User previously prohibited autonomous
fresh confirmation; the current repair/DEV request does not silently remove that
boundary. Candidate and actual results are reviewable in the matched/reset DEV
receipts. This proposal is only P1 confirmation, not P2 or product deployment.

## Frozen candidate and data

- Candidate: `StoppedResetCredit`, multi_gated correct signal, H48/D8/O2/K8/R12,
  same single-compartment ALIF forward and existing local readout-error learning.
  Exact eligibility intervention: e_v<-u_e, e_a/filter/readout/optimizer unchanged.
  Code and hash set must be locked after technical regression, before execution.
- Peer source607ffca7c9222c5e3fd9741c3ebdf5939e10bedf. Same generators and numerical
  parameters as registered DEV; no hyperparameter or benchmark changes.
- Proposed12seed bank:2026111201,2026111207,2026111213,2026111219,2026111225,
  2026111231,2026111237,2026111243,2026111249,2026111255,2026111261,2026111267.
  A pre-proposal scan of all local tracked branch-head source/protocol/test/
  artifact paths found no occurrence. This is not a guarantee about private
  scratch/external experiments; disclose any known exposure before approval.
- Three fixed primary tasks: strict_binding, xor, distractor_binding. Protected
  order is independent regression, never a ceiling substitute for representation.
- Every seed/task:1024train(seed+200),256refit(+300),512ID(+400),512OOD(+500).
  Calibration first32TRAINinputs only. Validate oracle/split integrity before
  training; freeze raw tensors/hashes. No source lookup/labels inside inference.

## Comparators / budgets

Primary-task arms: stopped_correct, own_reset_correct, stopped_shuffle,
stopped_zero, stopped_readout_only. Initial/frozen hidden features are checked
against the readout-only arm (same initial weights, no hidden movement); do not
manufacture independent evidence from identical feature baselines. Keep zero as
a separate implementation control with identical stochastic draw schedule.
Protected order: stopped_correct vs own_reset_correct.

12*(3tasks*5arms+1protected*2arms)=204jobs. Primary exactly1024episodes; secondary
same fixed0.5trainingCPUsecond ceiling and max4096cycled episodes from DEV.
Record timer granularity/overshoot, actual applied norms and counts. At most
204*(1024+4096)=1,044,480source update calls, not new samples or language tokens.
No new data generation service, Qwen, AGY or paid API.

Measured DEV cost was~9seconds/job; forecast~31minutes plus validation/I/O, not a
guarantee. Register global60minute wall ceiling,120seconds/job,1numericalthread,
one job at a time,2GB per-process cap + sampled full-owned-process-tree RSS/
commit. Do not increase limits or discard slow/bad seeds after seeing results.

## Statistics fixed before opening bank

Seed is the independent cluster, not each512question. Bootstrap10000seed-level
resamples, RNG919, percentile95%CI. Exact paired two-sided sign-flip over4096
sign vectors. Holm correction across **7 primary tests** on equal-update ID:
1–3 stopped_correct vs readout-only for the3primarytasks;
4–6 stopped_correct vs stopped_shuffle for the same3tasks;
7 stopped_correct vs own_reset_correct on XOR (the specific DEV improvement).
Initial/frozen and zero are checked/report separately; no duplicate statistical
tests for mathematically identical baselines. Secondary CPU/OOD results are
descriptive except the predeclared non-inferiority protections below.

Why12rather than8: for7two-sided exact tests,8clusters have minimum attainable
Holm p=7*(2/256)=.0546875, so a full-family claim could not meet.05. Twelve gives
adequate permutation resolution without changing alpha or success thresholds.
This resolution argument is not a power guarantee or permission to increase N
after a negative result. The master minimum remains>=8fresh seeds.

## Gates / stopping

Master P1 requires>=2/3validprimarytasks with lower95hidden/refit gain>=5pp,
mean direct>=65%, positive readout-only benefit with corrected testing and
valid causal controls. Require candidate-shuffle contrast lower95>0 and Holm
p<=.05 on every task counted toward the2/3. Same data/normalizer/source and no
hidden update for zero/readout. All per-seed predictions and intervention scores
are published, not only means. No ceiling exemptions or absent-evidence PASS.

For protected order, lower95paired direct difference candidate-own>=-3pp and
candidate mean direct>=65%. On any task counted as successful, OOD direct/probe
differences vs own-reset must have lower95>=-3pp. Preserve the failed third task
as a limitation; two successful synthetic tasks do not establish language or
general robustness. The specific new reset-credit superiority claim on XOR
additionally needs positive lower95and primarytest7 Holm p<=.05.

P2gates remain acquisition/final>=65%, meanforget<=3pp/worst<=5pp EVERYseed, but
are NOT tested or granted by this confirmation. No main/Hermes/live import even
if P1passes. No model/neuron/compartment scaling or language training.

Engineering/NaN/invalid-oracle/persistence/memory/locality failure stops and
preserves partial evidence. A measured FAIL ends this confirmation; no tune,
re-label, seed replacement or rerun of an exposed bank as fresh. Any corrected
replay/new bank requires a separate disclosed decision. P1pass only authorizes
a proposal for the next research gate, not a production success declaration.
