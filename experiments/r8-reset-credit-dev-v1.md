# R8 own-reset vs stopped-reset eligibility — bounded DEV V1

Source-informed second investigation, not a new neural architecture. The
six-source map H1_RESET_JACOBIAN_CREDIT identified this exact untested difference
before the current R8 study: SCAR-X uses own-reset derivative terms, whereas the
pinned e-prop implementation stops spike-reset paths in its eligibility
approximation. User requested continued fixes within current SCAR-X constraints.

## Hypothesis / exact intervention

Keep ALIF forward, threshold/adaptation, recurrent topology, readout, optimizer,
data order, gamma[.25,.92], activity gate and credit routing unchanged. Only change
the eligibility recurrence from
`e_v = u_e - theta*z_e - spike*beta*a_old` to `e_v = u_e`.
Keep `z_e=psi*(u_e-beta*a_old)`, `e_a=rho*a_old+z_e` and all later filters/update
equations unchanged. This is an own-neuron local derivative approximation, NOT
full-network BPTT/autograd and NOT a claim to reproduce the complete e-prop paper.

H1 predicts this avoids premature local trace cancellation and improves delayed
binding/distractor transfer versus the same multi_gated own-reset arm. Falsifier:
probe/direct do not improve consistently, or protected order/XOR regress beyond
3pp. A larger trace norm is not itself a representation benefit; report norms,
clipping and both equal-update and fixed-CPU-ceiling outcomes. No LR/clip tuning.

## Fixed screen and budget

Wait for the matched-credit DEV run to finish; validate its complete96jobs and
dataset/source hashes. Reuse its frozen datasets, not regenerate/select samples.
Use the SAME three exposed DEV seeds140011/17/23, four tasks and split sizes.
No fresh confirmation, P2, source/model scaling or language training.

New arms: stopped-reset multi_gated correct, shuffled, zero, readout-only.
48jobs=(4tasks x3seeds x4arms), two budgets exactly as the matched protocol:
1024equal-updates; secondary0.5trainingCPUseconds, max4096cycled episodes.
Own-reset baselines come from the completed registered run, with identical
datasets/configs and no source equation changes on their own-reset path.
Explicitly label reused baselines rather than claiming a simultaneous fresh run.

Primary contrasts per primary task(strict_binding/xor/distractor_binding):
stopped-correct minus own-correct, stopped-correct minus stopped-shuffle,
stopped-correct minus stopped-readout. Exact3-seed bootstrap27draws and two-sided
sign flip, Holm across9contrasts. Order is protected descriptive regression.
Save per-seed predictions, oracle labels, applied update norms, timing, arrays
and full checkpoint state. No label or event oracle enters neural inference.

Three reused seeds cannot establish master P1 or independent confirmation. Keep
representation lower95gain>=5pp on>=2/3validtasks, direct>=65%, adjusted contrasts
and eventual>=8freshseeds unchanged. If this trace intervention fails, mark it
STOP/PIVOT, do not sweep reset coefficients or use a new name to repeat it.

## Engineering gate / limits

Before training: identical forward/no-update state, trace change isolated to the
declared derivative, zero/readout weights fixed, signal controls trace-matched,
snapshot/next-update exact and legacy own-reset numerical parity retained.
CPU1numericalthread,2GBprocesscap + sampled tree, one job at a time,120s/job and
30minutes/run. Preserve all failures; do not restart because polling timed out.
This screen does not authorize changing main, live model, Hermes, compartments
or neuron count. It is one bounded hypothesis, no promise that it will succeed.
