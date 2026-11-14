# Hybrid local credit direction diagnostic — before outputs,14/11/2026

Parent032f30 acquisition/retention and commonprobe813109 failed qualification.
This is a bounded diagnostic, no new training, tuning, confirmation or promotion.
Question: how closely do the current hidden local direction and its actual
momentum/preconditioned/clipped prospective update align with an isolated
full-temporal surrogate-gradient control on the SAME observed ALIF trajectory?

Prior art: official authors' e-prop repository describes separate numerical
factorization and hardcoded/autodiff verification scripts:
https://github.com/IGITUGraz/eligibility_propagation/blob/master/README.md .
This study is not reproducing their theorem, implementation or benchmark.
SCAR-X uses heuristic neuron-specific gamma/normalization/momentum; approximate
alignment is a diagnostic, not a requirement that all useful local rules be
exact BPTT. Surrogate gradient is not a true derivative of discontinuous spikes.

Select first positive and first negative training case in each parent family,
source order:8ownedquerypairs, fixed before neural outputs. Three parentseeds
260113/260137/260171, two states: initial and currenthidden final640updates.
Same D64/H128/16ticks/FP32/topology, original unmodified readout/statistics.
No finalDEV labels used to choose cases or solve an optimizer.

For each48cases, actual LocalALIFCUDA forward with tracking; archive probability,
rawfeatures, strictclipmask, local e_f direction, computed prospective hidden
delta (same .9trace/.99energy/clip/weightclip), and exact spikes/state tape. A
second reference rollout records tape; compare with authoritative forward.
Local protected/state buffers, RNG/replay/counters must remain unchanged.

All backward code lives ONLY in evaluation/controls.py, separate subprocess
using existing reference-env TorchCUDA. Reuse WordTapeSurrogate's observed spike
forward and matching logistic slope; compute full temporal surrogate derivative
of the same masked-readout CE through recurrence/reset/adaptation. No optimizer,
parameter update, learned weights import into serving or main. Control uses
FP32 CUDA, native VLM untouched. Reference and Torch final v/a/z/f within3e-5
(feature1e-6), probabilities/loss within1e-5, else stop and keep failedreceipt.

Report full vectors/cosine/signagreement/norm ratio and input-credit magnitude
for query first8ticks vs record last8ticks, perstate/seed/family. Metrics exclude
numerically zero coordinates with fixedabsreference>1e-8; zero norm gives null,
not invented perfect agreement. Interpret under surrogate/tape approximation;
gradient agreement or disagreement alone cannot explain all generalization/
forgetting or prove a replacement learner would succeed. No PASS threshold or
retuning of prior65%/3-5pp/5ppgates.

Max48backwardpasses,96referenceCUDArollouts,5minutes/stage; initial physicalfree
VRAM≥1800MiB and≥512MiB during stages. Stop on source/parity/persistence/resource
failure, keep partial output. No new dataset downloads, libraries, model/LoRA,
service restart, active8h source edit or live labels. Publish actual diagnostic
metrics on8775 actororchestrator, explicitly separate from neural learning.
