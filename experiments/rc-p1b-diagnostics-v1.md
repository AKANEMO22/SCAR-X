# RC D1/D2 — checkpoint forensics, locked before feature extraction/probe fit

16/11/2026. User Autonomous Research Orchestrator directive permits this NEW
diagnostic. P1b G2 FAIL is immutable; no hidden learning/TEST/G3/G4. Existing
21-ish distinct hidden states, not 24 new model seeds. Seed groups remain
261161/261167/261173; both signs and all four arms, no subset selection.

Hypothesis: adverse-sign teacher S may contain linearly usable endpoint
features that its co-trained online head does not use. Diagnose, do not repair.
Prior art: Alain/Bengio linear classifier probes arXiv1610.01644 (paper
abstract/method distinction), previous SCAR-X shared-head DEV tests and P1b.
Linear decoding success is not new raw information or local-learner success.

Only input source: committed P1b run20261115T174514Z-scope-p1b-v1-b30f0b,
48initial/final checkpoints, fit-N.npz TRAIN/DEV arrays and saved DEVpredictions.
Verify bytes against P1b final custody. Never load sealed-test*.npz. Never
read train_H, teacher weights or identity to a probe. Metadata labels/IDs only
for supervised probe fitting/scoring, not ALIF forward. Freeze all hidden
arrays; sourceFP32 ALIF with tape returns raw f, q and spike counts. Deduplicate
identical Win/R/indices/config hidden states within each seed; same features
serve initial/frozen/reverted cases without pretending new neural evaluations.
Actual counts recorded, no learned states fabricated. No backward/autograd.

D1 primary head: binary linear class-balanced ridge, alpha=1, TRAIN-only
mean/std (floor1e-6), unpenalized intercept, targets +/-1, weightsN/(2Nk).
Decision score>0 positive, ties negative. One closed-form fit, no lambda or
threshold selection. Same class/regularizer at all seeds/signs/arms. Algebra
float64 for numerical stability; neural/checkpoint weights stay FP32 unchanged.
Primary raw f endpoint; secondary q=clip(f/.1,-5,5), same fitting procedure,
reported separately, not selected using DEV. Probe coefficients are diagnostic
artifacts, never replace the saved online head. Initial hidden is the hidden-
reverted control; final readout-only with identical hidden shares that probe.

Report TRAIN and DEV accuracy, balanced accuracy, positive/negative recall,
pairwise rank AUC(ties half), mean/min signed margin and confusion. Verify
online probabilities reconstructed from endpoint q and checkpoint head match
the saved DEV p (atol1e-6); mismatch STOP affected study, no fixing model.
D2: rank of centered TRAIN raw features (standard float64SVD tolerance),
variance, singular values, observed silent neuron fraction across all ticks,
q saturation across all ticks, opposite-class exact endpoint collisions,
class centroid gap. Analyze raw vs normalized features, initial vs learned.
Probe failure does not establish zero information or capacity impossibility.

Interpretation preregistered: adverse teacher suggests online decoding issue
if primary common-head on every seed has balanced accuracy>=.75, both recalls
>=.65 and balanced gain>=.10 over direct head. Otherwise mixed/insufficient;
inspect raw rank/collisions and evidence, do not claim absent information from
one linear probe. No project quality GO from this diagnostic. Report all
seed contrasts, conditional paired DEV bootstrap95% with10000draws seed991116
for fixed models; no population qualification at n=3. P1b old gates unchanged.

Resource envelope: D1/D2 <=300s active including profiling, forward extraction,
fits/statistics/audit; <=700MB owned tree during CPU jobs, profileGPUhost<=800MB,
max3CPUworkers, 1BLASthread each. Campaign total concurrent receipts/reservations
<=2GB; D3/D4 each<=300MB, operator neural jobs wait for diagnostics. Available
hostRAM>=4GB, freeSSD>=3GB, CUDA before>=1536MiB and after>=1024MiB NVML free;
no killing or borrowing live service capacity. No installs/downloads.

Backend profile is frozen inference on two actual P1b TRAIN episodes, same
checkpoint, tape/host output transfer/synchronization included, warmup+3pairs
perCPU/CUDA backend (14episodes total), no proposals/updates. CUDA in a separate
short-lived process, pool bytes not full VRAMpeak. CPU/CUDA parityatol1e-6;
select higher measured throughput if safe. Implement CPU data path first; if
CUDA wins, stop implementation before execution unless GPU extractor exists;
do not silently choose a slower backend. Worker calibration1/2/3 uses same
six jobs of64episode inference per config,192? actual6*64=384perconfig,
1152forwards total, include worker startup in elapsed. Choose bestthroughput,
not utilization. Run all real extraction jobs with selected workers (<=3).
Study records global source/config hashes before profiling, every job own
directory/checkpoint reference/log/result/resource; parent alone owns ledger.
Failure preserved, no retry with replacement seed. Runtime/resource violation
stops new launches; only owned research workers may be stopped if needed.

No claim of lifelong learning, teacher competence acquisition, P reuse, new
architecture or SNN advantage vs LLM/RAG from a head probe. No external API,
main/Hermes/live writes, push/merge/deploy or historical output writes.
