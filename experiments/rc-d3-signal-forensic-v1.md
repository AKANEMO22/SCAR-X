# RC-D3 v1: saved-array learning-signal forensic

Locked before execution, 2026-11-16. This is a data-only diagnostic, not a new
learner, training replication, neural forward, probe fit, or change to G2 FAIL.
The original P1b TEST is sealed and must never be opened or hashed by this job.

## Scope and provenance

Immutable source worktree: `scarx-scope-alif-design-20261115`, expected checkout
`1ea065be194d44000ba795138c5917f3d13adda0`. Historical registration source commit
`a0f5ac4c1a146bd07a554c6ed32919dc8a0ed8fb` is a separate provenance fact.
Run: `20261115T174514Z-scope-p1b-v1-b30f0b/learning`. Analyze all 24 registered
cells, seeds 261161/261167/261173, signs +/-1, label/teacher/shuffled/readout,
192 updates each. No cherry picking and no new model seeds.

Inputs are the original registration, completed records, data-lock metadata,
registered source snapshots; each TRAIN/DEV fit archive; each cell's saved
learning diagnostics, learning CSV, initial/final checkpoint, and saved DEV
predictions. The analyzer does not import any learner/model code. Only explicitly
named allowlisted files are read; there is no recursive run-directory traversal.
Hash input files and source/protocol/config before metrics. Outputs use a new
exclusive directory and terminal status is atomically replaced.

## Questions and locked computations

1. Reconstruct label component `L = ((Wout.T @ (onehot(y)-p))/.1) *
   1[abs(q)<5] * E_endpoint`, mask rows0:8. `H` is the saved hint bank. Compare
   saved `D` to `L` for label/readout and `L+H` for teacher/shuffled.
2. For every update, measure L/H/D/E/physical-delta norms; cosine L:H,
   D:delta, H:delta, L:delta; H/L norm ratio; sign conflicts on jointly active
   coordinates; nonzero fractions. Active means absolute value >1e-12.
   Undefined cosine/ratio is null, never coerced to zero. Norms and dot products
   use float64 over rows8:64 and all four registered columns (224 coordinates).
3. Reconstruct momentum `.9M+.1D`, energy `.99V+.01D^2`, normalized proposal
   `.008M/(sqrt(V)+1e-4)`, scalar clip +/-.01, weight bound +/-3. Compare stored
   optimizer state, physical delta, next saved weights and final checkpoint.
   Readout arm retains M/V and has zero actual hidden delta. Quantify clipping
   among active proposal coordinates, actual:D sign disagreement, momentum:D
   conflict, bounds, and endpoint feature clipping. Recompute head and bias
   physical updates and their clipping/bounds from saved arrays.
4. For teacher/shuffled only, compute an instantaneous, frozen-pre-state
   omission diagnostic: replace D by L for one arithmetic proposal, retaining
   that update's saved preceding M/V and weights. Measure its physical delta's
   difference from the actual proposal. This never changes a state, fits, or
   rolls forward. It is not the hypothetical trajectory of label-only training.
5. Pair B/C by seed/sign/update. Compare saved hint bank, eligibility, label
   component, direction, physical delta and pre-update weights with cosine and
   symmetric relative L2 `||x-y|| / ((||x||+||y||)/2)` (null if denominator zero).
   Step1 shares initialization; later differences include diverged histories.
6. Recompute final DEV confusion/accuracy/positive recall from saved p/y, solely
   to describe associations with the signal trajectories. Summarize every cell,
   sign/arm, class, and fixed update windows1:48,49:96,97:144,145:192. No p-value,
   seed-population confidence interval, post hoc gate, or architecture claim.

Numerical audit tolerance is 1e-5 for direction, optimizer, physical/head/bias
updates and CSV norms/loss, with 192 records and exact frozen-coordinate and
initial/final invariants. Tolerance is an audit check, not learner modification.
Metrics are descriptive; three independent topology/model seeds do not become
24 independent replications. G2 FAIL, G3/G4 not run remain unchanged.

## Missing information and interpretation limits

Saved eligibility is endpoint-only, and saved q is clipped endpoint-only. Full
student time-series saturation, individual-tick modulation, decomposition of H
over time, and causal responsibility for failure are NOT MEASURED here. This
job does not replay the literal ALIF equations used by the old auditor. A
direction difference is not evidence of useful information or improved learning;
cross-trajectory B/C comparisons are not same-state interventions.

## Resource and stopping policy

One CPU process, BLAS/OpenMP/NumExpr threads=1; GPU forbidden. Active process
wall budget60s, sampled RSS<300,000,000 bytes, all bookkeeping/I/O included.
Check budget per trial and per update, save failure and partial outputs if any
limit/integrity assertion fails. No rerun with relaxed thresholds or extra seeds.
Output resource receipt reports elapsed time, sampled process RSS and OS peak
working set where available, no unsupported claim of a hard RSS cap. This same
Codex-authored independent arithmetic implementation is not external peer review.
