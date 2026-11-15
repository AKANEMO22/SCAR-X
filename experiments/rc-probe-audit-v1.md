# RC P1b probe audit v1 — data-only independent-path algebra

Registered before this auditor executes. Target is the new completed D1/D2 run
selected by explicit `--run`; it is not permission to extract more features or
train hidden. Same Codex system; no external peer-review or training replication.
P1b G2 FAIL, G3/G4 not run, and original custody bytes remain immutable.

Question: do saved raw features, allowed TRAIN/DEV labels and frozen checkpoint
metadata support the D1/D2 probe arithmetic and reported extraction provenance?
Include all three seeds261161/261167/261173, both signs, all four arms, initial
and final hidden keys, both prespecified f/q probes; no selection by scores.

Inputs: new `registration.json`, `result.json`, extraction/job receipts,
`features/*/features.npz`, saved probe coefficient/score archives, D1/D2 JSON;
original committed P1b custody manifest/completed/G2/result and the original
checkpoint arrays needed for hidden identity or frozen online-head algebra.
From fit-N archives materialize **only train_y and dev_y**. From saved prediction
archives materialize **only dev_y and dev_p_u192**. Never materialize train_H,
teacher weights/identity, sealed TEST, original TEST/confirmation arrays or H.
Hashing immutable archive bytes does not materialize their unused array keys.

Independent coefficient route: float64 TRAIN-only mean/std, floor1e-6; same
class weightsN/(2Nk), targets+/-1, lambda1 and unpenalized intercept as D1.
Build sqrt-weighted TRAIN rows and64unit ridge rows; solve augmented weighted
least-squares with `np.linalg.lstsq`. Do not call the extractor's ridge routine
or use its normal-equation solution as the auditor's coefficient. No alternate
lambda, normalization, threshold, variant or score-driven choice. Count every
fixed-problem algebra solve separately; these are verification recomputations,
not additional candidate head fits or evidence from additional seeds.

Recompute scores, confusion, accuracy, balanced accuracy, both recalls,
Mann-Whitney AUC from average tied ranks, mean/min signed margins; scores with
abs<=1e-12 become zero, positive iffscore>0. Compare normalization, coefficients
and scores atol/rtol2e-7; scalar metric comparisons atol/rtol1e-7. Confusion
shape and values are checked. Recompute the registered adverse-teacher
diagnostic from every seed: balanced accuracy>=.75, each recall>=.65, balanced
gain>=.10 over direct. No project-quality approval from probe success.

D2: raw float64 centered SVD rank at standard maxshape*epsilon*largest-singular
tolerance; singular values, raw variance, exact byte endpoint collisions,
class-centroid distance, silent units from saved all-tick spike counts, and
saturation from saved all-tick aggregates with denominatorN*32*64. Check count
integrality/range and q=clip(f/.1,-5,5). Information comparisons atol/rtol1e-6;
q algebra atol1e-6. Tick counts are aggregate verification, not independently
replayed spike tapes; raw per-tick tapes are absent from these artifacts.

Verify source Git blobs match registration; executed source snapshots match raw
working-byte hashes and committed bytes after canonical CRLF-to-LF comparison,
as preregistered by D1/D2 provenance-only recovery. Both raw hashes remain
preserved; no source bytes are rewritten by the auditor. Original inputs match pinned custody
at1ea065be; hidden keys match checkpoint Win/Wrec/indices/config; extraction
receipt hashes match features/checkpoints/data and zero-hidden-update receipt.
Disallow a hidden key shared across different seed datasets. Reconstruct frozen
online probabilities by matrix multiplication of saved DEV q and checkpoint
head, compare saved DEV probabilities atol1e-6. This is data-only readout algebra,
not neural forward. Hash all inputs again after audit. Source/hash checks do not
independently reproduce neural extraction or prove every runtime counter.

Budget: one CPU process, forced one BLAS/OpenMP thread, <=60seconds and sampled
RSS<=300,000,000bytes, no GPU. Cooperative checks before inputs and after each
solve; brief RSS peaks may be missed. No neural/model imports, forward/backward,
hidden updates, optimization, installs/downloads, live/main actions or retries
with substituted seeds. Do not recompute conditional bootstraps in this audit.
STOP on scope/provenance/count/arithmetic/resource mismatch and preserve failure.

Command after root commits/locks these files:
`python scripts/research/rc_audit_p1b_probes.py --run <completed-D1-D2-run> --output <new-runs/rc-audit-p1b-unique> --seconds 60 --max-rss-mb 300`

Target announced before auditor execution: run
`runs/20261115T192506Z-rc-d12-v1r1-73f7fe`, extraction source
`73f7fe733e6fd6a63182a4e94b2fb3d0529c42f5`. Prior intake-only abort192329
is preserved separately with zero forward/fit; this auditor does not rescue it.

Output is a new unique audit `result.json` with every read input hash, errors,
recomputed metrics, counts, explicit limitations, resource receipt and preserved
FAIL/PIVOT. Never write into the source D1/D2 or historical run directory.
