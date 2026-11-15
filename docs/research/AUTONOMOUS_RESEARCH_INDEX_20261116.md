# Supplemental research index — 2026-11-16

Audit level: same Codex system, source review and bounded data-only recomputation;
not external peer review or reproduced training. This supplements the historical
121-entry catalog; its classifications do not promote every catalog entry.

Immutable source: `1ea065be194d44000ba795138c5917f3d13adda0`, clean worktree
`scarx-scope-alif-design-20261115`, branch `research/scope-alif-p1b-20261116`.
New worktree `rc-1116`, branch `research/autonomous-reusable-20261116`.
Ponytail discovery found no enabled tools. No main/live/Hermes changes, push,
merge, historical rewrites, neural execution, or locked TEST access.

## D4 data-only protocol, locked before execution

Question: do saved DEV prediction equality counts and saved credit vectors
support their reported arithmetic, and which historical mechanisms are reusable?
Use all 375 ACR DEV records, seeds 140011/140017/140023, and all 48 saved Hybrid
training-case vectors, seeds 260113/260137/260171, two states. No arm/seed selection.
No fitting, model loading, neural forward/backward, updates, bootstrap tuning,
confirmation, or TEST input. Script: `scripts/research/rc_d4_archaeology.py`.
One CPU process, one numerical thread, no GPU, <=60 seconds and sampled RSS <=300MB.
STOP on hash/count mismatch, time/RSS budget, dirty/wrong source, or existing output.
Output must be a new `runs/rc-d4-*` directory; record hashes and resource receipt.
Equality error <=1e-12 and cosine error <=1e-12 establish arithmetic agreement only.
No new uncertainty claim: paired records share three seeds, not 375 independent trials.

## Navigation and current classifications

| Evidence | Classification in this supplement | Reuse and limit |
|---|---|---|
| Catalog/archive file integrity | VERIFIED | Existing CLI inspected and list/search replay/show CAPACITY/verify executed: 121 entries,333 report references,227 immutable copies; no model calls. Hashes verify files, not scientific claims. |
| Grok R4b/R5 readout scaling | AUTHOR CLAIM ONLY for historical numbers here | Archived source/report exposes variance-floor compression of small informative coordinates and fit/order sensitivity; no raw rerun in this audit. STOP survives. |
| Grok R6 / R5B reproduction | PARTIALLY VERIFIED | Prior Codex forensic reproduction saved raw pairs and reports head caseA~66.76% vs ridge~98.58%, no-clip CI crosses zero; this audit does not reopen confirmation. Use the readout diagnostic distinction, not a candidate success. |
| DeepSeek R4 | PARTIALLY VERIFIED | Source and archived protocol/report inspected; confirmation FAIL and posthoc diagnostic extensions preserved. Fixed bilinear cue interactions are programmer supplied; no learned-hidden or general binding proof. |
| DeepSeek R5 | PARTIALLY VERIFIED | Source/report and prior forensic provenance support STOP/caseA confirmation failure. Three DEV screen seeds did not predict eight historical confirmation seeds. No new old-TEST measurement. |
| ACR DEV raw arithmetic | IN PROGRESS | Pending bounded script below. Three DEV seeds,375 model records incl reuse; isolated task models/joint IID do not measure continual composition. Source and final archival SHA differ and must be named separately. |
| Hybrid Credit Direction vectors | IN PROGRESS | Pending saved-vector cosine recount below. History/preconditioning is an arithmetic location of direction change; no causal proof that removing history improves quality/retention. |
| Phase2/2.1 | PARTIALLY VERIFIED | Prior15-trial audit reports head restoration rescues task0 much more than hidden restoration; report/source navigation inspected, historical matrices/TEST not rescored. Acquisition/retention gates remain FAIL/PIVOT. |
| Shared head / common readout | PARTIALLY VERIFIED | ACR joint guards fail binding/distractor/order; GeminiR11 shared IID is not continual retention. Hybrid common-head DEV reports weak/variable earlier gains and conditional later signed-credit gains; neither proves H3/H5. |
| Semantic common-head raw recount | NOT MEASURED in this audit | Saved measurements exist, but feature labels.npy contains600 TRAIN/DEV/TEST labels. Avoid loading this mixed archive. Existing scoped DEV audit can be reused with its limitation. |
| R13 legacy / R13.1 claims | INVALID / RETRACTED | R13.2 erratum explicitly retracts R13.1 zero-shot claim (B saw composed supervision), target leaking prompt, substring scoring, unfair one-hop control and hardcoded measurements. Distinguish legacy f4272ba from ffddbca/R13.1. |
| R13.2 at4e01834c | PARTIALLY VERIFIED | Protocol/source/erratum reviewed; reuse prior equality recount in operator-lattice source receipt only (fixed spike19.92%,aligner33.40%,symbolic98.05%,4 seeds). No new raw TEST opening, causal-generation audit, or training reproduction. Natural neural interface/H3 not established. |
| R13-family current process status | NOT MEASURED | Catalog IN_PROGRESS is a dated snapshot; refs exist for R13.1/R13.2, not evidence a process still runs. Protected worktree/process/cache untouched. |
| Reuse improves acquisition / retention after new primitives | NOT MEASURED | No audited H5 evidence; checkpoint persistence, probe performance and symbolic lookup do not establish it. |

Key paths: `docs/research/forensic-r6-20261111/R5B_R6B_REPRODUCTION.md`,
`docs/research/ACR_DEV_RESULTS_20261112.md`, `ACR_CREDIT_SPACE_LIMITS_20261112.md`,
`docs/hybrid/CREDIT_DIRECTION_DIAGNOSTIC_20261114.md`,
`docs/hybrid/HIDDEN_BINDING_AND_COMMON_PROBE_20261114.md`,
`docs/hybrid/LOCAL_CREDIT_AND_RETENTION_20261115.md`,
`docs/research/08_PHASE21_RESULTS.md`,
`docs/research/literature/sources/operator-lattice-review-20261115.json`.
R13.2 protocol is in Git object4e01834c at
`docs/research/R13_2_PROTOCOL_AND_ERRATUM.md`; raw remains unopened here.

## Reuse decisions and gaps

Keep: common teacher-blind readout, raw-feature/scale and margin diagnostics,
matched physical delta controls, causal hidden reversion, counterfactual query
pairs, strict seed accounting and intermediate-state neural-interface checks.
Do not rerun historical failed families merely to rename or tune them.
Missing: robust readout-sign teacher specificity, causal quality effect of
history removal, natural representation compatibility, held-out composition,
faster new-skill acquisition, retained primitives under continuing learning,
and practical advantage over strong small-LLM/RAG baselines.

## Inventory limitations

Read-only Git inventory found11 worktrees,14 local branches and preserved archive
tags. R13.2 ref is4e01834c; P1b SHA is local and must not be called pushed.
`gh pr list` could not run because `gh` is unavailable in PATH; no complete remote
PR inventory was established. No network mutation, model download or clone.

## D4 result receipt

Pending root source/protocol lock and execution. Append only the new unique run
path and scoped arithmetic result here; historical reports/catalog stay unchanged.
