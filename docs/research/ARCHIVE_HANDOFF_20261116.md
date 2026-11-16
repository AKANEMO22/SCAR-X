# SCAR-X research handoff before starting a new project on main

16/11/2026, Asia/Saigon. The user stopped the research campaign and requested
archiving the existing main, then reusing main for a new project. No further
experiments or probe fits were started under this archival task. Research
outputs below are preserved as they stood, including failures and unfinished
work. Main before reset: `37749041ef45d54951558fdeb0c3695616b47cb5`.

## Preserved research status

| Job | Actual outcome | Evidence |
|---|---|---|
| P1b original | G2 FAIL/PIVOT, unchanged; no TEST/G3/G4 | `docs/research/SCOPE_P1B_TEMPORAL_TRANSFER_RESULT_20261116.md` |
| D1/D2 | 21 frozen hidden states, 42 fixed diagnostic ridge fits, 0 hidden updates, 7886 frozen forwards including profiling/calibration | `docs/research/RC_P1B_DIAGNOSTICS_20261116.md`; run `20261115T192506Z-rc-d12-v1r1-73f7fe` |
| D1 initial intake | ABORT at EOL provenance guard, 0 forwards/fits; no data/model/threshold rescue | run `20261115T192329Z-rc-d12-v1-c22e39` and recovery protocol |
| Probe audit | Same-system independent augmented least-squares/metric verification passed, 42 verification solves, 0 model calls | run `rc-audit-p1b-20261115T192828Z-301744` |
| D3 | Saved-array arithmetic audit of4608updates, 0 new learning/forward/fit | `docs/research/RC_D3_SIGNAL_FORENSIC_20261116.md`; run `20261116T-d3-saved-signal-v1-bee9b68c-01` |
| D4 | Scoped recount of1,228,800 ACR DEV entries and48 Hybrid vectors; not all historical experiments revalidated | `docs/research/AUTONOMOUS_RESEARCH_INDEX_20261116.md`; run `rc-d4-20261116T092000Z-archaeology-a0f88a` |
| Operator D0 | STOP_DATA: state00023 has only1 valid changed SWAP outside TRAIN, needs2; 0 neural training/profile/fit | `docs/research/RC_OPERATOR_PREFLIGHT_20261116.md`; run `20261115T193825Z-rc-operator-pilot-v1-a76d1c` |
| Independent operator STOP audit | Source and protocol prepared, NOT RUN when task changed | `scripts/research/rc_audit_operator_stop.py`; `experiments/rc-operator-stop-audit-v1.md` |

Teacher common probe reached100% on DEV for all three favorable-sign seeds;
adverse-sign balanced accuracies were70.83/100/70.83%, compared with direct50%.
That preregistered all-seed diagnostic remains FALSE. Probe accuracy is not
local-learner direct inference. These probes change objective weighting,
normalization and optimizer together; they do not isolate a production fix.
Sample-specific temporal teacher benefit remains unproven against shuffled
teacher. N=3 seed-population uncertainty remains INCONCLUSIVE. No continual,
20-stage retention, restart/reuse, language/video or LLM/RAG superiority claim.

Operator STOP_DATA is a limitation of the fixed data sampling contract, not
evidence against ALIF/GRU composition. Its postmortem files are explicitly
derived from saved data; they were appended after the D0 result, not captured
original primitive-generation traces. Partial primitive rows were not saved.
The separate independent STOP auditor was not executed; do not label it PASS.

D4's run ID is an identifier, not a reliable actual execution timestamp. Its
elapsed time/resource receipt and input hashes are the recorded observations.
No old TEST was reopened to choose a new architecture.

## Source locks and files for resuming

- Evidence base: `1ea065be194d44000ba795138c5917f3d13adda0`.
- D3/D4 lock: `bee9b68c530d174f1bbc28d42cee5e11aebc81a9`.
- D1 recovery lock: `73f7fe733e6fd6a63182a4e94b2fb3d0529c42f5`.
- Probe audit lock: `3017442ac4b357ffc1dde08d58c46148e8eb777d`.
- Operator D0 lock: `6cadbbe2113f94c88b44ecc87678adb5770bb552`.

All old protocols/results/source snapshots remain unchanged. The historical
121-entry catalog has not been rewritten to call these new jobs verified;
the supplemental research index and reports are the navigation for this round.
No new neural operator model or proposed online readout fix was implemented.

Local model/runtime/user data under ignored directories are outside Git
archives. They are retained locally and must not be mistaken for uploaded
weights. Existing branches/worktrees are retained; no force push or branch
deletion is needed to prepare new main. Hermes backend was gracefully stopped
at the user's request and must not be auto-restarted by this handoff.
