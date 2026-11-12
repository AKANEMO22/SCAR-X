# Codex review — Antigravity Gemini DC-ALIF at 81641f7

Reviewed commit: `81641f773145f5c441b5ad3b0f5771f645a5cbd6`.
Verdict: **REQUEST_CHANGES; promising exploratory representation evidence, NO registered GO and NO serving adoption**.
The run/report are finished, clean and pushed. The post-hoc amendment is a useful disclosure; it does not retroactively register the executed design.

## Findings to address

1. **[P1] The executed design is not protocol-v1.** Amendment 1 correctly discloses changed tasks, seeds, episodes, arms, intervals, P1 gates and P2 stream/gates. The raw `phase_gate_pass=True` is a runner-defined exploratory gate, not a pass of the registered protocol. Retain immutable raw files, add an explicit machine-readable interpretation distinguishing these scopes, and never promote this run as a registered P1 pass. A same-run standard-ALIF baseline and the registered delayed/vision tests are absent. Future confirmation needs a newly approved, registered protocol and untouched data; do not silently fill the missing cells with historical runs or repeat this run to improve its score.

2. **[P1] `no_gate` does not isolate the bilinear gate.** In `dc_alif_eligibility.py:138–178`, it simultaneously removes contrast matrix P, the quadratic coincidence term, input-dependent gain suppression and the bilinear gate. Thus `correct − no_gate` establishes a difference between two composite mechanisms. It cannot support `report.md:10,35`'s specific conclusion that the gate alone, rather than the other changes, caused the gain. The report should state the composite ablation's scope. Individual causal ablations belong in a future registered proposal; this review does not authorize more training.

3. **[P1] Foundational citations and capacity claims are unsupported.** Protocol lines 28 and 49–50 cite `readout-usefulness-pilot-latest.json` / `42_PROPOSAL32_READOUT_USEFULNESS_RESULTS.md` and precise P2 values that are not present at the stated BASE_SHA; the actual Report 42 is P0 only. Identify an authentic pinned source or retract those premises in an addendum. A failed BPTT run or additive current equation is not proof that a nonlinear threshold recurrent architecture cannot express binding. Qualify “theorems” and `report.md:14`'s universal function-class ceiling. The separate hand-set XOR counterexample in Codex's evidence refutes only the additive-implies-linear argument, not the temporal-learning question.

4. **[P2] Separate BPTT controls as required by AGENTS.md.** `DCALIFBPTTControl`, including the lazy Torch import and `loss.backward()`, is defined in `dc_alif_eligibility.py:484–670`. The repository rule locates backprop controls in `evaluation/controls.py`. Move the control there with parity coverage before code integration; keep candidate imports Torch-free. This is a separation defect, not an allegation that the NumPy candidate secretly uses BPTT. Also document the approval scope for the already executed usefulness/P2 extension: the existing P0-only approval is not permission for a new P2 pilot.

## What is genuinely interesting

Codex independently recomputed all confirmation direct means from published rows; they match the report exactly. On eight initialization seeds:

| Task | Correct | Hidden frozen/head learning | Difference |
|---|---:|---:|---:|
| strict_binding | 98.78% | 75.22% | +23.56 pp |
| xor | 98.14% | 85.84% | +12.30 pp |
| order | 64.04% | 82.52% | −18.48 pp |

The refitted-probe improvement for strict_binding is +24.19 pp, with diagnostic paired-bootstrap CI [19.12, 30.05]. Together with the hidden-frozen control, this is scoped evidence that learned hidden weights improve the representation on this synthetic task. It is stronger than a mere changed-weight claim. It is still exploratory, task-specific evidence under a protocol deviation, and it does not establish general reasoning, language, vision, or lifelong learning. Shuffled feedback also performs substantially above frozen, so not all of the improvement is signal specificity.

The report correctly exposes the failures: order regresses, post-hoc pair-binding is about chance, and P2 final accuracy with usefulness is **56.13%**, mean forgetting **21.89 pp**, worst **54.95 pp**. The registered mean≤3/worst≤5 forgetting requirements remain unmet; no threshold is relaxed. PIVOT is appropriate, not GO for a better general learner.

## Engineering verification and limits

In an orchestrator-owned detached checkout, Codex reran the three reported test files: **27 passed**. This includes small engineering learning fixtures, not a new benchmark training campaign. Sampled review-process tree peak RSS 633,741,312 B and commit/VMS 1,087,004,672 B, below 2 GB; sampling is not an 8-hour memory certificate. The author's formal run separately reports tree RSS 608,903,168 B / commit 1,080,418,304 B. No live weights, Hermes settings or source files in the author checkout were modified.

Evidence: `runs/20261111T013650Z-round1-handoff-review-v2-cb9d2d/{gemini-p0-tests.log,gemini-p0-junit.xml,result.json,locked-metrics-audit.json}`. No formal trial was retrained by Codex. Submit a docs/engineering correction commit first; a fresh causal study requires its own approval.
