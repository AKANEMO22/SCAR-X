# Six-source neural learning map — research review status

Completed source/math mapping, not a new learner or superiority benchmark.
Pinned source receipts were fetched earlier; finalization verified37cached files
against recorded SHA256. No external code was copied/executed/imported into serving.

The required matrix is in
`runs/20261111T092024Z-neural-method-map-v2-fc6a05/report.md` and machine-readable
`configs/neural_method_audit_v2.json` / `artifacts/neural-method-map-v2.json`.

| Source | Main difference to SCAR-X | Reuse status |
|---|---|---|
| e-prop `efd02e6` | reset/stop-gradient convention, adaptive eligibility indexing/surrogate and feedback | Concept/math reference; BSD-style license text cached, API NOASSERTION; no code reused |
| DECOLLE `7d72cb1` | local auxiliary classifiers and detached layer/state graph | License text/README conflict; concepts only, no framework copied |
| BindsNET `314c334` | separate delayed eligibility low-pass and reward timing | AGPL3cached; no code reuse; do not call scalar-STDP proposal30 |
| ContinualAI `4900462` | wrappers delegate to Avalanche SI/EWC/GEM; SCAR-X uses output-local adaptations | Wrapper MIT, dependencies not automatically covered; P2not authorized |
| SuperSpike `9bd0c41` | filtered PSP/surrogate/error timing and routing | GPL3cached; no source copied/executed |
| SpikingJelly `9618271` | simulation/online/STDP toolkit with multiple graph/reset modes | Apache2cached; not a single credit rule or serving dependency |

Array memory formulas are arithmetic estimates, explicitly excluding allocator,
framework and snapshot runtime. Original e-prop verification code stores a time
axis; that is not a necessary streaming serving cost. SCAR-X35 local surrogate
finite-difference agreement does not establish full-network gradient correctness.

Main35–38stopped quality studies remain stopped. Research questions suggested by
the map include reset-Jacobian credit and query-feature separability; neither was
implemented/trained merely by writing this map. Subsequent branch audit found
task-scoped Gemini EW-HE hidden gains and unresolved retention/measurement issues;
see the separate `codex/scarx-forensic-r6-20261111` review, not a claim that this
map or generic e-prop adoption has solved them.
