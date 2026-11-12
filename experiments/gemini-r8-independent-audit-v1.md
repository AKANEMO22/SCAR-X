# Gemini R8 independent audit / evaluator repair V1

User requested continued work on current blockers and review of Gemini's new
branch. Pin `fe472b3b1257cdb264805bf79fd53e5647e3568d`, research only. Ponytail
tools/context unavailable. No changes to author/main/live/ALIF equations; no new
candidate, quality training, DEV campaign or fresh confirmation is authorized by
this audit. Peer tests use their existing bounded engineering fixtures only.

Before execution: inspect source, raw JSON, protocol and prior seed exposure.
Recompute the stored DEV R-matrices using the unchanged Phase2.1 convention:
acquisition=mean(diag), final=mean(last row), forgetting_j=max_{t>=j}R[t,j]-R[-1,j],
including last task zero in mean. Apply acquisition/final>=.65, mean<=.03,
worst<=.05 to EVERY seed; report summary effects separately from certification.

The master's representation lower-CI threshold .05 and >=8 fresh confirmation
seeds remain unchanged. Never silently overwrite the author's historical verdict.
Missing raw/control/resource evidence cannot yield PASS in the independent view.
Implement a separate reusable gate/data-integrity module and mutation tests;
do not patch the candidate learner to make a test or quality score pass.

Reproduce empty/missing-evidence gate acceptance; inspect high-accuracy exceptions.
Exhaustively inspect finite query codes (2/4/8 keys), reporting collisions and
conditional-label ambiguity rather than blaming neuron capacity. Model-free
fixtures may reject invalid codebooks before any future training.

On an owned detached pinned checkout: run the existing R8 unit tests, then tiny
non-training fixtures for snapshot/RNG/threshold, dtype and missing serialization,
trace-control equivalence and allocated-array accounting. No learner.learn call
in new audit fixtures. No language dataset, external model/API, paid tool call,
confirmation run or live SQLite. One numerical thread,2GB cap, process-tree
monitor, bounded commands/logs. Preserve every unique audit run/source snapshot.
