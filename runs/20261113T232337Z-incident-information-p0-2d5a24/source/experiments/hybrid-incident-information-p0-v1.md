# Incident information-path P0 — registered14/11/2026 before fixture execution

Previous goal turn made progress: bounded hooks/feedback implemented and actual
GPU candidate/rollback evidence committed64d5a581. It did not establish learned
semantic recovery. Current source must be audited before another training run.

Hypothesis: HybridEngine.goal_sequence discards reported tool errors and probe
outcomes. Distinct failure states requiring different next steps then produce
identical neural input, preventing even an ideal stateless classifier from
distinguishing them. Weight growth/training cannot recover information not input.
This is an information-path engineering audit, not formal P2 or generalAI proof.

Use actual Hermes read_file through the isolated source/private HERMES_HOME,
not mock tool replies. Three missing requested paths with identical basename:
alternative root file readable; alternative absent; alternative exists but is a
directory. Two further outputs exercise bounded partial read and full file read.
Only owned temp fixtures, no user project/video/chat data or permission changes.
Record exact real tool JSON/errors, filesystem existence/type and source hashes.
No network, model-generated labels, synthetic success strings or serving edits.

Replay the exact public observation-hook payload construction into a new CPU
HybridEngine with no model forward/learning. CPU is only encoding/serialization;
all later neural computation, if separately registered, stays GPU as requested.
Create3identical calls per independent scope, same tool/goal/flags/presence.
Record reason, incident, actual goal sequence and its SHA. Independent next-step
contract from measured tool results: readable alternative→inspect that source;
absent→request missing source; directory alternative→clarify file/path type.
These are next-step annotations, not proof of resolved task or factual learning.

Compute exact input-equivalence groups and empirical deterministic-classification
ceiling sum(max label frequency in each group))/N, an information diagnostic only.
No CI from these hand-built engineering fixtures; no acquisition/retention or
5pp hidden-advantage gate can pass here. Retain observation privacy/no-self-label
boundaries, zero neural updates and full state persistence. Runs immutable.

P0 may implement a RESEARCH-ONLY sequence encoder that reads allowlisted actual
incident/probe flags in chronological order, using existing64D row features and
unchanged ALIF equations/dimensions. Never input fixture name, gold label, future
success, scope/hashID or human answer. Counterfactual tests must prove label/ID
invariance, causal order preservation, changed observations distinguishable and
malformed/future fields rejected. Do not connect encoder to live goals, change
checkpoint identity or train in this registration. A new learning DEV requires a
separate locked protocol, data/seed/budget and matched strongRAG/readout controls.

Budget≤5minutes, fixture artifacts≤10MB, no GPU/model restart. Stop on actual
tool/parse/persistence failure; preserve failed receipt. Publish real fixture
outputs and input-collision findings to localhost under actor orchestrator, not
live neural counters. Main/peerR13/ALIF/weights/LoRA remain unchanged.

References: Reflexion(arxiv2303.11366) and ExpeL(arxiv2308.10144) improve agents
using experience/text memory without parameter updates; include those mechanisms
as future controls, never attribute their benefits to SNN weight learning.

Technical timestamp amendment before any learning: first P0fd128e confirmed an
input collision but the proposed encoder relied on caller-supplied chronology.
Preserve it. Add actual tool-return epochs and required decision cutoff; reject
future/backwards/nonfinite times, never encode absolute time or identity. Repeat
only this engineering fixture with a new run/source bundle. No training, gate
change, label change or selecting a winner on these three cases.
