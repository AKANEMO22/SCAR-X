# Adapter lifecycle v1 — user request 2026-11-13

The user re-issued the previously withdrawn Antigravity/LoRA request after the
pipeline audit. This newly authorizes candidate data generation, an isolated
local-policy teaching pilot, and engineering of bounded adapter lifecycle.
It does not change the original ALIF identity, scientific thresholds, native
precision, context16K, 4.5GB runtime cap, or permission to touch active peer/R13.
Previous "no LoRA" records describe their historical stage; new LoRA work is
restricted to the hybrid VLM branch. Never backpropagate through the ALIF core.

## Stages and declared limits

P0: use installed AGY1.5.5, gemini-3.8-flash-high, efforthigh. Exactly2calls,
<=4min/call, <=10MB output/call, no automatic retries. Send synthetic scenario
briefs only in an empty dedicated workspace, no private chats/video/weights.
Require requested/reported model match. AGY proposes natural-language wording;
labels are independently derived from a deterministic scenario contract and
remain synthetic policy-supervision, not evidence LoRA actually improves tasks.

P1: small isolated H128 ALIF policy using existing LocalALIF equations/encoder.
Six controller actions have an explicit separate purpose/checkpoint wrapper:
base / memory / repair / verify / reuse_adapter / propose_adapter.
No production action-head relabelling or serving weight replacement. Train on
training families only; DEV for diagnostics, locked new family templates for
TEST, split by scenario family not generated paraphrase.3seeds270113/270127/270131,
one fixed pass budget512updates/seed/arm. Arms learned-hidden/readout-only/frozen;
fixed-rule baseline separate. No BPTT, tuning on TEST, dataset-scale escalation.
Publish every loss/update/answer and exact integer scores. Bootstrap by scenario
family, not token/paraphrase. Existing lifelong≤3pp mean/≤5pp group at20stages
remains required separately; do not claim it from this single pilot.

P2 engineering: registry binds base model hash, artifact hash, purpose, source,
evaluation/retention status, RAM headroom and versions. At most1active adapter;
bounded3storedversions. No implicit mixing, no applying an adapter from another
base checkpoint, no switching mid-answer or learning from unverified content.
Return-to-base and rollback retain audit. Existing unqualified adapters cannot
activate. Actual VLM LoRA training/loading needs real compatible artifacts and
independent base-vs-adapter quality/retention/RAM validation; candidate metadata
or a mocked loader is not evidence of successful application.

## Before inspecting pilot answers

Pilot may inform an advisory controller only if TEST balanced accuracy≥85%,
unsafe propose/reuse on forbidden cases=0, all3seeds beat frozen by≥10pp,
and checkpoint cold next-update parity exact. To claim hidden benefit require
learned-hidden minus readout-only lower95%CI>0; otherwise report readout learning.
A rule baseline is an explicit-contract upper reference, not proof neural policy
beats a deployed adaptive controller. No promotion just for fitting labels.
Policy result is NOT a factual recommendation to train a real adapter; live
qualification also needs recorded failure evidence, labelled skill data,
retention and expected benefit exceeding memory/retrieval/tool repair.

## Scenario label contract

Priority: known pipeline fault→repair; unverified supervision→verify; ephemeral
fact with adequate local memory→memory; base already sufficient→base; existing
compatible qualified adapter with measured applicable gain and headroom→reuse;
otherwise repeated verified skill failure with enough clean data, no suitable
adapter, affordable training and idle time→propose; all other gaps→verify.
Numerical limits are controller-pilot design: ≥8independent failures,
≥64verified training examples, candidate gain≥5pp, runtime reserve≥150MB.
They are not a biological law or proof such thresholds are optimal.

No teacher confidence becomes groundtruth. The synthetic contract can teach
workflow distinctions; it cannot replace evaluating real adapters on new tasks.

## Data-only amendment before the FIRST student training run

CLI reported1.5.8 at generation (initial help probe1.5.5); both generation
init events confirm gemini-3.8-flash-high, requested efforthigh.96wordings
reviewed: no LoRA/adapter label keywords in the inputs; examples are synthetic.
Replace the source-ID index split above BEFORE training with deterministic
within-action lexical duplicate clusters (Jaccard>=.4 or containment>=.7 with
>=4shared words).95clusters,1,072rows including counterfactual fault/unverified
states:716train/100DEV/256TEST. Whole clusters never cross a split. This is
lexical grouping only, NOT proof of semantic independence/OOD; CI remains an
exploratory synthetic-contract diagnostic. All gates/seeds/512steps unchanged.
Same prompt with a broken pipeline or unverified labels must change action;
do not reward memorizing wording group names. No real adapter gain is measured.
