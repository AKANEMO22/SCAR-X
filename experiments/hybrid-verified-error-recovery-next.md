# Next research step: verified error recovery, not silent retraining

Status: design following the negative14/11current-learner audit; not executed,
not a claim that semantic self-repair works. Mastergates/ALIF/localcredit remain.
Do not scale neurons or open languagepretraining/LoRA to conceal a failed mechanism.

The completed event-source DEV answered whether the current relevance learner
beats matchedRAG controls: no demonstrated advantage. It did not train/test a
full root-cause agent. Production loops are currently bounded by runtimeguards.
The native2B probe used availableVRAM but failed mixedheadroom; no live promotion.

## Required causal chain to investigate

Actual tool observation → mechanical symptom → candidate cause → bounded probe
→ independently checked recovery outcome → permitted local SNN update → unseen
counterfactual task and cold restart. At every point record whether selection
came from a fixed guard, VLM, neuralpolicy or owner feedback. No self-critique
as truth and no negative label merely because the owner skipped a question.

Use real read-only sandbox faults, not invented successful tool replies:
wrongcwd/nonexistentpath, absent evidence, intentionalpermissiondenial, explicit
local ratelimitserver, oversizedtoolspillover, stale-but-validsource and weak
generatedanswer. Same raw observation may have multiple causes. Ask/abstain is
correct when no bounded independent probe can disambiguate.

Compare baseline RAG+existing strongguards, episodic verifiedexperience,
currentSNNfrozen/readout/hidden and proposed observationencoder. Same source/
time/action/token budgets, same frozenVLM, isolated state, task-family-disjoint
splits. Labels require observed recovery success on executable fixtures, never
just the constructor's familyID or anotheragent's report. Include noisy feedback
and legitimately repeatedpolls so “stop every repeat” cannot pass.

Inspect first whether the64D text hash throws away necessary state. A possible
encoder adds explicit permittedsource/time/status/progress and a causal event
sequence; this is a proposed representation change, not evidence. Keep current
localupdateequations fixed to separate representation from credit changes.
Any new credit/importance formula is a distinct research proposal, not implicitly
authorized by this document or imported into Hermes.

Lock data, seeds, encoder and recovery scoring BEFORE running. Primary benefit
>=5pp over strongestmatchedcontrol with lower95%CI>0; hidden above readout-only
and sensitive to hidden ablation. Acquisition plus20stage3/5ppretention, full
checkpoint/RNG/replay, RAM/VRAM/latency and rejectedupdates all required. If strong
rule reaches ceiling, add a separately registered harder task bank rather than
relaxing the gate or hiding that control. No researchwinner selected on finalTEST.

Production GPU direction: native2B may require component scheduling/buffer work
before safe video+ASR serving. Keep real16K/nativeprecision until a measured
profile is registered; do not silently lowercontext, moveweights toCPU or use
sharedGPUpaging to call a full-VRAM configuration stable. Preserve512MiB initial
headroom gate for this probe; add8hsoak before a productionstability claim.
