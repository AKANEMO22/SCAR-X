# Stored video evidence across real process restart — before inference,14/11/2026

Keep new local-rule ablation pending user approval. This is an independent
engineering/known-case diagnostic of actual memory and answer paths, no training.
Existing a98122 captured one21slicensed speech/shapes clip with nativeWhisperbase
and small. Original errors remain; do not recapture, retuneASR/prompts or call
these new unseen samples. Original3locked questions/answers15,9,yellow unchanged.

Four privatecells:2storedASR sources×2selectors (currentuntrainedALIFscorer and
fixed RAGcandidateorder). Use SQLitebackup of owned a98122 data only, never the
live conversation database. Original evidence/stream/source media/checkpoint
hashes must match and remain unchanged. No new sensor/ASR/VLMweightload.

Eachcell runs in subprocessA, exits; subprocessB starts on same private data
and restored checkpoint. Before/after record exact evidence table/stream/time/
source/text digests and neural learnedstate weights/importance/stats/RNG/replay/
updatecounters. Working membrane/ticks and controller presence/history may
change during legitimate ask; persistence must be compared at the actual
saved boundary, not to arbitrary initial ephemeral state.

Use unchanged HybridEngine.ask(candidate filtering, evidence budget4, query
timestamps, prompt/source status). RAGcontrol explicitly replaces relevance
with constant0 so stable candidateorder from FTS/temporal filtering is kept;
it is a harness override, not a new learned arm or full strongestRAG. Preserve
engine's raw retrievalstring even when generictext still saysSNN; add separate
selector_override field to avoid misattribution. No data/answer label enters
core, no engine.start/idle worker/feedback or tools executable by model.

Shared originalQwen0.8B/16K/CUDA/nativeprecision, max128outputtokens, same live
worker identity in allcells; foreground userwork retainspriority.24actualcalls:
4cells×2coldprocessphases×3questions. Do not reuse cached response as evidence
of a real second inference; archive all requests/rawoutputs and hashes.
No retries to turn semantic errors green. Limits10minutes total/90s pernative
request and512MiB physicalVRAMreserve at eachphase/question. No workerrestart,
new gateway, active8hsource edits, live model or label updates.

Primary engineering checks: phaseB starts after A ended with different verified
PID/create-time; initialB fullmodel+controller payload equals finalA payload;
evidence/stream/sourcebytes and neural learnedstate unchanged across phases;
same query produces same selectedordered evidence and exact prompt. Report
answer equality separately from persistence; identical evidence does not
guarantee deterministic nativegeneration. Cold process is private memory/ALIF,
not a cold reload of VLM, so do not call latency a whole-model coldstart.

Strict semantic score remains casefold(trim(answer))==originalexpected. Source
citations and numeric answer formatting may conflict, held constant and recorded;
do not add a forgiving rubric after results. Count sixunique source/question
pairs, not24independent qualitycases. Compare selectors descriptively only.
No P1/P2/P3/lifelong or SNNrepresentation claim even if all3recallcases pass.

Immutable unique run with source/sourceDB/checkpoint/media hashes, questions,
PID exit receipts, actualpre/postoutputs/evidence and independentdata-onlyaudit.
Publish on8775 as measured diagnostic answers, separate from liveSNN counters.
