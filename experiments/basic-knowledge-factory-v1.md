# Basic knowledge factory V1 — protocol before generation/quality scoring

DirectiveV2,2026-11-11. Data/engineering only; no new neural learner, tokenizer,
encoder, vocabulary migration, P2pilot or production integration authorized.

## Public sources and domain

Pin user-verified public commit495dab2f4bac0180e4d1cd28674391f055ccf09d of
AKANEMO22/SCAR-X. Fetch only4publicsourcefiles, compare read bytes with local
git-show at that commit, store SHA256 and source span. Never upload live data.
AST literals, constructor keywords and bounded comparisons are independent
source oracles; no source code execution to determine truth.

40facts:16Configdefaults and16UnifiedCoreconfiguration facts in the trained
domain;8API facts form out-of-domain/unknown bank. They are version-specific
technical knowledge, not allInternet knowledge or proof of general language.
Unknown labels mean outside this32fact exposure/domain, not false/unknowable in
the world. API fact actual values remain in sourceledger, not model inputs.

## Actual Antigravity factory

Reuse installed agy.exe. User explicitly selected **gemini-3.8-flash-high** with
**effort high** on2026-11-11; this supersedes Low for new batches only, no other
model/provider escalation or newpaidAPI. Verify executablehash, CLIversion and
reported init.model. Keep all historical Low provenance unchanged.
4sequentialbatches of10facts,8variants/fact; one bounded repair call at most for
rejected/missing structured records (no qualityscore-driven repair). Total<=5
calls,270s/call,1GBobservational process-treebudget, one numericalthread.
Rawlogs/privateCLI IDs stay local. Public receipts record exact invocation,
input/output hashes, requested/reported model, usage/cost if reported (else null).
Only public factledger/contract is sent; freshTempCWD outside repo, no history,
SQLite/checkpoints/weights. Nativefinish is the only allowed observed toolstep.

AGY generates prompt AND candidate canonical answer. Codex/Qwen references from
older data remain reference-only; they are not relabelled as AGY training data.
Candidate answers pass only if independently consistent with AST oracle or the
registered outside-domain abstention policy. No model consensus as verification.

## Schema, alignment and splits

Final rows include all16directive fields: id/domain/task_family/language/prompt/
canonical_answer/source_id/source_hash/label_origin/verification_method/
verification_status/split/difficulty/generator/generator_revision/dataset_version.
Additional fact_id,paraphrase_cluster,raw_candidate_sha256 are allowed. Teacher
cannot assign verificationstatus/sourcehash/split; factory joins trusted metadata.

8variants odd=vi/even=en. Knownfacts:1–4train,5–6development,7–8heldoutconfirmation.
Variants must use distinct constructions, not punctuation/name-only changes.
Unknown API bank:4facts unknown-development,4factfamilies unknown-confirmation;
never part of training. Retention schedule references training/probe IDs across
20stages with seed44901; no task ID/answer/source given to neural inference.

Controlled technical question oracle: known prompt must name its exact public
parameter in backticks and Config(default) or UnifiedCore(initial) scope. Only
one backtick parameter is allowed. Max/limit qualifiers required for boundfacts.
Reject yes/no, negation, digits or written-out numbers/answer leakage, wrong scope/field, bad Unicode,
duplicate IDs, and unverified question intent. This is controlled language, not
verification of arbitrary natural paraphrases. Invalid/free-form rows quarantine.

Same-fact normalized exact duplicates or cross-split wordJaccard>=0.9 or
character4gramJaccard>=0.8 quarantine later-split row. Do not relax thresholds
after outputs. Cluster-level count, not every paraphrase as independent evidence.
Finalconfirmation files sealed from modelselection/score use; validators can
inspect schema/source/representability, but no inference/qualityscores thisstage.

## Compatibility gates before learning

Audit256vocabcap/currentcheckpointOOV, tokenize→render roundtrip, EOS/answer24token
limit, Unicode, contradictorylabel/inputcollisions under fixed public G1encoder,
unknown policy and source-to-target leakage. Never change codec or fabricate
normal answers for compatibility. Unrepresentable corpus:
`DATASET_NOT_LEARNABLE_UNDER_CURRENT_CODEC`, even if source labels are correct.
Data verified != authorized to train != neural learned knowledge.

No P1training without valid new causal hypothesis/protocol/authorization; no
reopening STOP35–38 just by changingseed, corpus or methodname.
