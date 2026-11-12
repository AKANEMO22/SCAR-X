# Verified basic-knowledge factory — source handoff

DirectiveV2,2026-11-11. **SOURCE_AND_ORIGIN_VERIFIED_CODEC_BLOCKED**.
No claim of DATASET_READY_FOR_TRAINING or NEURAL_PRODUCTION_READY.
No new learner, codec/vocabulary/live migration or P2pilot is implemented here.

## What exists and was executed

- Four actual Antigravity batches plus one registered repair: **5calls total**,
  requested/reported`gemini-3.8-flash-high`, effort`high`, historical CLI1.5.5
  executable SHA recorded in receipts. No call was made to finalize this handoff.
-40public AST facts pinned to main495dab2:32known model-configuration facts,
 8API facts treated as outside the registered training domain. This is not
  Internet knowledge, and outside-domain abstention is a policy, not factual truth.
-320accepted question/answer rows in Vietnamese/English. All16required provenance
  fields are present. AGY generated the candidate answers; independent source
  literals/bounded comparisons or the declared unknown policy verified them.
-128train,64development,64heldout-confirmation,32unknown-development,
 32unknown-confirmation. No final model-quality score was opened.
- The four original candidate files and the repair candidate file remain intact.
 11original rejections remain visible; only11original missing slots were filled
  from72repair rows. This is not a new score-driven generation sweep.
- Frozen question/source/Unicode/duplicate/input-boundary integrity passed after
  the controlled-query parser was made explicit. Definitions, units and derived
  quantities cannot be mislabeled by the numeric literal oracle.

Final pool:
`data/candidates/20261110T191533Z-basic-knowledge-repair-v1-231806/verified-pool.json`

SHA256: `13bb051467b27caa10bf8d16039d52fa830baf77a5e87bd63e3520d8b96ed062`.
Facts ledger SHA256:
`036035daf4106853e607dba742dabafb87c0d5f34b391baab33adefb13d70ef1`.
`.gitattributes` preserves these frozen data/source-cache bytes across checkout.

## Independent origin/source recheck

`scripts/verify_basic_knowledge_provenance.py` was run on the existing evidence:

- All4cached public source files exactly equal pinned Git content;40facts re-extract
  exactly from AST without executing source to decide truth.
- All5public candidate arrays match their recorded hashes.
- All5private CLI streams match hashes, reported model, SUCCESS status and public
  structured candidate arrays. Invocation prompt hashes match. Only logged`finish`
  tools were accepted. **No private stream/invocation content is published.**
- Every one of320accepted raw-candidate hashes points to an actual recorded
  candidate. Trusted source/split metadata and the prompt-only inference boundary
  are rechecked. All split JSONL files match the frozen pool exactly.

Receipt: `artifacts/basic-knowledge-provenance-review-v1.json`, run
`20261111T161733Z-basic-knowledge-provenance-review-v1-ac50ac`.
Public consumers can recheck candidate/source/hash lineage. CLI stream attestation
was checked locally by Codex; public hashes alone do not expose or reproduce the
private authentication/session logs. Cost is null/not reported, not asserted zero.

## Why training is still blocked

Public G1 checkpoint has24vocabulary tokens; the complete data+predeclared unknown
policy needs a union of55, still within the supported256cap. But **all320targets
have an OOV token relative to that checkpoint**, and128rows fail the existing
surface tokenize→render roundtrip for decimals. Token capacity, loaded vocabulary
and rendering are separate issues. No vocabulary was expanded to hide this.

No conflicting exact context vectors or over24token target were found in this
pool. Those narrow checks do not establish semantic representation quality or
absence of all statistical shortcuts. Known fact families intentionally occur
across train/paraphrase splits; this measures paraphrase transfer, not inference
of facts never taught. The lexical duplicate thresholds and clustering scope are
recorded in the factory protocol.

The current train split has no unknown examples; unknown DEV/final are evaluation
only. Reserving unknown words in a vocabulary does not teach abstention. A future
neural unknown curriculum or separately labeled controller calibration needs a
registered design; do not silently move unknown DEV/test into train.

## Baseline/reference evidence, not teacher labels

- Public G1 baseline96development questions: known exact/literal semantic0/64,
  registered abstention0/32. The checkpoint has not learned this dataset and
  cannot emit its missing tokens. No private state or learning was used.
- Qwen3-4B-Instruct-2507 Q4_K_M comparison24fixed development prompts completed;
 22responses hit128token limit. CPU2threads,context1024,temp0,seed919; sampled
  reference treecommit3.333GBunder separate4.5GBbudget; owned server stopped.
- Neither model was given the pinned repository source at inference. A strict
  literal-only rubric does not grade unrestricted explanatory prose/fluency.
  The zero rubric scores are not general intelligence equivalence or a claim
  Qwen lacks ordinary knowledge. Qwen output never became labels or proxy answers.

Receipts: `basic-knowledge-public-baseline-v1.json` and
`basic-knowledge-qwen-reference-v1.json`. Actual responses remain in their runs.

## Source entrypoints and guard

`prepare_adaptive_curriculum.py --factory basic-knowledge-v1` extends the legacy
wording entrypoint; its old default/schema remains available. **Do not rerun the
completed generation command:** completed receipts or interrupted run directories
now stop it before invoking CLI. Repair has the same started-budget guard.
Atomic reservation prevents two concurrent invocations from each spending the
same budget; unregistered protocol names fail closed. No reservation was made
for another generation during this handoff.
Another generation/version requires a separately approved protocol/budget.
The legacy entrypoint also uses the user's current FlashHigh/high preference for
any future approved invocation, verifies the reported model, and refuses a
completed/interrupted wording budget. Historical Low receipts are untouched.

Offline rechecks (no AGY/Qwen calls):

```powershell
python scripts/verify_basic_knowledge_provenance.py
python scripts/verify_basic_knowledge_provenance.py --public-only
python scripts/validate_basic_knowledge_integrity.py
python scripts/audit_basic_knowledge_compatibility.py
python -m pytest -q tests/test_knowledge_factory.py tests/test_basic_knowledge_eval.py tests/test_adaptive_curriculum_data.py
```

Private CLI logs must remain local. Do not run a probe against
`local/generation1`, restart Hermes, migrate a live checkpoint or train until the
research/codec approvals are explicit. The next concrete proposal is
[BASIC_KNOWLEDGE_CODEC_PROPOSAL_V1.md](BASIC_KNOWLEDGE_CODEC_PROPOSAL_V1.md).

`--public-only` is portable without private CLI logs: source/candidate/split and
submitted-public-prompt hashes are rechecked, while origin attestation is explicitly
marked not independently reattested. It must not emit the stronger local-origin
verdict. When the original checkout is on main, public mirroring is skipped;
new evidence stays in the isolated research worktree.
