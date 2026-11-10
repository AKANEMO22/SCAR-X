# G1 terminal punctuation input normalization — bounded product acceptance

Registered2026-11-09 BEFORE this inference comparison. Current encoder hashes
punctuation; e.g. `hi?` differs from `hi` although these basic utterances have
the same greeting intent. Fixed proposed normalizer removes ONLY finalASCII
`.!?`tokens (after existingNFKC/lowercase tokenizer) when at least one
nonterminal token remains. Internal decimal, code/file punctuation is kept;
all-punctuation inputs stayunchanged. This is input normalization, not semantic
parsing, answerlookup, newlearningrule, targetaccess or general-languagefix.

Default old checkpoints/runtime remain unchanged. Optional versioned encoding
field `surface_normalization=terminal-punctuation-v1`, same weights/dimensions,
must survive checkpoint. No parameter training in this acceptance run. Compare
all6final rehearsalclones fromg1-basic-rehearsal-v1 with normalizationoff/on;
no further epoch/LRsearch, no stoppedfamilyreopening or hiddenneuronsincrease.

Data: SAME112publiccases(before personaloverride exclusion) and privateguards,
same4states, exactsame sourcefixture/sourcepersonalcheckpoint. AGY48already
opened diagnostics, notfreshfinaltest. Original rehearsal failure remains
unchanged. Publicknown/AGY scoring remains stricttokenmatching toone style.
Personalguard uses EXISTINGRUNTIME normalizedcomparison (already ignores
terminalpunctuation), fixed BEFORE thisrun. This distinguishes `15.` vs `15`
from a lostvalue without rewriting the priorstrictstyleexperiment or P2gates.
Proofguards include allrepresentableverifiedproofs from source, even if not
in learnedstatus; expect no previouslycorrectguard lost under any state.

Same screen thresholds: known>=80%onreset/>=75%eachotherstate, AGY>=25%AND
gain>=10pp vs ORIGINALsource inEACHstate; no oldcorrectpersonalguard loss under
runtime matching; digitoriginalcorrectnone lost and mean drop<=3pp. All3order
seeds must pass for a mode. Preferreadout-only and fixedseed1701, never pick
bestscoreseed. If fails, do notnormalizeorreplace live from thisexperiment.

Even if passes, applying requires separate atomic guardedtransaction against
unchangedsourcefingerprint, exactsourcevocab/encoding/topology and freshguard
checks; preserveprivatehistory/currentneurondynamics, never import trained
clone's past conversation/controller or overwrite logs. No autoapplyinrunner.

Synthetic exposure audit also reports word-sequenceduplicates vs oldfixtures
after deletingterminalpunctuation; previousstrict-tokenduplicatecount remains
historical. No false heldout/novel-skillclaim. Weights remainlocal and excluded
packaging. Publish actualinference/acceptance, no fakeQwen or successlabels.
CPU1thread,2GBcap,180sround, noQwenconcurrently. No normalchat/vision/P2claim.
