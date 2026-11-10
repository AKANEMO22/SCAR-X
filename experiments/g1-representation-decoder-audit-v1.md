# G1 representation/decoder split diagnostic v1

Registered2026-11-10 BEFORE measurements. MasterP1/P2/P3 distinguish learned
features, output-local acquisition and final utterance quality. Read-only
analysis of existing public data and three fixed private curatedreadoutclones,
NOT new modeltraining, deployment, familyretuning or researchgate change.

Source chosencondition=surface_expanded, allorders1801/1811/1823 fromcompleted
g1-curated-curriculum-v1. No winner/cherrypicked model. Their hiddenweights are
identical; preserve whole source model state/RNG/encoding, cloneonly. Train
probe data=63oldpubliccases after priorpersonaloverride exclusion +117AGYtrain.
Dev31, priorAGY48diagnostic (already inspected, not a newfinaltest), disjoint
after normalizedterminalpunctuation reported; no target in featureextraction.

Eight intenttargets known; calculate 8classridge diagnostic at fourviews:
bagcontext128; first response-token hiddenfeatures(beforeanyassistantword);
whole-response teacherforced featuremean; target-token-independent recurrent
idlebaseline. Whole-responsefeatures intentionally CONTAIN goldpreviouswords,
so this is a disclosed leaky diagnostic uppercontrol, NEVER an inference
representation/qualitymetric or runtimealgorithm. Idlebaseline is zero-input
neuralfeatures same12boundaryticks, no prompt/target.

Actualprompt featureview: canonicalepisode state on diagnosticclone; normal
foregroundcapacity selector; boundary12 then word_step(BOS,context,position0).
Same words/hash/ALIF/activefloor as currentG1, no target tokens allowed. Save
vectors and originalheadfirsttokenprobabilities locally. Probe ridge.01,
train-onlycenter/std;8classlabels only in probe fit, not hiddenweights. Also
readout8class localsoftmaxdelta reference, SAMEfixedfeatures, train20passes,
seed1901/1907/1913-order,lr.08,clip.025,weights3. It is a diagnosticoutputhead,
never a template selector or answer generator. Actualutterances still generated
by unchanged autoregressive model; do not return canonicaltemplate byintent.

Decoder measures separately: tokenNLL/argmaxaccuracy under teacherforcing;
firsttoken correctness; freegeneration strictstyleexact; prefixagreement;
EOS truncation/repetition; total target vocabexpressibility. Keep original
metricsfailures unchanged. Report train/dev/priorAGY separately, not just
easyrecall. We do NOT infer broadsemanticcorrectness from stricttargetmatch.

Intentdiagnosticinformative if prompt-onlyprobe>=65%onDEV AND gainvsidle>=20pp;
descriptive criterion only, not P1gate. Source carries initialrandomhidden plus
personalSTDP modifications; no comparison toinitialhidden here, so NO hidden
learning attribution from probe accuracy. If inputprobe high but hiddenlower,
propose representationpassage audit; ifbothlow, encoding/dataissue; ifprompt
hiddenhigh but freegenerationlow, decoder/cross-token conditioning implicated.
Multiple issues cancoexist. No reselection/extra learningbudget after data.

Commontrain/dev/AGY feature extraction samecanonicalstate; realgeneration
also on canonicalclones and same24wordlimit. Samples acrossprompts NOT treated
as independentmodelseeds; threeclones differ outputweights/curriculumonly.
No significance/normalchat/vision/retentionclaim. Rawpersonalweights/vectors
staylocal, aggregate/publicpredictionreports underuniquerun.

Publish realfirsttoken/freeutterance outputs/progress tolocalhost. No fake
Qwen/labels asanswers. Existing Qwenexactpromptreference mayattach; no new
referenceLLMprocess or teacherAPI needed. CPU1thread,2GBcap,120s/model/360sround.
Stop on NaN/state mutation, labelleak inprompt-onlybranch or deadline.
