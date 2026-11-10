# Visual information-path audit v1 — no neural training

Registered2026-11-10 BEFORE code/measurement. Master§4.1 and report33 explicitly
prioritize input→state→visualtrace→wordhead diagnosis. No new learner, encoder,
decoder/state-view grid, parameter tuning, scale, deployment or proposal30 code.
STOP32/33 and earlier closed state/byte/score families remain closed.

## Sources and invariants

Use pinned datasetSHA0fb04abefe67451952a2d73634fded9c86243d5874356d68299ff047deecf6a2,
192ownedPNG/864train/864devQA, actualAGY16/16wordings. Pairedseeds33113/33119/33133:
three frozenFULL32 checkpoints and three isolatedALIF-BPTT33 checkpoints. Source
receipts/checkpoints/data/math hashes checked. No private source, teachercall,
Qwenlabel, synthetic fakeQwenvision, new neural training or mainweightschange.
Inspecting BPTT controlweights is not permission to learn BPTT in main.

NumPy VisualWords is the ONLY authoritative forward. No Torch/autograd import.
Unwrap bounded/checksummed control envelopes ONLY to read their experimental
VisualWords copy; never expose a serving loader or import control into main.
For each source, run prepare(question,actualPNG)→word_feature(BOS,0) for all1728
cases. Every input tape exactly36ticks, with33prepareticks. Neither target,
rendererattributes, kind/language/scene ID nor probe labels enter this forward.

Record existing path only, no alternative states/mixes/masks: complete inputtape
SHA, question128, actual16RGBpatches×48(flat768), stateafterprepare(voltage/
adaptation/spikes/features), cachedvisual48, actualpooledBOS48, actualnormalized/
clippedpresynaptic48, outputlogits. Per-stage vectors/signatures and diagnostics
are measurements, NOT new model representations to select/deploy.
Sourceweights/stableweights/topology/calibration/learningcounters/RNG unchanged;
discardedcopy workingstate/ticks/imagecounter necessarily advance. Sourcefiles
and mainprivateweights/topology/vocab/encoding unchanged. Preserve history.

## Exact information-loss checks

Eachcase has two external diagnosticlabels: firstgoldword (trainvocab ID) and
semanticjointkey kind/language/expected(scene,kind,'en') from OWNEDrenderer.
These labels are evaluation/supervisedprobe targets only, never forward inputs.
Semanticjoint includes20 known task/language/attribute classes; it is classification,
not verbal generation, and cannot be compared as if it were sentence accuracy.

For eachstage/split report exact-byte collisiongroups, contradictory labels and
oraclemacro/micro ceilings: sum overequivalencegroups of largest label mass.
Macroweights give equalmass per(kind,language). These bounds concern only THIS
observed finite dataset, not total neuralcapacity or futuregeneralization.
No rounding/quantization to forcecollisions. Absenceofcollisions is NOT learnability.

Counterfactualpairs built from existingcases, not new inputdistribution: eachcase
gets FIRSTeligiblecase with SAMEquestion/differentsemanticattribute (imagepair),
and FIRSTeligible SAMEimage/differentsemanticjointkey (questionpair). Deduplicate
unorderedpairs by IDs. Targets selectpairs outsideforward ONLY. Report counts,
exactcollapse fraction and medianL2distances at eachexistingstage. If distinct
pre-transformvectors become identical postclip with contradictory labels, report
observed many-to-one loss; do not infer that every performanceerror has thiscause.
Record fractionofheadchannels at±5 and constantchannelcount. Descriptive centered
SVD rank uses singularvalue>maxSV*1e-6, no featureselection.

## Fixed diagnostic readouts, not neural learning

Reconstruct eachseed's EXACT1536sampled TRAINcase multiset with existing sampler,
initial sampler RNG seed+12100000. Compare stratumcounts/final RNG to source32/33.
No extra traincases/targets are used by any fittedprobe; rowswithsamplecount0 have
weight0, excluded from trainingstats. Dev labels never choose any parameters.

Single fixed closed-form weighted ridge with intercept, alpha1 on SUMsquaredloss,
no grid/epochs/backprop. Weighted TRAINmean/std(population,floor.05), no extra
clipping; solve (X'WX+I)B=X'W(Y-meanY), intercept meanY. This is offline headfitting
with different compute/order from online delta, NOT a new locallearner or its score.

For EACH6sources, fit only on the EXACT ACTUAL normalizedBOS presynaptic48vector:
1.firstword probe, outputs allsamewordrows/mask as realBOShead; compare realhead
  firstwordaccuracy on identicalcases/labels. This isolates availability to a
  fixedclassifier at the FIRSTWORD, not completeutterance/hidden-learning success.
2.semanticjoint20class probe, removes worddecoder from attribute diagnosis;
  rendererclass/intent/language never supplied as probeinput or taskrouter.
No trainedprobe on voltage/cache/alternativeviews. Thus no reopening stateviewsearch.

Two fixed rawinput references perseed(multiset shared by bothmodels): question128
→kind/language8class (questioninformation), and flattenedactualRGB768+question128
→semanticjoint20class (inputavailability). Not a learned visionencoder/mainpolicy.
For everyjointsemanticprobe fit one label-shuffled negativecontrol, fixed RNG
seed+34000000 applied to the1536exposurelabels before countsaccumulation. This
preserves exposure frequencies but breaks input-label association. Report it,
never choose a seed or method from its score. No oraclekind-conditioned probe.

Report trainweightedfit, devmacro/perkind/perlanguage/novel, actualfirstwordhead,
paired3seed95%bootstrap and probe-vs-realBOShead difference. Historical/inspected
dev is exploratory, NOT freshconfirmation or wholeSCAR-X quality/retentionproof.
No classifier-label formatter may replace actual neural answers in UI/Hermes.

## Locked interpretation and budget

Diagnostic flags, NEVER project-success gates:
* OBSERVED_INFORMATION_COLLAPSE if an existingstage's devsemanticoraclemacroceiling
  <65%, or conflictingpair distinctupstream→identicaldownstream is observed.
* BOS_READOUT_TRAINING_GAP if mean firstwordprobe−realhead devmacro>=10pp and
  pairedseedlower95>0, separately per sourcefamily. This concerns FIRSTWORD only.
* CONTROL_REPRESENTATION_GAIN if BPTT33−frozen32 semanticjointprobe mean>=5pp and
  pairedlower95>0. Even then it is globalCONTROLrepresentation, NOT permittedlocal.
* Otherwise UNRESOLVED_WITH_FIXED_DIAGNOSTICS; do not add views/probes/LR/tokens.
Multiple flags may coexist. No claims of uniquecause, impossibility or normalvision.

CPU1thread, cap300s/2GBprocess AND sampled tree, no iGPU/weightsSSDoffload. Stop
nonfinite/provenance/targetleak/stateinvariant/budgetfailure; no measurement-based
reruns or modeltraining. Meaningfultests beforefullaudit; fullpytest/nativeparity
for runtime/UI changes. Publicpercase records/metrics go to registeredrun/passive
localhost; mark AUXILIARYPROBE vs realmodeloutputs. Existing33answerarchive may
remain linked with originaltimestamp, never claimed as newauditgenerations.
No mainlearningloss/ΔW fakeentries. Source/private invariants verify afteraudit.
