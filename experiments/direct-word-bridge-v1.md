# Approved proposal30 → small autoregressive word bridge v1

Registered2026-11-10 BEFORE data generation, implementation or performance.
Protocol35 explicitly permits a separately registered word/vision bridge after
math/state integrity. P0 passed, P1screen35 FAILED: this application pilot does
NOT reopen P1confirmation, waiveP2, scale/pretrain10Mtokens or deploy intoHermes.
Same approvedone-stepreadoutfeedback/own-neuronforwardeligibility. No Transformer,
pretrainedweights/proxy/retrieval/class-IDresponseformatter/networkBPTTcandidate.
One fixed application screen, no postresult encoder/LR/data/epoch/decoder grid.

## Corpus registered before teacher call

Eight public stylefamilies: greeting/identity/thanks/farewell ×vi/en. Canonical
targets owned by project, not teacherfactlabels: vi_greet="Chào bạn.",
en_greet="Hello.", vi_identity="Tôi là SCAR-X.", en_identity="I am SCAR-X.",
vi_thanks="Không có gì.", en_thanks="You're welcome.", vi_bye="Tạm biệt bạn.",
en_bye="Goodbye.". Style supervision only, not broad factualtruth/intelligence.

Installed AntigravityCLI gemini-3.8-flash-low generates PROMPTS ONLY,128items,
16/family, each1..180characters/<=32tokens. Publiccontract only, no privatehistory/
images/sourcefiles/weights/tools. IDs family_01..16; CLI<=270s, unexpectedtoolsteps
quarantined exceptnativefinish containing EXACTfinalpayload. Retainrawprivatelogs.
At most one generation attempt plus explicit datarepair of schema-invalid rows
before any neuralqualityresult; no selecting easy prompts using modelperformance.

Read/review ALL128 for intended language/intent before training. Quarantine
ambiguous/incorrect/injected/toolrequest/factualquestion examples, never trust a
verifiedflag. Required complete128 reviewed or STOP_DATA_REVIEW_INCOMPLETE, no
silentlabels/duplicates/replacements. Originalcandidate immutable; reviewedcopy
keeps row IDs/caveats/sourceSHA. Teacher never supplies modelanswers.

Split per family using fixed RNG36101,8randomIDs TRAIN and8DEV; teacher not told
assignment. Exactly64/64. No normalizedduplicate acrosssplits. Report historical
prompt overlaps, never call inspected/similarstyles a freshfinaltest. Newmodel
seeds/randominit means no oldcorpus/livecheckpoint import or devtraining.
Vocabulary ONLYcanonicalTRAINanswerwords+BOS/EOS,<=256, target<=7tokens+EOS.

## Fixed one-core forward and learning

Newresearch wrapper reuses UNMODIFIED UnifiedCore.context (legacy signedhash128)
and UnifiedCore.word_step's actual3tick wordinput (contextgain1.6,prevgain.8,
same6flags). No encodersearch, intent/language router or embedding model.
H48/K8/R12, D=V+198/O=V+10 reserved digitrows, SAME G1ALIF leak.65/threshold.5/
adaptstrength.02/adaptdecay.95/featuredecay.25, w_in initial offset-.15.
Reset dynamics per researchresponse +12zeroticks, then3ticks peroutputword;
currentquestion128 numericinputbuffer retained/repeated. This is explicitbounded
workinginput, not neural retention. Productioncontinuousruntime unchanged.
This is wordinterface/task adaptation, not a claim exactsame dynamics/task as35.

Use ReadoutEligibility implementation35 UNCHANGED: psi logistic.4, ownreset/
adaptation/feature sensitivities, ONEW_out.T@error/scale/interiorclipmask,
no hiddenerrortransport/RTRL/Ttape/autograd. Head normalized with128unlabeled
TRAINquestion/randomprefix0..2 samples seed+36300000; mean/std frozen beforelabels.
Output-local head delta.08/clip.025/weight3. Hiddenbank averages allwordendpoint
directions, apply at ENDresponse only, same synapse-localAdam.001/.9/.999/1e-8,
deltaentryclip.002/weight3, all48rows. Initial weights random, noLLMweights.
Training teacherPREVIOUSwords only; generation previousNEURALwords only.
Shared fullwordmask, BOSneveroutput, EOSforbidden firstposition, firstpunctuation
masked exactlyG1policy. Max8neuraloutputwords, EOS orlimit, no labeltemplates.
No currenttarget/intent/language metadata to prepare/generate. Expose actual
question/wordoutputs, not teacherlabel copied into response.

Conditions correct/shuffled/zero/frozen + matchedisolatedBPTT;3freshseeds36313/
36319/36341 =15trials. Same corpus/init/calibration and1024TRAINresponses,
samplefamily uniformly→sample8TRAINprompts with RNGseed+36400000. Same endpoint
count/wordticks; zero/frozen neural/headweights mustmatch. Shuffled neuronL
perword using existing signalRNG, headlabelcorrect. No perwordhiddenupdate.
BPTT ONLYcontrols.py on identical observedNumPy trajectory, mean perwordloss
using pre-local-headupdate W snapshots; same localhead/Adam/no globalnormclip.
Disclose extra graphcompute, never import its weights/control into main.

Separate checksum/nonpickle boundedcodec preserving neuron/head/localtraces/
bank/moments/RNG, context128, vocabulary, previousword/position/phase/counters.
Interrupted wordresponse and transactionrollback exact; main/candidate/control
formats cannot silently interchange. Feedback disconnected inferenceforward
unchanged. Wholepytest/nativeparity + TRAIN-only64response timing before formal;
no devqualityselection. Fixed256tick segmentlimit; normalresponse<=36ticks.

## Measurement/stop criteria

OnlyFINAL1024checkpoint, all64DEV andall64TRAIN, freeautoregressive outputs,
tokenNLL/EOS/firstword separately. Macro over8families; eachfamily/lang, actual
ticks/time/RAM. Never use teachertokens to score freegeneration.
Resethead diagnosticprobe: firstwordclassifier fit on exact observed TRAINprompt
multiset using BOSfeatures from trained vs initialhidden; fixedridgealpha1,
trainmean/std floor.05, no devstats/grid. Classifier is diagnostic, not decoder.
Restore trainedmodel hiddenw_in/w_rec to INITIAL at inference (samehead/stats)
to measure dependence; this lesion is diagnostic/OOD, not independentmodel.

Pilotpositive requires EACHseed macrofreeDEV>=65%, EACHfamily>=50%, meanpaired
correct−frozenfreeDEV>=10pp allpositive/lower95>0, meanpairedresetheadprobe gain
>=5pp/lower95>0. Report correct−shuffled and restorehidden lesions even ifnegative.
Positive=WORD_BRIDGE_FRESH_VALIDATION_REQUIRED; NOT normalchat/main/P1/P2pass.
Failure=STOP_DIRECT_WORD_BRIDGE_V1, no re-opening by moreprompts/LR/epochs/encoder.
3pairedseedbootstrap weak;64prompts not64independentmodelseeds. P1/P2thresholds
unchanged; old numericXOR100/recall never counts for this application criterion.

## Actual Qwen comparison / resources

Existing pinnedQwen3-4B-Instruct-2507Q4_K_M, SHA from verifieddescriptor. Independent
CPUreference on same64DEVsingleuserprompts, no system/history/tools/target/family
or SCAR-Xreply supplied. Temp0/seed919/ctx1024/max_tokens64/t2/ngl0/no-mmap,
ownedloopbackserver stopped in finally. No9B/autostart/modeldownload, no proxy.
Referenceup to4.5GB explicitly ABOVE SCAR-X2GB; run separately from Torchtrials,
reportRAM/truncation/latency, no equalcost/equivalence claim or canonicalexactgrade
of Qwen's different validstyle/identity. RawQwenanswers NOT traininglabels.
Qwen64referencecap600s/startup90s/request30s; incomplete outputs retained.

SCAR-Xpilot2GBprocess+tree cap, CPU1thread/worker,max2NumPyworkers/oneTorchchild,
300s/trial/2400sround includingevaluation/coldverification. TRAIN-onlytiming64,
forecast<300/trial and15trialforecast<2400; otherwiseSTOP_BUDGET_NOT_FEASIBLE.
No higherbudget silently. TechnicalIO/reportrepair only unchangedmath/data/gate,
priorcost accounted, completedcheckpoints never retrained. Preserveallfailedlogs.

Passive localhost followsactualCLI/review/training/outputs/loss/ΔW/neuron andQwen
with source/time/CONTROLlabels. ALL64DEVanswers from latest completedtrial,
includingerrors, no selector/runbutton. Publicproofonly; no privateHermes/session
contents or credentials packaged. Maincode/privateweights/config remain unchanged.
