# Joint question/RGB to autoregressive word readout v1

Registered2026-11-10 BEFORE code/data/performance. Master§4.1/audit31 permits
independentprototypejointinput while proposal30 STILLPENDING. No newhidden
learning/errortransport/BPTT/Transformer/pretrained/RAG/mainweightschange.
This is localREADOUTskill feasibility, NOT hiddenrepresentation learning,
normalphotos/OCR/generalVQA/lifelongretention or reopeningclosedscore/byte/
localclassifier/observedassociation rules. No memory/decoder/LR/grid search.

One deterministicSpikingCoreALIF H48/K8/R12, sameG1leak.65,threshold.5,
adaptstrength.02,adaptdecay.95,fdecay.25. D=V+198,O=V+10 reserved digitrows;
freshseedinit, subtract.15w_in, stablecopy. ALLhiddeninput/recurrentweights
FROZEN. Vocabulary ONLY actualtrainrenderer targetwords (vi/en colors/shapes/
counts)+BOS/EOS, no answerclass/template lookup. One core, onewordhead; head
ConditionedReadout normalized with train-onlycalibration, outputlocaldelta
.08,updateclip.025/weightclip3. No teacher/readout error reaches hidden.

Question numericfeature=sum InputEvents.token_vector for tokenizedquestion,
divide by max(1,L2); at most64tokens/2048chars. No intent/language/labelparser,
trainedembedding, devstatistics or target supplied to prepare/generate.
Prepare: resetneurondynamics,12zeroevents,3BOSquestionticks likeG1wordinput
(1.6questionfeature,gain.8previousword,position0control). Then existing
InputEvents.image RGB16×16→16patches+2boundaryevents, SAME64imagechannels.
Overlay1.6questionfeature in128contextchannels throughout18imageevents and
set existinglanguageflag1; imageflags/coords unchanged. Allquestion/image
phases fixed33ticks, no questionlength/kind/countmetadata leaks into timing.

Workingmemory: cache48filtered-spikefeatures averaged over16contentpatches.
This is a boundedvisualneuraltrace, not rawpixelretrieval/template/secondcore.
For eachoutputword,3existingG1wordticks with previousword and questionfeature,
imagechannels0; readoutfeature=.5*(cachedvisualtrace+currentneuralfeature).
Fixed mixture, no holdlength/tracegrid or reopening byteworkingfamily.
Onlycurrentoutputreadouterror multiplies this presynapticfeature. Teacher
PREVIOUSwords allowed duringtraining; freegeneration previouswords exclusively
NEURALoutputs. Max8words, sameword/BOS/EOS/punctuationmaskpolicy asG1, no mask
byquestionkind/language/rendererclass. Targetsentence not an inferenceinput.
Cache/question/neuronstate/RNG/calibration/counters must save/resumeexactly.

Dataset createdPROCEDURALLY, ownedRGB128×128, palette sameaudit31 red/green,
circles/squares/triangles and1–3objects. Renderer verifies nonoverlap, connected
components andpixelpalette. No printedlabels/answersinimage, no privatephotos.
Trainimage RNG33101, dev33103, no samePNGhash across splits, failduplicates.
96train/96devscenes:
* single24: train(circlered,circlegreen,squarered,trianglegreen)×6geometry;
  dev(squaregreen,trianglered,circlered,circlegreen)×6. Firsttwo combinations
  genuinelynovel; circlesseencomposition/newgeometry, reportboth separately.
* two36: train orderedshapes(c,s),(s,t),(t,c); dev reversed(s,c),(t,s),(c,t).
  EACH3shapeorders×all4red/greencolorpairs×3geometry. Devshapeordersnovel.
* three36: train cyclic(c,s,t),(s,t,c),(t,c,s); dev reverse(c,t,s),(t,s,c),(s,c,t).
  EACH3orders×colorpatterns(rrg,ggr,rgr,grg)×3geometry. Devordersnovel.
Singlecenter(x64±3,y64±3),halfsize20..28; twocentersx36/92±3,y64±4,half14..18;
threecentersx24/64/104±2,y64±3,half12..16. Integeruniform inclusivebounds.
Never repair split/selectgeometry based on modelperformance.

Questions=actualreviewed32Antigravitywordings from audit31. TRAIN IDs perkind
01/02/05/06 (2vi/2en); DEV03/04/07/08 (differentwording2vi/2en). Eligiblekind
mapping sameaudit31. Exactly864train/864devQA cases. Outputs bilingualrenderer
expected() strings, not teacher/Qwen labels. TrainwordvocabmustexpressallDEV
targets beforetraining. Wording/devdata inspected historically; exploratory,
NOT freshfinaltest or naturalphotoquality. Allnovelcompositionmembership
determined by shapes/colors/order signature BEFORE reading labelscores.
AutomaticlearningimportFalse; these are explicitsupervisedresearch examples.

3freshnetworkseeds33113/33119/33133 ×3matchedTRAINconditions(full,no_image,
no_question)=9trials. Blanking zeros ONLYimageRGB48values or ALLquestion128
channels in EVERYphase, retainsfixed33+wordtickcounts andfixedcoords/control.
No questionlength sidechannel. Decoderallvocabshared across tasks/languages.
Allconditions sameweights/initialization/sampler/calibrationindexschedule,
1536supervisedexperiences. Sampler seed+12100000: uniformkind4→language2→
uniformtrain-targetstratum withinkind/lang→uniformcase. This uses labels for
supervisedbatchbalancing ONLY, not modelinput/router/masks. Record allcounts.
Calibration128UNLABELEDtraincases seed+12200000, randompreviouswordprefix0..2
fromtrainvocab, not actualanswers. ObservepooledfeatureafterBOS/randomprefix,
freeze mean/std BEFORE labeledreadouttraining. No calibration/hiddenupdates
from dev, language/intenttargets or futureanswerwords.

OnlyFINAL1536checkpoint, no winner/epoch/LR/mixture/encoderselection. Evaluate
all864DEVcases and first64fixedTRAINdiagnostic; strict/normalizedfreeutterance,
teacherforcedwordNLL/accuracy/EOS/firstword, macro by(kind,language), everykind
andlang, novel/seencomposition separately, actualticks/cachebytes/time/RAM.
Report pairedseedbootstrapCI95 across3models, not casesasmodels. No taskoracle
orclass-ID formatting allowed to replace freewords.

FULLtrainedhead gets samecases with threeCAUSALinterventions (neverretrain):
blankimage,blankquestion,swappedimage. Swap to FIRSTeligibleDEVscene with a
DIFFERENTrenderertarget for samekind/language; target usedONLY to construct
interventionassignment, never given tomodel. Keep originaltarget scoring,
record mapping/hashes. These are distributioninterventions, not independent
models; separatelyTRAINEDnoimage/noquestion controls distinguish shortcuts.
InitialFULLheadcheckpoint (untrainedoutputweights) is saved beforelabels and
evaluated onlyAFTER finaltraining; no initialDEV-basedselection/adjustments.

Informative ONLY if FULLmacroDEVfreeexact≥65% onALL3seeds, eachkind≥50% onALL3,
novelcompositionmacro≥60% onALL3, meanpairedgainsvsTRAINEDnoimage ANDnoquestion
≥20pp positive onALL3 and bootstraplower95>0; FULLmean dropvs EACHblankimage/
blankquestion/swappedimage≥20pp. Allcodec/locality/targetfreecheckspass.
Positive=FRESH_VISUAL_WORD_VALIDATION_REQUIRED, NOT normalvision/mainP1/P2,
hiddenlearning or mainactivation. Failure=STOP_VISUAL_WORD_READOUT_V1;
one fixedscreen, no automaticround2/featuremixture/LR/epochs/modelscale/search
to reinterpret failure. Historicalgate/STOPs remain. Any later hypothesis must
address a distinct cause with explicitprotocol, not rename thisroute.

Beforetraining: actualinput tapes containquestionandRGB (unlessablated), fixed
tickbudgets; no rendererattributes/currenttarget enter prediction; hiddenweight
fingerprints unchanged by localwordtraining; outputerroronly athead; ownformat
nonpickle/bounded/checksums, interruptedprepare→nextword/generation/learning/
RNGcontinuation, corruptionrejection. Fullpytest/nativeparity. No serverimport.

CPU1thread/worker,max2workers,2GBtree/Windowscap,300s/trial,2400sround. Dataset
build120s. StopNaN/targetleak/format/source/budgetfailure, preserveallrawresults.
Actuallearningloss/update/ΔW/neuronframes and image/question/wordoutputs go
passivelocalhost, no runbuttons/modelselection, no mockedoutputs. Qwen4B current
text-only is NOT visionbaseline; no fakeimage-as-text/newVLM/9B/teacherAPIcall.
No modification of liveuserweights/vocab/Hermes config or proposal30approval.
