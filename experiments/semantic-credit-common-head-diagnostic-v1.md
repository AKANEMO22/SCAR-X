# GPU-only common-head diagnosis of signed-credit501749

User latest: all model forward/learningGPU; CPU onlyI/O/audit. Original501749
failed terminalJSONserialization (numpybool), not silently converted toPASS;
ff274a rescored saved60matrices/60checkpoints, actual1446updates/9126forwards.
eta.20DEV79.17% acquisition but5.83ppmean/30ppworstforget; familytwo-rate screen
closed. No extra rates/training/TESTcalls or thresholdchanges.

Reuse literature/prior matched-head b71d81: independently fitted identical
readout on each representation is needed before attributing weakonlinehead
comparison to usefulhiddenfeatures. New question is signed-error features,
not replaying prototypeprobe. No new synapse learning, allfeatures frozen.

For same224seenTRAIN/120DEV seed260113, extract8tickALIFactualfeatures from
final eta.05,eta.20 and same frozeninitial inputweights. Max1100GPUforwards/
60active seconds/512MiBphysicalreserve/2GBworkerRSS. Persistentprecharge.
No new ALIFupdates; noTESTrows selected. One registeredattempt and stoponerror.
Keep source9126/1446charged; diagnostics separatelyreported, not refunded.

Commonreadout: TRAINmean center128D, unitnormalize, appendconstant1, alpha1ridge
on129coordinates. Solve/evaluate onGPU (CuPy), no CPUfitting. Samefeatures/
labels/normalization/equation as earliercommonhead. Fresh fixedcontrol means
same initial model/seed andnewfixedpooledhead, not trainedheadvalues.
Saveactual features, commonreadout coefficients and DEVpredictions, stratified
source-textbootstrap10000/RNG260216. One seed only, reusedDEV andposthocprobe:
even positiveCI isdiagnostic, not fresh3seedconfirmation or SNNvsstrongRAGPASS.
Report whether learnedrepresentation helps undercommonhead, and gap torawPCA64
ridge79.17% onall240TRAIN (224observedTRAIN countdiff explicit).

Engineering recovery after20d178: missing nvJitLinkDLL blocked CuPy directsolve
after344forwardattempts, before any fittedhead/DEVscore. KeepFAIL and344charged.
Do not install anotherCUDAstack, switchCPU or lowerprecision. One isolated
recovery uses GPU conjugategradient on the SAMEpositive-definite ridge system,
fixed256iterations/FP32, actual relative residual must<=3e-5 orSTOP. Verify known
SPDsolution onGPU beforefeatures; no ALIFupdate. Recovery max1050forwards and
55seconds, original4.531seconds +344forwards retained explicitly. Combined with
source9126 remains below12000forwards; no originalresult is changed. Allthree
conditions still required, no dropping an arm/sample to fit quota. Preserve
features beforefitting so anotherfailure cannot hide consumedobservations.
