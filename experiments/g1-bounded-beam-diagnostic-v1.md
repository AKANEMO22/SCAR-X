# G1 bounded neural-token beam diagnostic v1

Registered2026-11-10 BEFORE decoder measurements. Masterreadout/finiteworkingstate
proposal report24. Final bounded G1worddecoding study after historicalsentence-
stop audit failed; NOT another stoppedbyte-memory/localclassifier/scoretraining
round. No neuralweights/training, cannedresponses, label-inference or deployment.

Threefixed surface_expanded curatedcheckpoints, allorders1801/1811/1823. Exact
31dev +48priorAGY cases, allalreadyinspected, no newfinaltest claim. Original
greedy24word decoder remains baseline and production. Sixconfigurations:
beamwidth1/4/8 × lengthalpha0/0.6. Width1alpha0.6 duplicatesgreedy and is an
identitycheck; report actualcomputation. No extra penalties/norepeat/targetrerank.

Onecloned model with sharedimmutableweights; snapshot ONLYstatearrays, ticks,
lastactiveclocks and activecount perhypothesis. Restorevirtualstate perexpansion,
word_step(previous,context,position) once then topwidthpositiveprobabilitytokens;
children shareimmutableafternode snapshot until expanded. Same vocabulary/mask,
BOS/EOS/boundary as existingmodel. No data/targets/retrievedtext in decoder API.

Keep atmostwidthactive +widthfinished nodes. Active same-length prefixes ranked
sumlogP; finished ranked sumlogP/(((5+emittedtokens)/6)^alpha), includesEOS.
Stop ifactiveempty,24wordlimit,256wordstepexpansions, or bestfinished>=upperbound
of allactive usingmaxallowedlength and futurelogP<=0. Ifnocompletepath, return
bestpartial with explicitlength/budgetreason, no fabricatedEOS. Width1 must
match existinggreedy text/tokens/stopped/neuralstate exactly. 256nodebudget
counts wordsteps, not candidate-tokens; rootboundary12ticks included in actual
computationticks. Selectedpathvirtualticks separate from totalworkticks.

Before measurements tests prove greedy/stateparity, boundedtermination, fixed
weightobjectidentity/fingerprint and no target parameter. Snapshotmemoryunique
arrays actualbytes counted, not nparams×width; weights once. This implementation
changes virtualneuralclock duringdiagnosticsearch; no mainclock/trace rewriting,
no claim productionalways-runningneuronpolicy is alreadycompatible.

Select width/alpha only by developmentmean normalized-styleaccuracy averaged
3orders, tie lowerwidth then smalleralpha. Freeze choice BEFORE prior48 decode
for selected vs baseline; do notfallback based onprior. Informative only if
devmean>=25%ANDgainvsbaseline>=10pp, positive gain ALL3orders, priorgain>=10pp,
andcasep95<5s/no case>5s. This is a descriptiveproductdiagnostic, not a P1/P2gate,
normalchat/vision or usefulsynapselearning claim. Exactstyle/normalizedstyle,
firsttoken, repetitions, EOScompletion, oldcorrectlost and CPU/ticks/statebytes
allreported. A gain only from terminalpunctuation cannot pass normalizedendpoint.

If fails STOP_G1_WORD_DECODER_SEARCH, no alpha/beamgridextension/newcandidate
selection. If passes PROMISING_RESEARCH_ONLY; no deployment without separate
clock/statecontrollerengineering and fresh privateguard/vision checks. Do not
use an intent-to-canonicalanswer table even as a successfulfallback.

CPU1thread, sequential,2GBcap,5s/case,300sround. Budgetfailure saves incomplete,
not qualitypass; stop on NaN/weightmutation/greedyparityfailure. Publish actual
tokenoutputs/progress/neuronsamples, no fakelearningloss or Qwen. ArchivedQwen
exactpromptattach may be read independently. Privatecheckpoints/rawoutputs local.
