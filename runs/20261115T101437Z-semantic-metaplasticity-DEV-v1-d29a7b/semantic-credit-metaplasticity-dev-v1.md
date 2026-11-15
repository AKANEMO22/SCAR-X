# Three-seed GPU local importance / fast-stable synapse DEV study

User15/11 authorizes isolated localALIF improvements; latest requires all neural
forward/trainingGPU. NoCPUfallback, modelscale/precision/corechange, LoRA,
TESTreuse or livepromotion. Prior failed501749 remainsFAIL; ff274a recovered
componentmetrics withoutretraining. Freshhypothesis is synapticprotection after
improvedcredit, not morelearningrates ontheclosedtwo-rate family.

Evidence: signed-error common-head9383b3 DEV80% vs samefrozen65.83%, gain14.17pp
conditionalone-seed CI positive, diagnosticonly. This closes one prerequisite
better than positiveprototype b71d81; not proof of strongerRAG/fullassistant.
Currenteta.20 actualheadDEV79.17%, meanforget5.83pp/worst30ppstillFAIL.

Literaturefirst: reuse SI paper https://proceedings.mlr.press/v70/zenke17a.html
and author optimizer pin86c5e07c5603d6a44c2f53585c19ee70773ef15c lines96–172
readagain. Its Keras/TFglobalgradients/taskupdates are NOTlocalSCAR-X, no codecopy
or SIreproduction claim. Reuse idea of per-synapseimportance andslowanchor;
old SCAR-X inertimportance/protectionFAIL remain. New question is whether a
definedforwardeligibility importance now protects genuinelytask-alignedfeatures
instead of merelyreducing overallupdates. Matched-normcontrol is mandatory.

## Fixed rule, no hyperparametersearch

Same originalALIF8ticks/input64/H128/fan8/FP32/fixedpooledG as501749.
d_ij=(G.T*(onehot-p))_i * e_f,ij, input synapses only; no recurrentupdates.
I <- .99I+.01d^2. Perneuron dendriticnormalizer q=I/max(mean_j(I),1e-8), clipq[0,10].
Delta=clip(.20*d/(1+q) - .002*q*(w-wstable),-.025,.025); weightclip[-3,3].
wstable <- wstable+.005*(w-wstable) aftereach acceptedTRAIN/replayupdate.
I0=0,wstable0=w0, no taskboundarysignals, no DEVguard/tuning. Allsynapsesremain
allocated. Per-row mean is neuron-local normalization, broader than an isolated
synapse; recordthis assumption, not strict chemical neuron replication.

Arms: currenteta.20rule; importance+slowanchor; normmatchedcurrentrule using
same sample/label andthe protectedarm's realizedupdateL2 scalar only. Matchwith
monotonicbisectionscaling of own d before sameelement/weightclips, error<=max(
1e-6,1e-4*targetnorm); ifunreachableSTOP. Globalnorm scalar is an evaluation
control, not newlyclaimed biologicallylocalalgorithm. No protectionstate/head
or futureDEVoutputs transferred tocontrol. Same computationbudgetcap for each
comparison; report actualwall/forwards/normerror rather than claimidenticalcost.

Seeds260113/260137/260171,20identical archived TRAIN/replay64 schedules from0748ea.
Reuse seed260113 unprotectedcurves/weights from501749 (samecode/codec/stream),
do nottrainagain. Other2unprotectedseedstates arefresh; protected/normmatched
all3fresh.24attempts/stage,120DEV/stage,0TESTcalls. Same24sourceexposures/epoch
and224uniqueTRAINobserved perseed; no replay/data additions or taskIDinputs.

Budgets newhypothesis max4000updates/30000forwards/240seconds/100MBartifacts,
PythonRSS2GB/512MiBfreeGPU. Persistentprechargeincluding6GPUfixtureupdates;
noreset/refund. GPU-only clone/coldresume/rollback P0; schema rejectsCPU/wrong
codec/rule/counter. SNNoldsourcebudgetsremainclosed, allnewbudgetsexplicit.
Nonfinite/resource/checkpoint/normmatch failureSTOP; no confirmationretry.

DEVscreen scientificgates unchanged: acquisition>=65%, finalmeanforget<=3pp,
worstintent<=5pp inEVERYseed; compare5pp/lowerCI>0 vs strongestregisteredraw
RAG/classifier andreadoutcontrols beforeanyfreshconfirmation. Reportallseeds
includingnegative, gainsconditionalonreusedDEVnotfreshTEST. No broadsuccess
fromonegroup/zero-updateforgetting or selectingwinnersonTEST. Ifprotection
doesnotimprovejointgate, stopthisrulevariant, diagnose, don'taddmoreparams.

PublishactualGPUloss/synapseupdates/importance/anchordrift/normmatches/DEVcurves
andguardevents tolocalhost, nofake thoughttext or livecounterincrements.
