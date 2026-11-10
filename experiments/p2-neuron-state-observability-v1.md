# P2 existing neuron-state observability diagnostic v1

Registered2026-11-10 BEFORE measurements. Distinct hypothesis from stopped
timescale/temp/worddecoder views: keep ALLALIFdynamics/weights unchanged, ask
which ALREADYEXISTING neuronstate exposes useful inputinformation to a local
outputhead. No hiddenruletraining/familyreopening, scale or runtimechange.

Source same6offlinejointfull/frozen checkpoints/3seeds from P2diagnostic, all
included. Same3840train and20×128heldoutsamples. MC8,temp0.4,prefix12 ONLY;
same commonuniformnoise for every readoutview. FiveH48views (same outputparam
budget): filteredspikes(original), lastspikes, membrane_afterreset, adaptation,
membrane_afterreset/(threshold+adaptation_strength*adaptation). No concatenated
rawinput/cue bypass, taskID, template/RAG, extra neurons or hiddenfeatures.
Stateview f must bitwise match original features_many, source.dumps unchanged.

Neuralcore remains ALIF with spike-mediatedrecurrentconnections. Reading
presynapticcell membrane/adaptation through an analog channel is an EXTRA
informationtransport assumption, not pureaxon-spike communication or proven
biologicalsynapse. Allowed numericalreadout research, never claim event-only
efficiency or unchangedbiologicallocality. Proposed futureoutputlocal updates
could use their own presynapticstate×postsynapticerror, no globalBPTT; NOT
implemented into main/hiddenlearning by this diagnostic.

Ridge4classprobe(.01, train-onlycenter/std) diagnostic upperreference; output-
localsoftmaxhead5fixedtrainpasses, lr.08,updateclip.025,weightclip3, normalized
trainstats floor.05, orderseed2003. Features/labels identical across algorithms.
No sourceheadtraining, replacement/probeheaddeployment or futurelabels in a
lifelonglearner: source offlinejoint explicitly used FUTUREtraininglabels.

Descriptiveinformative screen, NOT P2gate: endpointlocalheadmean>=65%AND
meanpairedgainvsfilteredspikes>=5ppANDpositiveeffect>=2/3seeds. Ridgealone
doesnotopen runtime or claimlocallearning. Assess everyview, report ALL, no
new grid/temp/MC change iffails. Ifinformative, propose separatelyregistered
fresh localreadout/retention study and guard compatibility; sourceweights
never imported. Ifallfail, stopthisview-screen, no unlimitedfeaturetuning.

Save all statevectorslocally, targets/predictions/publicaggregate newrun.
Publish actual progress and classificationoutputs with analog-readoutscope;
do not fabricate modelchat/Qwen. CPU1thread,sequential6models,2GBcap,90s/model,
600sround. Stop on source mutation/nonfinite/budget/parity failure.
