# Technical audit of jointvisualword readout

Registered before audit execution. All9finalcheckpoints, no winner/quality
changes or newtraining. Verify sourcecode/protocol/data/imagesSHA,1536experience/
128calibration counters, same sampler/wordupdates/trainingticks byseed, hidden
input/recurrentweights exactlyinitial for full/noimage/noquestion, owncodec
checksum/workingmemory/headalias consistency. First2recovered models must be
byteidentical to the1536checkpoints from original failedaggregate run; no
training repeated. Preserve originalNaNmetric/wrapperfailure journals.

FreshPythonprocess foreach9checkpoint: on DISCARDEDcopy prepare firstfixed
publictrainimage/question, runBOSwordstep, save/load there, runnextwordstep,
generate withouttarget, learn ONEknownexample, drawsamplerRNG then fullstateSHA.
Compare exactly to parentdiscardedcopy. This is technicalcontinuation only,
not another qualityexperiment or extraformaltraining; sourceSHAunchanged.
Core.py/unified.py hashmatch boundaryreceipt. CPU1thread,2GB,120s wholeaudit.
No liveweights or pendingproposal30implementation. Softwarepass cannot
overturn STOP_VISUAL_WORD_READOUT_V1 or demonstrate hiddenlearning/normalvision.
