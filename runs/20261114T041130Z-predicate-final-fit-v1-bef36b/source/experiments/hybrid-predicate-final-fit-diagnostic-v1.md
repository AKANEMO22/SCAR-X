# Final-state fit diagnostic —14/11/2026 before additional inference

Parent032f30 completed12cells/3seeds/20stages/868.266s, protected53/57/49.25%
unseenDEV, alljointgatesfailed. Independentrescore corrects seed-onlyCI to paired
seed/familyCI[-.75,6.33]pp, pointgain2.5pp; original retained. No trainingdata/
threshold/tuning change. This diagnostic cannot promote the failedcandidate.

Freeze original final-full-state checkpoint (640scoredupdates, not additional
coldnextupdate641). For each3seed×3learningarms readout_only/current/protected,
run inference on320seen trainingpairs and400heldoutfinalDEVpairs; verify latter
against original prediction/probabilitytable. No learn/consolidate, normalization
fit, labelreward or modifying8h/liveweight. CUDAneuralforward only.

Report trainaccuracy vs unseenaccuracy to distinguish failuretofit vs poor
generalization. Count |standardized raw|>=5 channels (localfeedbackclipped mask)
and actualactive spikes as observational diagnostics, not proofofcausalforgetting.
Check alltrainable/stableweights+normalization/importance, replay/RNG unchanged.
Inputs/labels originalownedSQL/Pythonverifiedbank; not natural-language tasks.

Budget≤5minutes/6480forwards (9×720), no newdatasets/model/teacher/bankselection.
Holdfailedjointgate/5pp/3-5ppcriteria. Publishmeasuredfitdata8775 asorchestrator.
