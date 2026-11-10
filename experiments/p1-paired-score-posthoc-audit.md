# Additive paired-score technical/attribution audit

Registered as POSTHOC descriptive analysis after viewing partial confirmation
stream. NOT a new preregistered primary quality endpoint, selection, training
round or altered gate. Run only after original run is terminal and receipt exists.

Inherit original decision, selection and primary statistics unchanged. Report
same-seed selected-minus-frozen/shuffled/clippedSGD/clippedAdam probe/direct
contrasts (bootstrap95 and signflip/Holm3tasks), clearly exploratory except
already registered primary frozencontrast. Never choose a new winner here.

For each selected checkpoint: newPythonprocess continuation on discarded
random-input clones, same RNG/neuron/eligibility/moments/readout state; compare
whole checkpoint hash. No original weight modified, no test label training.
Feature diagnostics use64existingheldoutinputs,8rolloutmeans, labelsunused:
fraction outside fixed normalized clip5, nearlyconstant neuron fraction at
raw-featurestd<1e-4, covariance participation ratio tr(C)^2/tr(C^2), original
direct-vs-refit gap. Correlations are not causal retention/normalchat proof.

All source/checkpoint paths constrained to project artifacts/runs. CPU1thread,
subprocess20s max each, no new model/data/gate/serving import. Retain the original
raw run and every output. Script: scripts/audit_paired_score.py.
