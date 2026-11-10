# P2 fixed timescale-view diagnostic v1

Registered2026-11-09 BEFORE measurements; proposal in master/report22 first.
Same6offlinejoint checkpoints/3seeds. No training, hidden-rule tuning, task
change, extra neurons/synapses, deployment or efficacygate. Oldweights optimized
under olddynamics; failure cannot refute future learning with heterogeneousALIF.

Ten fixed views, no grid extension: baseline; leak_spread/leak_mean;
adaptation_spread/adaptation_mean; feature_spread/feature_mean;
all_spread/all_mean; all_independent_spread. Each parameter has baseline
tau=-1/log(decay). H48quantiles log-uniform tau0*[1/8,8], deterministic
permutationseed+8600000. Spread cases use these neuron-specific fixedtau;
meancontrol uses SAME arithmetic mean tau as corresponding spread population.
All_spread shares permutation across3parameters; independent uses next
permutations from sameRNG. Constants not learned or fit to labels.

Temp0.4,MC8,prefix12 ONLY. Same source3840train/2560heldoutdata, same nested
uniformnoise perview. Refit4classridge(.01) and originalhead measured separately.
Save decayvectors and all outputs; assert core.dumps unchanged. Baselinevector
forward must bitwise match original scalarALIF before running.

This is posthoc exploratory source-data reuse, not freshconfirmation. Descriptive
informative screen: meanendpointprobe>=65% AND meanpairedgainvsbaseline>=5pp
AND positive gain on>=2of3seeds. It is NOT P1/P2success or permission to scale.
If informative, freeze hypothesis for separatelyregistered fresh neural-learning
test; no backdoor production/refit deployment. If not, stop this view-screen,
no unbounded spread/window/temperature tuning or invented abilityclaim.

MC/compute same acrossviews. Report meancontrol-vs-spread effects, fixed3seed
bootstrap exploratory, neuronmetadata576bytes and timing/RAM. No borrowed
paper accuracy/energy claim: original heterogeneitypaper uses different
dynamics/training including surrogategradient/FORCE.
https://neural-reckoning.org/pub_heterogeneity.html

CPU1thread,sequential,2GBcap,120s/model,600sround. Stop on NaN/state mutation,
parity/budget failure. Publish actual progress/outputs; Qwen unmeasured stays
missing. Retain all prior results and source/checkpoints unchanged.
