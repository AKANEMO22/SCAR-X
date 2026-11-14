# Posthoc data-only decomposition of fc84eb — before derived output

Original48case direction diagnostic is fixed. Its initial median local cosine
~.867 vs prospective~.848, final640local~.724 vs prospective~.139 is an observed
direction mismatch, not a proof learning would improve if momentum changed.

Do not rerun neural forward/backward, train or select additional samples.
From archived directions and parentstate only, independently reconstruct the
current update: nexttrace=.9oldtrace+.1direction; nextenergy=.99oldenergy+
.01direction²; .008trace/(sqrtenergy+1e-4), clip±.01, weightclip±3. Compare recorded
prospective delta with absolute tolerance2e-7 (NumPy/CUDA rounding), derive all
48cosines/sign metrics independently against stored surrogate direction.

Report intermediate current-only numerator (.1direction), accumulated trace,
current-only numerator under SAME actual energy denominator, and actual
preconditioned/clipped delta. The current-only variant is an observational
counterfactual, not an approved learner or implicit recommendation. Keep same
energy even though a real changed algorithm would have a different history.

Report clip/weight-bound fractions and query-vs-record input credit norm. No
significance threshold or samplecount inflation; only8oldtrainingpairs×3seeds×
2states. Reconstructed state comes from originalcheckpoint, initial arrays
documented0trace/1e-4energy. Check all state weights/indices/normalization match
saved state NPZ and original hashes. Audit does not rerun or independently prove
the BPTT surrogate derivation or local eligibility calculus; primary parity,
unitcheck and archived source are separate evidence.

Budget60seconds CPU data arithmetic, no GPU inference/updates/network/newfiletools,
same failed originalquality/retention gates. Momentum against a single example
may intentionally protect other skills; surrogate alignment is not performance,
causality of forgetting, or a true gradient of hard spikes. Do not promote any
formula on this decomposition alone. Preserve original diagnostic and report
as a supplement with its own source/hash/protocol and explicit posthoc scope.
