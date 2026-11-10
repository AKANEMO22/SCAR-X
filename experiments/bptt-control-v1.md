# BPTT control v1

Registered before execution. Same seeds, 512 training examples, 128 held-out recall examples, separate delay-8 samples as phase2-v1. Same ALIF topology and initialization. CPU only. Surrogate derivative max(0, 1 - abs(v-threshold)), SGD 0.08, gradient norm cap 1. This is an isolated backprop comparison, not an allowed SCAR-X learning implementation. Train all weights, report this compute difference. No hyperparameter search. Continual retention and restart are not established by this control run.
