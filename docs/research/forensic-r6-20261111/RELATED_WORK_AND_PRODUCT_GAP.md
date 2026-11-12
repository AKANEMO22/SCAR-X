# Related work and remaining advantage claim

SCAR-X has **not demonstrated an advantage over established methods** under a
shared benchmark, capacity, initialization, training budget and memory/runtime
measurement. Its local2GBproduct objective is still a research constraint, not a
measured superiority claim. No open-source framework was imported into serving.

Relevant primary work already addresses parts of the problem:

- [e-prop, Bellec et al.2020](https://www.nature.com/articles/s41467-020-17236-y):
  recurrent spiking learning with online eligibility traces and learning signals.
  Thus “no BPTT means hidden SNN cannot learn” is not a defensible conclusion.
  Its approximation, feedback and tasks must be matched before claiming that
  SCAR-X reproduces or improves it. This does not prove lifelong retention or
  useful natural-language/vision under SCAR-X's2GB/locality constraints.
- [DECOLLE, Kaiser et al.2020](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00424/full):
  local auxiliary readouts and forward state support spatiotemporal feature
  learning without full BPTT. It uses different local objectives/layer structure;
  it is not the same as SCAR-X's single recurrent head. The paper explicitly
  discusses local-error approximation and weight-transport assumptions.
- [Synaptic Intelligence, Zenke et al.2017](https://proceedings.mlr.press/v70/zenke17a.html):
  loss-linked synaptic importance reduces forgetting on tested continual tasks.
  SCAR-X's activity/update importance and output-local SI adaptation are not a
  validated reproduction of full-network SI. Low forgetting without acquisition
  or under a different task protocol cannot prove equivalence.

Fixed nonlinear recurrent dynamics with trained readout are a long-established
research family. R6D's frozen core and engineered query-conditioned features
need comparison to reservoir/linear-reader controls before any architectural
novelty claim. CKCER uses recognizable normalization, temporal-state and Oja/gating
ingredients. Its possible contribution is an empirically useful combination under
specific constraints; label it as such until matched ablations and fresh approved
testing distinguish the ingredients from extra tuning/changed conditions.

What SCAR-X has improved: auditability, explicit source/data/metric boundaries,
readout diagnosis, and promising small-event retrieval. What remains unsolved in
its own evidence: persistent useful hidden acquisition, robust transfer under the
same unseen perturbations, continual reuse, and useful neural natural-language/
vision output. Neither commercial-model equivalence nor autonomous “thinking”
is inferred from these experiments.
