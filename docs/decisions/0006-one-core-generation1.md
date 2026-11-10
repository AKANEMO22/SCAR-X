# ADR0006 — One shared core / Generation 1 / observable idle processing

Accepted2026-11-09. User explicitly authorized implementation and replacing
v0.1/v0.2 product choices after correcting the independent pilot split.

One48-neuron ALIF population, one recurrent array/state, one active checkpoint
and one HermesID `scarx-generation1`. Text/digits use input channels and modality
slices of ONE readout; no routing to independent language/vision cores. Train
from scratch: old pilot hidden neurons have no justified alignment.

Bounded background thread advances SAME core. Foreground waiting preempts next
quantum, cancels unfinished probe and uses neural settling before new input;
runtime never reset_state. Idle probes do not learn weights. Generated text is
unverified. Trace records measurements, not invented verbal thoughts.

AtomicSQLite whole-session persistence, checksum, lockownership, learningrollback
and memoryguard. Session includes probe/cursor,memory,RNG,neuron/eligibility/
weights/stable/controllerpolicy. G1 has no importance arrays; plasticity-v1 stays
loadable by historical implementation, not coerced into G1.

Legacyweights/reports remain offline; ZIPs retired, personal sessions preserved.
No growth,rewiring,automatic consolidation/idlelearning,largepretrain,LLMproxy.
Jointtrainingfailedgreetinggeneralization; generalchat/vision/reasoning/actions
unimplemented. Phase2.1 gates stillfail; PIVOT_NO_SCALE.

Adaptivecapacity: conditionalactivationwithin48 first; fixed-budgetrewiring
second; growth/shrink after causal capacity evidence and separate approval.
Protect rare skills and migrate topology/eligibility/readout before pruning.
Inactivity is not uselessness; masks do not free allocatedRAM.

No Git initialized. Recommend private localGit for source, ignored localdata,
artifact manifests. No repo rearrangement or remotepublication.
