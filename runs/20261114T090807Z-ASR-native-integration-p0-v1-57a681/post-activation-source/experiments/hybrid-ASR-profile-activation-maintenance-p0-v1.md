# Selected-ASR operational activation addendum — before service restart

The four locked ASR calls in57a681 and data-only audit already passed, with no
live source/profile change. Final8h602b72 audit passed engineering only. The
original native integration protocol and 4ASR/4VLM aggregate limits remain.

Confirmed operational race: the open Hermes plugin polls and may auto-start
the gateway while its source/config transaction is in progress. Add one short
PID+creation-time/nonce maintenance lease checked by run_hybrid.py before serve.
It blocks fresh gateway starts during shutdown/apply, does not stop Hermes,
and never blocks an existing foreground request. Invalid/partial leases fail
closed; provably dead/reused owner is recoverable; release deletes only this
transaction's lease. No daemon, port, model, background inference or privilege.

Four unit checks cover distinct-process blocking/release, stale PID recovery,
partial metadata and replacement-owner preservation. First fixture failed
because child PYTHONPATH omitted this worktree, before any live lease/action;
fixed by explicit owned source root, failed JUnit retained alongside passed.

Operational driver activate_hybrid_ASR_profile.py requires idle live service,
actual completed prerequisites and exact reviewed six targets. Keep an owned
lease across graceful stop, source/config activation and full saved-state
normal HybridEngine-constructor restoration, which runs no model workers.
Reserve the three mandatory VLM parity attempts before releasing the lease;
whether Hermes or the driver wins subsequent startup, one writer lock applies.
Wait for actual ready plus exactly three fresh parity events, then reserve the
fourth fixed arithmetic probe. The probe goes directly to owned native worker,
never conversation/review/training. No fifth VLM or fifth ASR call.

Compare all learned arrays plus complete non-controller metadata, including
RNG/replay/update counts, after startup. Full controller byte equality must be
proven on normal saved-state restoration before background start; controller
clock/phase may change during normal operation and is disclosed separately.
Record new PIDs, source hashes, selected ASR storage and execution placement.
This activation is not a new live ASR inference or small-profile long-run pass.
Original8h and private call source archives stay immutable. No SNN rule/LoRA/
hidden learning change, no main/peer worktree change, no production quality claim.
