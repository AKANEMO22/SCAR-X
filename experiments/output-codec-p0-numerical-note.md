# Numerical compatibility note — P0, before final gate run

The first migration fixture exposed a full-width float32 GEMV difference:
expanding222input columns to253changed voltage by4.76837158203125e-7 after one
word step, despite unchanged mathematical input and identical spikes/features.
The exact-parity gate remains zero; this failed naive method is not a pass.

The research prototype will preserve the legacy input block's C-contiguous order
and evaluate added word-input columns as a separate block. Legacy word+digit
output rows retain their original contiguous operation order too. This is a
numeric compatibility kernel with identical ALIF/STDP equations, not a new credit
rule, representation mechanism or performance optimization. No serving source
is changed. Record naive vs compatibility errors separately in the P0 run.

Initial prototype scope is the public48neuron release, active48, without a new
capacity allocator. Other allocated sizes/foreign learner state are rejected
rather than silently losing state. Main's existing48–128capacity policy is not
changed or claimed certified by this public-H48codec fixture.
