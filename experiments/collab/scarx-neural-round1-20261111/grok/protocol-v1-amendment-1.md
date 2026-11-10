# Protocol v1 amendment 1 — confirmation split only

This amendment does not change the hypothesis, seeds, training budget, arms, or
the 0.05 lower-bound gate. It corrects a technical error in the runner at
`14bf4b2574d77052e3c8d4d682238b2dbd5320c5`.

The locked confirmation data seed is 71005. That runner scored the eight
confirmation initializations on development seed 71004. `runs/grok-slow-trace-v1`
remains the record of that mistake and is not rewritten.

The corrected runner writes only
`runs/grok-slow-trace-v1a-confirmation-71005`. It repeats the unchanged
development screen on 71004, and, only if that screen still opens confirmation,
scores the same eight seeds on 71005. The interval uses a sample standard
deviation with one degree of freedom and Student t = 2.3646242515923784 for
df=7, two-sided 95%. The bound stays 0.05.
