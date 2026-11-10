# Protocol v1 — slow forward eligibility at G1 feature decay

Campaign: `scarx-neural-round1-20261111`
Agent: `grok`
Environment: Cursor on Windows. Model that wrote this protocol: Grok 4.7.
Ponytail: not available in this session. No ponytail run was pretended.
Baseline: `1735ad66ea9e7b416f5df15df99d9c13cea46b4e`
Branch: `research/grok/scarx-neural-round1-20261111`

This file is the locked protocol. A formal measurement run is valid only if it
starts after the commit that adds this file. Later edits are amendments, not
silent replacements. Trial 35 remains STOP. This protocol does not reopen it
by changing the learning rate, the epoch count, or the name of `e_f`.

## Authority and limits

The round-1 prompt allows a research prototype that multiplies the one-step
readout error `W_out.T @ error` by a forward eligibility at the same neuron.
That signal is not biologically local. This protocol does not add
hidden-to-hidden error transport, BPTT, RTRL, autograd, a Transformer, or an
LLM/VLM proxy. BPTT is not the candidate. Nothing here is imported into the
serving core or Hermes. The live process on port 8765 and `local/generation1`
are out of scope.

## Hypothesis

At Generation-1 `feature_decay = 0.25`, the fast feature that proposal 30's
`e_f` differentiates has decayed before a delayed query, while a neuron-local
spike trace with the same decay constant as adaptation (`0.95`) can still carry
the earlier event. Multiplying the approved per-neuron error by the forward
sensitivity of that slow trace, and reading that slow trace at the head, assigns
credit the fast feature path cannot.

This is not trial 35 again. Trial 35 kept `feature_decay = 0.92` and multiplied
the error by `e_f` of that slow-enough feature. Here the neuron's own feature
decay stays at the Generation-1 value `0.25`. The new state is an extra forward
trace. It does not enter the voltage recurrence.

## What would refute it

- Before any synapse update, a ridge probe of the fast features on the
  development sequences is already at least 0.65. The delay did not erase the
  fast feature, so the premise is false for this task.
- Before any synapse update, a ridge probe of the slow trace on the development
  sequences is below 0.40. The slow trace also does not hold the label, so
  changing which eligibility is multiplied cannot be the missing information.
- The information gate passes, but after the locked training budget the slow
  candidate does not beat both the slow readout-only arm and the fast `e_f` arm
  by the screen rule below. The trace may hold information while this credit
  signal still fails to use it.

A pass of the information gate is not a learning result. A weight change is not
a skill. This protocol does not claim conversation, vision, or lifelong retention.

## Equation

Generation-1 forward constants, not trial-35 defaults: hidden 48, fan-in 8,
plastic rows 12, leak 0.65, threshold 0.5, adaptation strength 0.02,
adaptation decay 0.95, feature decay 0.25, anchor 0. The Adam step, clip 0.002,
and weight clip 3 stay the proposal-30 research values already in
`ReadoutEligibility`. They are not tuned here.

Slow trace, forward only, decay locked at 0.95:

```
s <- 0.95 s + 0.05 z
e_s <- 0.95 e_s + 0.05 z_e
```

`z` is the spike. `z_e` is the same one-step spike surrogate already computed
for `e_f` in `ReadoutEligibility.step` before the neuron advances. `s` is not
added to the voltage.

The head error is the one-hot residual on the state the head actually reads
(`features` for the fast arms, `s` for the slow arms). The per-neuron factor is
`W_out.T @ error`, masked by the same normalized-feature gate as proposal 30.
The slow candidate accumulates `signal * e_s`. The fast candidate accumulates
`signal * e_f` through the existing class. Shuffled permutes `signal` across
neurons and still multiplies the matching eligibility. Zero and frozen do not
apply a hidden update. Every condition consumes one permutation draw.

## Data

Generator is local and deterministic. No external dataset and no model-written
labels.

Each sequence has 12 ticks and 8 channels, noise `N(0, 0.08)`:

- tick 0: `+1.5` on channel 0 or 1
- tick 1: `+1.5` on the other of those two channels
- ticks 2–9: noise only
- tick 10: `+1.5` on channel 4
- tick 11: `+1.5` on channel 5

The label is the channel written at tick 0. Both channels 0 and 1 occur once,
so a bag of the two writes does not determine the label. The query channel has
no label bit.

Splits, fixed before the run:

| Split | Seed | Count | Labels used for |
|---|---|---|---|
| Calibration | 71003 | 128 | not used; head statistics only |
| Train | 71002 | 64 | supervised head and hidden update |
| Development | 71004 | 64 | information probe and screen metrics |
| Confirmation | 71005 | 64 | used only if the screen below passes |

Model initialization seeds, development: `71011`, `71017`, `71023`.
Confirmation seeds, unused unless the screen passes: `71037`, `71039`, `71051`,
`71057`, `71063`, `71079`, `71091`, `71099`.
The information-gate network uses seed `71001` and is not one of the learning
seeds. The same train and development arrays are shared across model seeds.
Three seeds are three initializations, not three checkpoints of one model.

Training budget for the screen: 2 passes over the 64 training sequences, in
order, one segment each. No replay buffer. No second budget unless this
protocol is amended because a technical bug, not a disappointing score, made
the run invalid.

## Arms

1. `fast-frozen`: head reads fast features, hidden weights frozen.
2. `fast-correct`: proposal-30 `e_f` at feature decay 0.25.
3. `slow-frozen`: head reads `s`, hidden weights frozen. This is the readout-only control.
4. `slow-correct`: candidate, `signal * e_s`.
5. `slow-shuffled`: same eligibility, neuron-shuffled signal.
6. `slow-zero`: eligibility present, signal set to zero.

## Gates

Information gate, ridge probe, development labels, no synapse update:

- If the fast-feature probe is at least 0.65, stop. Premise false.
- If the slow-trace probe is below 0.40, stop. Slow trace has no usable information.
- Otherwise the learning screen is interpretable. The probe itself is not a skill score.

Learning screen, actual head accuracy on development, all three seeds:

- Open confirmation only if every seed of `slow-correct` is at least 0.65, the
  mean of `slow-correct - slow-frozen` is positive, and the mean of
  `slow-correct - fast-correct` is at least 0.05.
- Otherwise NO-GO for this hypothesis on this task. Do not open the eight
  confirmation seeds. Do not change the threshold after seeing the number.

Confirmation, only after that screen, eight fresh seeds, same frozen splits:

- Candidate mean at least 0.65.
- The 95% interval across the eight seed-level gains versus `slow-frozen` has
  a lower bound of at least 0.05.
- The same lower bound versus `fast-correct` is at least 0.05.
- `slow-shuffled` and `slow-zero` are negative controls, not extra seeds of the
  candidate.

This is a one-task acquisition screen. It does not measure a continual stream,
so the master forgetting bounds are not claimed. It does not measure chat or
vision.

## Technical checks before the screen

- Fast arm uses the existing `ReadoutEligibility` step.
- Predict and frozen learning do not change hidden weights.
- A failed update restores weights, slow trace, eligibility, and the signal RNG.
- Checkpoint load rejects a foreign format and restores a cold object whose next
  prediction matches the saved object.
- Peak process RSS and committed memory are recorded for the screen process.
  There is no child worker. The 2 GB budget applies to that process.

## Claim scope

The only positive claim this protocol can support is: on this delayed two-event
order task, with Generation-1 feature decay, the slow-trace eligibility teaches
the hidden synapses something the fast `e_f` path and a frozen hidden do not.
Anything broader is out of scope even if the confirmation passes.
