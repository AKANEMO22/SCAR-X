# A point ALIF neuron already computes XOR — and parity-3. The dendritic proposal has no target at H=48.

**Session:** deepseek (same session as `antigravity`; independence weight zero)
**Scripts:** `local/deepseek-r1/task_search.py`, `point_neuron_nonlinearity.py`,
`sampler_diagnosis.py`
**Method:** frozen untrained ALIF (H=48), 400/400 split, seeds 31001/31009, no synapse
trained.

---

## 1. What this was supposed to be

grok's conclusion, and my own gate-feasibility analysis, both ended the same way:

> *"The next step is only worth doing if there is a task where every first-order
> readout is below 65% and a local product solves it."*

So I went looking for that task. I wrote three generators whose raw-tape order-1
probe sits at chance (0.48–0.50), making them linearly unlearnable by construction:

| generator | raw-tape order-1 |
|---|---:|
| `sim_xor` — two bits on one tick, label = XOR | 0.4900 |
| `seq_xor` — same XOR, bits on different ticks | 0.4825 |
| `parity3` — three bits on one tick, label = XOR of three | 0.5050 |

These are genuinely order-1-blind. A linear readout cannot do them at any width.

## 2. What happened

**The frozen point neuron solved them anyway.**

| task | Bernoulli path | **hard-threshold path** |
|---|---:|---:|
| `sim_xor` (order-2) | 0.5200 | **0.9625 / 0.9875** |
| `parity3` (order-3!) | 0.5325 | **0.9075 / 0.9150** |

`parity3` is the striking one. Three-way XOR is not linearly separable and needs an
order-3 term; the raw tape confirms no linear shortcut (0.5050). Yet the frozen
point neuron with a linear readout reaches **0.91**.

**A point neuron with threshold-and-reset is not a linear device.** The threshold is
itself a nonlinearity, and combined with leaky integration across ticks it can form
conjunctions. The premise that motivated multicompartment dendrites — *"an additive
point neuron cannot express order-2, so add a product"* — **is false for the neuron
that SCAR-X actually ships.**

## 3. Why the Bernoulli path fails where the hard path succeeds

Same weights, same inputs, same ticks. Only spike generation differs.

| quantity | hard | Bernoulli |
|---|---:|---:|
| firing rate | 0.0299 | 0.1931 (**6.5×**) |
| per-unit variance across examples | 0.0264 | 0.1463 (**5.5×**) |

**My first explanation was wrong and the data refuted it.** I hypothesised the
neuron becomes "a coin near p=0.5". Measured: mean logit is **−1.57**, mean p is
**0.19**, and only 14% of units have `|logit| < 1`. The Bernoulli neuron is not
balanced at chance; it fires **6.5× too often**.

The corrected mechanism: stochastic sampling raises the firing rate and the
trial-to-trial variance by ~6×, so each spike carries much less information about
*which* input arrived. With only 12 ticks the readout cannot average that noise
down. It is a **density/selectivity loss**, not a coin flip.

## 4. Consequences

### 4.1 The "add a product" proposal has no target

The ordering proposed for future rounds was:

1. slow trace read by the head → **already done** (grok: `slow-frozen` 0.896)
2. per-neuron `W_out.T @ error` → **already done** (report 35 STOP; grok's newest
   result: on the only task below 65%, learned 0.417 vs frozen 0.443)
3. local nonlinearity / dendritic product → **no task needs it**, because the point
   neuron already computes XOR *and* parity-3

Step 3 was to be justified by "a task where every first-order readout is below 65%
and a product solves it". **On the hard-threshold path such a task may not exist at
H=48**, because the neuron's own threshold supplies the nonlinearity.

### 4.2 This resolves the campaign's central confusion

Three branches reported "frozen readout already solves it" and concluded the readout
is the bottleneck. The real statement is narrower and different:

- On the **hard-threshold** neuron (what `unified.py` ships), XOR/parity/binding are
  all already linearly readable because the threshold does the nonlinear work.
- On the **Bernoulli** path (what `PlasticCore.features_many` and all prior research
  used), a 6× firing-rate inflation destroys that readability.

So **the research harness has been measuring a neuron the product does not ship**,
and the "frozen head wins" result is partly an artefact of that mismatch. This is
not a small bookkeeping issue — it changes which task can detect learning at all.

### 4.3 What remains genuinely unsolved

Not nonlinearity, and not readout memory. The open problems are:

- **Which sampler is authoritative.** Until this is fixed, no cross-branch number is
  comparable. `xor` scored 0.525 (Bernoulli) vs 0.7225 (hard) on identical weights.
- **Credit assignment under a correct sampler.** All the negative learning results
  were obtained on Bernoulli. They may not transfer to the shipped neuron, in either
  direction.
- **A task that defeats both samplers.** If the hard neuron solves order-3, the task
  must require something a single threshold genuinely cannot do — e.g. a conjunction
  across *distant* ticks, where leaky decay erases the first operand before the
  second arrives.

## 5. What I am NOT claiming

- Not that SCAR-X "can bind" or "understands". These are frozen-weight linear probes
  on synthetic 12-tick tasks; an oracle probe is not a skill.
- Not that the dendritic idea is biologically wrong. It may still matter for
  capacity, robustness or scale. It has **no measurable target at H=48**, which is a
  narrower and testable statement.
- Not that my sampler diagnosis is final. I already had one mechanism hypothesis
  refuted by measurement in this same session; the rate/variance numbers are
  observations, and the causal story is an inference.

## 6. Suggested next experiment (not run)

Fix the sampler question first: run one task, one seed, one budget, on both paths,
with the **same generator** and report both. If the hard path's advantage is stable,
re-run the two decisive negative results — report 35's `strict_binding` and grok's
slow-head-signal — on the hard path. If the negatives reproduce there too, the
learning failure is real; if they do not, the campaign's conclusions were sampled
from the wrong neuron.
---

## 7. The decisive follow-up: does the learning negative survive on the shipped neuron?

Section 4.3 left this open. It is now measured.

Same generator (`sim_xor`), same 3 fresh seeds, same 1500 examples x 3 update passes,
same local scalar-score rule. Only the spike sampler differs. Head refit from scratch
on the learned hidden, paired by seed.

| sampler | learn - frozen (probe gain) | hidden moved |
|---|---:|---|
| Bernoulli | **-0.56pp [-3.00, +4.33]** | yes |
| **hard-threshold** | **-1.78pp [-7.67, +3.00]** | yes |

Per-seed paired differences - Bernoulli: -3.0, -3.0, +4.3. Hard: -7.7, -0.7, +3.0.
Signs flip across seeds on both paths and every CI contains zero.

**Both negatives reproduce.** The hidden weights move in every learning arm
(`hidden_moved = True` throughout) and the refit-head probe does not improve on
either sampler. So the campaign's central learning negative is **not** an artefact of
measuring the Bernoulli neuron.

### What this settles and what it does not

**Settled:** the "frozen head already wins" result is not explained away by sampler
mismatch. Whatever is failing, it fails on the neuron the product ships too.

**Still open, and now sharper:** the hard path starts much higher (frozen probe
0.69-0.72 vs Bernoulli 0.48-0.49), consistent with section 2, and learning still does
not improve on it. So the failure is **not** that the readout cannot see the
information. The information is visible and the local rule still adds nothing. That
points at the learning signal or the eligibility trace - not at the readout basis,
and not at nonlinearity.

This is the first measurement in the campaign that isolates the failure to the
learning rule while the readout is demonstrably capable.

### Caveats on record

- 3 seeds, 1500 examples. Wide CIs; this excludes a *large* gain, not a small one.
- One task (`sim_xor`), H=48, CPU.
- The rule tested is the scalar score rule, **not** proposal 30 (see the deepseek
  report's grok-002 correction). Proposal 30 remains untested on the hard path.
- My mechanism story in section 3 was refuted once already by my own data; the
  rate/variance numbers are observations, the causal reading is an inference.

### The experiment this implies

Proposal 30 - a per-neuron `W_out.T @ error` factor instead of one scalar - has been
tested only on the Bernoulli path (report 35, STOP) and only on tasks where the
frozen head already solved them. The measurement above shows that on the hard path
the information *is* readable and the scalar rule still fails. That is exactly the
condition under which a per-neuron signal could matter, and it has never been run
there. That is the next registered experiment, not a dendritic compartment.
