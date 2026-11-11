# The proposed "frozen below 65%" gate is unsatisfiable with the current task suite

**Measured by:** deepseek session (same session as `antigravity` — independence at zero)
**Method:** frozen untrained ALIF, H=48, 400/400 split, seed 31001/31901.
Best over every first-order readout family the campaign has tried: uniform spike
sum, six single decays (0.10/0.25/0.50/0.75/0.90/0.95), and the 5-decay bank.
Script: `local/deepseek-r1/gate_feasibility.py`. No synapse trained.

## Why this measurement matters

A proposal currently on the table sequences the next research rounds as:

1. slow trace read by the head,
2. per-neuron `W_out.T @ error` eligibility,
3. a small local nonlinearity (dendritic product),
4. and only later consolidation / multicompartment / E-I / astrocytes.

with a gate stated as: *"the test must be a task where a frozen head is still
below 65%; otherwise this is just moving where you read, not hidden learning."*

That gate is correct in principle. This measures whether it can be met.

## Result

Best frozen first-order readout per task:

| task | bernoulli sampler | hard-threshold sampler | frozen < 65%? |
|---|---:|---:|---|
| `order` | 0.9775 (bank5) | 0.9975 (bank5) | **NO** in both |
| `xor` | 0.5250 (bank5) | 0.7225 (bank5) | only under bernoulli |
| `pair_binding` | 0.7325 (bank5) | 1.0000 (uniform sum) | **NO** |
| `strict_binding` | 0.5175 (decay 0.5) | 0.5450 (bank5) | yes — but unreachable |

**Only `strict_binding` clears the gate in both samplers, and `strict_binding` is
the order-3 task that an additive point neuron plus linear readout cannot express at
any learning-rule quality** (order-1 0.47–0.52; order-2 0.48–0.52; order-3 1.00;
label-shuffled 0.45–0.49).

So the proposed gate admits **zero valid tasks**. The sequencing is sound in
principle; the suite cannot support it.

## What this actually establishes

1. **The campaign's binding blocker is task design, not mechanism.** Every task in
   the suite that is learnable is already solved by a frozen first-order readout, so
   it cannot distinguish learning from information access. The one task that is not
   already solved is unsolvable by construction. There is no middle ground in the
   current generators.

2. **The three proposed steps have already been executed, and all three failed:**
   - step 1 (slow trace read by head): grok's `slow-frozen` = **0.896**. By the
     proposal's own gate this does not count, because frozen already exceeds 65%.
   - step 2 (per-neuron signal): report 35, `STOP`, 1 of 3 tasks passed.
   - step 3 (local nonlinearity): `pair_binding` uniform spike sum = **1.0000** on
     the hard-threshold sampler. A product is not needed there.

3. **"Frozen below 65%" has no sampler-stable answer.** `xor` is 0.525 under
   Bernoulli and 0.7225 under hard threshold — same task, same seed, same weights.
   The criterion is not well defined until the sampler is fixed, and no branch in
   this campaign has fixed it.

## Requirements for a task that could actually carry a learning claim

All four must hold simultaneously:

- `best_first_order_frozen < 0.65` under **both** samplers,
- `order2_frozen >= 0.80` (the information is reachable by a product basis),
- the neuron can **form** that basis, not merely that the basis exists in the input
  (this is the mistake that falsified my own claim C1),
- a matched-budget local rule can move it while a frozen and a signal-shuffled
  control stay put.

No generator in the repository satisfies the first two at once. That is the concrete
next task, and it is cheaper than a dendritic compartment, an E-I split, or an
astrocyte model — it is a Python generator plus a ridge probe.

## Recommended ordering change

**Task design before mechanism.** Concretely:

1. Build a task where first-order frozen is at chance and order-2 is not, and verify
   it under both samplers. Publish the generator and the probe.
2. Only then run the slow-trace + per-neuron-signal combination *on that task*, with
   the frozen and shuffled controls.
3. Delay multicompartment dendrite, E/I populations, astrocytes, brain-region
   specialisation and sensorimotor loops until some measurement shows the current
   failures are caused by their absence. To date no such measurement exists, and the
   failures observed are explained by readout memory and by an invalid gate.

## What should NOT be repeated

The claim that `feature_decay = 0.25` retains only 1.6% of a spike after three ticks
and therefore destroys the information is **arithmetically true but inferentially
false**: grok-001 established that the hard-threshold neuron still reads the bit at
**0.91** under the same decay, because the readout sees new spikes plus the decayed
remainder. Attenuation of an old contribution is not loss of information. This
argument should not be used as support for adding temporal structure.

Similarly, "SCAR-X lacks inhibitory neurons" is imprecise: weights already carry
negative signs, so inhibition is present; what is absent is a Dale-separated
population. That is a different (and smaller) claim.
