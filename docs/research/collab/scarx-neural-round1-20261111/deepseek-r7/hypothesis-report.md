# R7 HYPOTHESIS REPORT — Round 1: locating the hidden-learning bottleneck

**Branch:** `research/deepseek/r7-causal-hidden`
**Base:** `495dab2f4bac0180e4d1cd28674391f055ccf09d` (main)
**Environment:** DSH Web, Windows 11 · `deepseek-ai/deepseek-v4.1-flash-1017`

All numbers below come from executed probes (`scripts/r7_probe0..3`), which only
instrument the production rule. **`src/scarx/neural/` was not modified.**

---

## 1. The production rule, read from source

`src/scarx/neural/core.py`, `SpikingCore.learn`:

```
error      = onehot(target) - p
w_out     += readout_lr * error ⊗ features            # readout: proper delta rule
advantage  = clip(p[target] - reward_mean, -1, 1)       # SCALAR
for name in (in, rec):
    delta  = hidden_lr * advantage * eligibility_trace
             - anchor_strength * (weights[rows] - stable[rows])
    weights[rows] += clip(delta, -update_clip, update_clip)
```

and `SpikingCore.step`:

```
elig_in  *= trace_decay
elig_in  += spikes[rows,None] * pre_trace[None,:]  - .5 * post_trace[rows,None] * x[None,:]
elig_rec *= trace_decay
elig_rec += spikes[rows,None] * old[prev,indices]  - .5 * post_trace[rows,None] * old[indices]
```

Config: `hidden_lr = 0.002`, `trace_decay = 0.9`, `update_clip = 0.025`,
`anchor_strength = 1e-4`, `plastic_rows = 12`, `hidden = 48`.

## 2. Three measured causes, in order of severity

### Cause 1 — **the eligibility trace is statistically indistinguishable from noise** (dominant)

| measurement | value |
|---|---|
| cosine(trace, dL/dw) with a **trained** readout, soft threshold | **−0.0303**, CI95 **[−0.0667, +0.0060]** |
| cosine(random vector, dL/dw), same setup | −0.0159 |
| episodes with \|cos\| > 0.2 | 10% (chance-level) |

**The confidence interval contains zero**, so the trace carries no detectable
direction toward the loss. The hidden update is therefore
`hidden_lr · advantage · (a vector uncorrelated with what would reduce the loss)`:
scalar-modulated noise. This is the primary bottleneck.

*Self-correction:* my script's printed verdict said "the trace carries real directional
information" because 0.0303 > 1.5 × 0.0159. **That was wrong** — comparing two point
estimates when both CIs span zero is meaningless. Corrected above.

### Cause 2 — **the hard threshold zeroes the gradient on most episodes**

| condition | mean \|dL\| | zero-gradient episodes |
|---|---:|---:|
| hard threshold | 5.39e-04 | **25/30** |
| soft threshold (β=8) | 8.22e-04 | **0/30** |

With hard spikes, perturbing a plastic synapse changes the loss in **5 of 30** episodes
by any amount at all. The other 25 are structurally insensitive: no weight on that path
can change the output.

**But softening the threshold is not a free win.** Ridge on soft features drops to
**0.5156** from hard **0.6562** — the hard threshold carries *more* information while
admitting *less* gradient. That trade-off is the real design constraint, and it is why
"just use a surrogate gradient" is not a solution here.

### Cause 3 — **the anchor spring is permanent**

`stable_in` / `stable_rec` are **never refreshed** (measured: unchanged after 40
updates). Every hidden update is opposed by a pull back to the *initial random* weights.
With `anchor_strength = 1e-4` and `hidden_lr = 0.002` this is small per step but it has
no rest state other than the initialisation, so it caps cumulative drift.

Also measured: the scalar `advantage` has mean +0.146 and correlates −0.80 with loss —
it *is* informative about *whether* the trial was good. It carries no information about
*which* of the 96 plastic synapses to blame. So the reward scalar is not the defect; the
trace it multiplies is.

### Sparsity context

41 spikes over 12 ticks × 48 neurons (mean rate 0.071); 28/48 neurons ever fire; 9/12
plastic rows fire. The plastic rows are not dead, so Cause 1 is not "no activity".

## 3. What this rules out

| candidate cause | verdict |
|---|---|
| hidden holds no information | **rejected** — frozen-feature ridge 0.6562, trained readout 0.7969 |
| learning signal wrong (reward scalar) | **rejected** — advantage correlates −0.80 with loss |
| eligibility trace loses causal influence | **CONFIRMED** — cosine ≈ 0 |
| synapses learn then get overwritten | partially — anchor spring to init, plus noise walk |
| representation unstable across tasks | not yet tested; deferred to continual round |
| benchmark ceiling (frozen solves it) | **partially** — frozen readout already 0.7969 on sim_xor |

**The frozen readout already reaches 0.7969 on this task.** Any hidden improvement is
measured against that, and the headroom is 20pp, not 80pp. Q7 of the brief applies
directly and will be respected in the design.

## 4. Two competing mechanisms

### Mechanism A — **Selective Eligibility with Credit Shaping (SECS)**

*Principle.* Keep the production trace machinery but (i) gate what enters the trace on
**prediction relevance**, and (ii) modulate **per-neuron** rather than with a global
scalar.

```
per-neuron credit      c_i = W_out[:,i] · error                (locally computable)
relevance gate         g_i = |c_i| / (mean_j|c_j| + eps)
elig_in[rows]         *= trace_decay
elig_in[rows]         += g_i * (spikes_i * pre - .5 * post_i * x)   scaled by c_i sign
update                 Δw = hidden_lr * sign(c_i) * elig_i * |advantage|
anchor                 refresh stable every K updates to the current weights
```

*Causal hypothesis.* The production update broadcasts one scalar; SECS routes the
readout error back through `W_out` **as a per-neuron coefficient**, which is the
proposal-30 relaxation already approved for research. Predicts a cosine between the
shaped trace and `dL/dw` materially above the noise floor.

*Falsifier.* If `cosine(SECS trace, dL/dw)` does not exceed the random null's CI, the
mechanism fails at its own stated purpose and is rejected regardless of accuracy.

*Cost.* `c_i` is one matvec of size 2×48 — negligible. No new state per synapse beyond
the existing trace. Streaming, no future ticks.

### Mechanism B — **Contrastive Trace Selection (CTS)** — independent of A and of Gemini

*Principle.* Do not trust a trace at all. Instead, keep a **short ring buffer of the
last K eligibility traces** and update only the trace whose sign agrees with the
*change* in readout error since it was recorded. This is a local, causal
temporal-difference selection over traces, not over rewards.

```
record trace_k at step t, remember error_e_t
at step t+Δ:   δ = |e_t| - |e_{t-Δ}|
if δ < 0 and sign(e_t·W_out_col) == sign(trace_k): strengthen that trace's contribution
else: decay it faster
```

*Causal hypothesis.* The failure is that *all* traces are accumulated equally. Selecting
the minority of traces that provably reduced the error should raise alignment without
needing any feedback matrix.

*Falsifier.* If CTS's alignment does not beat the random null, or if its accuracy gain
vanishes under shuffled-error control (feeding random δ), it is rejected.

*Cost.* K×96 floats. Streaming. No backward pass.

## 5. Choice for the first DEV round

**Mechanism A (SECS)** is chosen, for one reason: it makes a **directly measurable
prediction about the quantity I just showed to be broken** (trace–gradient cosine). B's
prediction is weaker (it selects among traces by a statistic that itself may be noise).
A also has a clean negative control — shuffled `c_i` — which B lacks in the same form.

## 6. Required controls (brief §VIII)

| control | implementation |
|---|---|
| initial hidden | weights at init, no learning |
| frozen hidden | production `mode="frozen"` |
| learned hidden | SECS |
| zero signal | `advantage` forced to 0 |
| shuffled signal | per-neuron credit `c_i` permuted across neurons |
| same-readout evaluator | one head, one scaler, one protocol for all arms |
| common refit | refit head on each arm's features with identical budget |
| weight intervention | restore learned weights → re-measure; and swap a learned row block |
| held-out | separate data seed offset |
| protected regression | sim_xor frozen baseline must not drop >3pp |

## 7. Gates for this DEV round

Not formal confirmation. DEV_PROMISING requires **all** of:

1. SECS `both_A` or accuracy > frozen by ≥ **+5pp** on held-out, mean of 3 dev seeds.
2. Paired CI95% over seeds excludes 0.
3. Shuffled-credit control does **not** reproduce the gain (gap ≥ 3pp).
4. Zero-signal control does not reproduce the gain.
5. Weight-restore intervention removes the gain (proves the weights carry it).
6. Cosine(trace, dL/dw) with trained readout **> upper bound of the random null CI**.
7. No protected-task regression > 3pp.

Failing (6) rejects the mechanism **even if accuracy improves** — that is the
anti-false-progress commitment.

## 8. Declared false-progress risks for this round

- **FP2** (ridge better but hidden unchanged): explicitly guarded — we require a
  weight-intervention test.
- **FP4** (one benchmark at 100%, related task at chance): guarded by the protected
  sim_xor arm.
- **FP7** (optimizer change attributed to architecture): SECS changes *what enters the
  trace and how it is modulated*, not the optimizer step size. `hidden_lr`, `clip`, and
  `trace_decay` stay at production values in the primary arm.
