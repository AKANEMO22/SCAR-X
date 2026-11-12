# R7 ROUND 8 — REPORT: the trace is directionless for its own reasons, not because of the spike

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: REJECT the "spike starves the gradient" explanation. The eligibility trace is
directionless independently of the readout channel.**

---

## 1. The hypothesis I was testing, and it was mine

Rounds 1–7 left two findings that looked like they had a common cause:

- **round 1**: `cosine(eligibility trace, dL/dw) = −0.0303`, CI [−0.067, +0.006] — at
  noise level;
- **round 1**: the hard threshold gives `dL/dw = 0` on 25/30 episodes, because a spike is
  piecewise constant in the presynaptic weight.

I inferred that the trace looked directionless **because** the readout consumed a
quantised signal with zero derivative, and that a continuous readout would reveal a usable
gradient. That is a clean, falsifiable claim, and it is wrong.

## 2. The measurement

3 dev seeds → 1 seed here (seed 140011), parity3, trained linear head on each channel.

### Gradient availability

| channel | dims | frozen linear acc | mean \|dL/dw\| | zero-gradient episodes |
|---|---:|---:|---:|---:|
| spikes | 48 | **0.7812** | 2.00e-02 | **18/20** |
| **voltage** | 48 | 0.5391 | **7.18e-02** | **0/20** |
| adaptation | 48 | 0.7695 | 1.87e-02 | 18/20 |
| spikes + voltage | 96 | **0.8594** | 1.69e-02 | 0/20 |

**My prediction about the gradient was correct.** Voltage gives a **3.6× larger**
gradient and **zero** dead episodes versus 18/20 for spikes. The spike really does
quantise away the derivative.

### Trace alignment — where the hypothesis dies

| channel | cosine(trace, dL/dw) | random-vector null |
|---|---:|---:|
| spikes | **+0.0587** | +0.0668 |
| voltage | **+0.0463** | +0.0278 |

**With a fully continuous, 3.6× stronger, never-zero gradient, the trace is still
indistinguishable from a random vector.** +0.0463 versus a null of +0.0278, on 15
episodes, is not a directional signal.

The trace's failure is **not** downstream of the spike. It fails on its own terms.

## 3. So what is wrong with the trace itself?

The trace in `core.py` is:

```
elig_in *= trace_decay
elig_in += spikes[r]·pre − .5·post[r]·x
```

Three defects, now unambiguous because the gradient is known to be available:

1. **It is a Hebbian/anti-Hebbian correlation, not a derivative.** `spike · pre` measures
   co-activation, which is only proportional to `∂spike/∂w` when the neuron sits exactly
   at threshold. For a neuron far above threshold the true derivative is 0 but the
   correlation is large; far below, both are 0 but the trace still accumulates.
2. **The `−.5·post·x` term is a fixed anti-Hebbian offset**, not a normalisation. It
   subtracts a constant fraction of the postsynaptic rate regardless of whether the
   neuron contributed.
3. **Decay 0.9 over 12 ticks** means the trace at update time is dominated by ticks 10–11
   (the readout gate), not by the ticks where the task-relevant input arrived (1–3).

Any of these alone would decorrelate the trace from the gradient. Fixing the readout
cannot help because the readout was never the problem for the trace — it was the problem
for the *loss*, which is a different quantity.

## 4. What this round therefore settles

| claim | status |
|---|---|
| hard threshold zeroes dL/dw | **CONFIRMED** (18/20 episodes) |
| a continuous channel fixes that | **CONFIRMED** (voltage: 0/20, 3.6× magnitude) |
| the directionless trace is caused by the spike | **REFUTED** (cosine 0.0463 vs null 0.0278) |
| the credit rule is the primary bottleneck | **REFUTED** — it is one layer down, and it fails even with a good gradient |

And a separate result worth keeping:

**spikes+voltage reads parity3 at 0.8594 versus 0.7812 for spikes alone** (+7.8pp) with
the hidden frozen. That is a codec gain, consistent with round 1's finding that the
production decay channel discards 13–21pp. It is **not** learning and is not reported as
such.

## 5. Revised, complete map of the bottleneck

| layer | measurement | broken | fixable within constraints |
|---|---|---|---|
| memory horizon | 3 ticks; needs 7 for cross-tick tasks | yes | **no** — `leak` is frozen |
| cross-tick conjunction | best linear XOR 0.5527 at any tick | yes | **no** — needs nonlinearity |
| readout channel | decay loses 13–21pp vs spike sum | yes | **yes** — a codec change |
| readout capacity (retention) | joint linear head 0.8620 → capacity is fine | **no** | — |
| sequential interference | 31.84pp forgetting | yes | **not solved** by 3 mechanisms |
| **eligibility trace** | cosine ≈ 0 **even with a continuous gradient** | **yes** | unknown — needs a trace that is a derivative |
| reward scalar | correlates −0.80 with loss | no | — |

The trace defect is now the **clearest actionable target**, precisely because round 8
ruled out the excuse I had been using for it. A trace that fails with a perfect gradient
must be replaced, not re-modulated.

## 6. What I will NOT do

- I will not retune the existing trace's constants; rounds 1 and 8 show the signal is
  absent, not small.
- I will not propose a surrogate gradient: round 8 shows gradient magnitude is not the
  binding issue, so a surrogate would add engineering without addressing the finding.
- I will not claim any of this is a product result. `spikes+voltage` 0.8594 is a codec
  measurement on a synthetic task.

## 7. Declared exposures

| risk | exposure |
|---|---|
| **single seed in this probe** | seed 140011 only; stated, and the effect sizes (3.6× gradient, 18/20 vs 0/20 zeros) are large enough that seed variance is not a plausible explanation for the *gradient* result |
| **cosine measured on 15 episodes** | small; but the comparison to a same-episode random null is what makes it interpretable |
| **my own hypothesis refuted** | stated in the title and §1 |

## 8. Message for Gemini

One experiment would settle whether CKCER's gain is credit or codec:

**Compute `cosine(eligibility, dL/dw)` with the readout trained, once with spikes and once
with voltage.** Mine is +0.059 (spikes) and +0.046 (voltage) against a random null of
+0.067 / +0.028. If yours is also at null, then whatever accuracy improvement you measure
is not coming from directed credit, and the shuffled-signal control should match it.

The related codec number, in case it is cheaper than a new rule: **spikes+voltage reads
parity3 at 0.8594 versus 0.7812 for spikes alone**, hidden frozen. That is +7.8pp for
concatenating a state variable the production readout already computes and discards.

## 9. Artifacts

`scripts/r7_r8_voltage.py` · `report-round1.md` (trace cosine, gradient zeros) ·
`report-round7.md` (mechanism rejections)
