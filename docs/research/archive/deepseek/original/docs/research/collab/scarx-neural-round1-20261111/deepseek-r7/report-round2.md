# R7 ROUND 2 — REPORT: my own testbed was invalid; the bottleneck relocates again

**Branch:** `research/deepseek/r7-causal-hidden`
**Protocol:** `protocol-round2.md` (committed before these runs)
**Production `src/scarx/neural/` modified:** no

**Decision: REJECT both mechanisms (SECS, CTS) — and REJECT my own round-2 testbed.**

---

## 1. What round 2 set out to do

Round 1 found no headroom on the inherited suite and located a testbed that appeared to
have it: `delayed_xor` through a spike-sum channel, spike-sum linear 0.5404 vs quadratic
0.9948 = **+0.4544 headroom**. The protocol registered the prediction that a working
credit mechanism should climb from 0.54 toward 0.99.

## 2. Result: every arm was at chance, including the mechanisms

3 dev seeds, same population, same data, same spike-sum channel, same common refit:

| arm | refit train | refit test | vs frozen | \|Δw\| |
|---|---:|---:|---:|---:|
| frozen | 0.5547 | **0.5430** | — | 0.00000 |
| production | 0.5566 | 0.5456 | +0.26pp | 0.00112 |
| secs | 0.5638 | 0.5417 | −0.13pp | 0.00128 |
| cts | 0.5605 | 0.5430 | +0.00pp | 0.00072 |
| shuffled | 0.5579 | 0.5378 | −0.52pp | 0.00113 |
| zero | 0.5547 | 0.5430 | +0.00pp | 0.00000 |

Ceiling (quadratic) = 0.9948. **No arm moved.** Not SECS, not CTS, not the production
rule. Weight deltas were ~0.001 against the 45pp of headroom the testbed was chosen for.

By the registered falsifier this was the strong negative: on a task with proven headroom,
the local signals cannot produce a useful representation.

## 3. Then I invalidated my own testbed

I did not stop at "the mechanisms failed". I probed why, and the testbed collapsed.

### 3a. The oracle search was too weak to settle E1 vs E2

A random-direction hill climber over the plastic rows lifted test accuracy by only
**+1.6 to +7.4pp** in 240 steps. **I do not use this as evidence that 45pp is
unreachable** — it is a weak search, and reporting it as a negative would overstate what
it shows. It is recorded as a **failed instrument**.

### 3b. The memory horizon is 3 ticks; the task needs 7

| tick | Δ | relative response |
|---:|---:|---:|
| 1 | 0 | 1.0000 |
| 2 | 1 | 0.3986 |
| 3 | 2 | 0.2228 |
| 4 | 3 | 0.1454 |
| 5 | 4 | **0.0945** |
| 8 | 7 | **0.0272** |

A single pulse is below 10% of its initial amplitude after 3 ticks at `leak = 0.65`.
`delayed_xor` needs it at 7. (Diagnostic only: `leak = 0.80` reaches 7 ticks, but the
production leak is not something I change here.)

### 3c. But the value *does* survive — in a channel the readout cannot see

At tick 8, decoding the tick-1 bit:

| seed | from spikes | from voltage |
|---|---:|---:|
| 140011 | 0.5049 | **0.9668** |
| 140017 | 0.5127 | **0.9707** |
| 140023 | 0.4980 | **0.9795** |

**The neuron's analog memory holds the bit; its spikes do not.** The production readout
sees only `features = decay·features + (1−decay)·spikes`. It is blind to the channel that
carries the answer.

### 3d. Yet reading voltage does not solve the task either

3 seeds, no hidden learning, linear readout on each candidate channel:

| channel | dims | test | vs production |
|---|---:|---:|---:|
| production (decay) | 48 | 0.5195 | — |
| spike sum | 48 | 0.4831 | −0.037 |
| **voltage** | 48 | **0.5202** | +0.001 |
| adaptation | 48 | 0.5000 | −0.020 |
| sum + volt + adapt | 144 | 0.4974 | −0.022 |

Handing the readout the analog memory does **not** help. So "the readout is blind" is
true but insufficient as an explanation.

### 3e. The per-tick scan explains everything

| tick | spikes | voltage | both | **bit a** | **bit b** |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4961 | 0.4961 | 0.5020 | **1.0000** | 0.4590 |
| 4 | 0.5234 | 0.5215 | 0.5137 | **1.0000** | 0.4766 |
| 8 | 0.4766 | 0.5449 | 0.5527 | 0.9512 | **1.0000** |
| 9 | 0.5137 | 0.5430 | 0.4844 | 0.6172 | **1.0000** |
| 11 | 0.5000 | 0.5234 | 0.5039 | 0.7656 | **0.9961** |

**Both bits are individually recoverable at ~1.00. Neither is ever jointly linearly
readable: the best value of `a XOR b` at any tick is 0.5527.**

And the control that closes the case (round 2f): the **same** task with both bits at
**one tick** is linearly readable at **0.9792** (quadratic 0.9941, gap only +0.015).

## 4. The corrected diagnosis

`XOR` is not linearly separable in `(a, b)`. The point neuron resolves it *only when both
bits arrive on the same tick*, because the spike nonlinearity then forms the product
`a·b` inside one membrane integration. When the bits are separated by 7 ticks, the
population never holds both in a form that a single linear readout can combine: `a` lives
in the voltage, `b` arrives later and perturbs it, but the resulting state is not a linear
function of the conjunction.

**Therefore: on `delayed_xor`, no hidden-weight change that leaves the readout linear can
solve the task, and the "45pp headroom" was headroom for a *quadratic reader*, not for a
representational improvement.** My round-2 testbed was invalid for the question it was
built to answer. That is my error, disclosed here.

**However**, the round-1 finding survives intact and is now sharper:

- On `sim_xor` and `parity3` the production **channel** discards 13–21pp that the same
  frozen neurons produced (probe 5). Those are same-tick conjunctions, where the spike
  nonlinearity already supplies the product — so a channel that keeps more spike detail
  *does* close a real gap.
- The trace is directionless (probe 3, cosine −0.030, CI containing 0).
- The hard threshold zeroes the gradient on 83% of episodes.

## 5. Where this points next (registered before running)

The bottleneck is **not** one thing, and I have been treating it as one. The evidence now
separates:

| layer | evidence | is it broken? |
|---|---|---|
| input dynamics | horizon 3 ticks at leak .65 | **yes** for delays > 3 |
| spike encoding | carries `a` and `b` but never their conjunction across time | **yes** for cross-tick XOR |
| readout channel | discards 13–21pp of same-tick conjunction detail | **yes** |
| credit trace | cosine ≈ 0 vs gradient | **yes** |
| reward scalar | correlates −0.80 with loss | no |
| hidden information | frozen linear 0.83–0.97 on same-tick tasks | no |

**Round 3 will target the one layer where a local rule can provably matter**: a
same-tick task on a channel that preserves conjunction detail, where frozen-linear is
measurably below the trajectory ceiling (sim_xor / parity3, gap 0.129 / 0.207). That is
the only configuration where "hidden plasticity improved the representation" is a
falsifiable claim rather than a mis-specified one.

I am **not** proposing to raise `leak` or add neurons: both are forbidden, and the
horizon finding is a diagnosis, not a licence.

## 6. Declared false-progress exposures

| risk | how it was handled |
|---|---|
| **FP2** (ridge better, hidden unchanged) | `weight_delta` per arm; frozen is 0.0 |
| **FP4** (one benchmark solved, related at chance) | `order` and `parity3` measured separately; `order` shows the *opposite* channel effect |
| **FP7** (optimizer change called architecture) | `hidden_lr`, `trace_decay`, `update_clip`, `leak` unchanged in every arm |
| **new: invalid testbed** | disclosed as the round's main result, not hidden |
| **new: weak instrument reported as negative** | oracle hill-climber explicitly **not** used to settle E1/E2 |

## 7. Self-corrections in this round

1. **Round 2's premise was wrong.** I selected `delayed_xor` because its *quadratic*
   refit was 0.9948 and called that "headroom for hidden learning". A quadratic refit's
   advantage is unavailable to a linear readout, so the figure never measured what I
   claimed. Caught by round 2f's simultaneous-XOR control.
2. **I nearly reported the oracle as evidence.** +1.6 to +7.4pp from a random hill
   climber would have been written up as "even an oracle cannot" — it cannot support that
   claim. Recorded as a failed instrument.
3. **A test-script bug** (`refit_head` with `lr=.5`, 300 epochs, reaching 0.627 *train*
   accuracy) made every arm report ~0.49 in the first run. Caught by comparing against
   `ridge_probe` on the same features and fixed by z-scaling and reporting train accuracy
   alongside test.

## 8. Message for Gemini

The question that matters before either mechanism is believed:

**On your task, what is `frozen + common refit` and what is `quadratic refit` on the
same frozen features?**

If quadratic ≫ linear, your task is like my `delayed_xor`: it has "headroom" only for a
nonlinear reader, and a linear online head can never capture it no matter how good the
credit signal is. A gain you measure there may be your head becoming accidentally
nonlinear, not your hidden representation improving.

If quadratic ≈ linear and frozen is low, you have a genuine testbed and I would like the
generator definition and the exact frozen-refit number, because every task in the
inherited SCAR-X suite fails that test (headroom −0.042 / +0.007 / −0.044).

## 9. Artifacts

`scripts/r7_r2_mechanisms.py` · `r7_r2b_oracle.py` · `r7_r2c_horizon.py` ·
`r7_r2d_resolve.py` · `r7_r2e_channel_fix.py` · `r7_r2f_nonlinearity.py` ·
`r7_r2g_pertick.py` · `runs/20261111T114943Z-r7-r2-delayed-xor-v1-b86a42/result.json`
