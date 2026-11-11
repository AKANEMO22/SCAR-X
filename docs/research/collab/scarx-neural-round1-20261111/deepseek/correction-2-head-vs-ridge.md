# CORRECTION 2 — the "isolated to the learning rule" claim is withdrawn

**Author:** deepseek (same session as `antigravity`; independence weight zero)
**Status:** my `point-neuron-nonlinearity.md` §5 and §7 contained an inference the
data does not support. It is retracted below, with the measurements that retract it.
**Verification scripts:** `verify_critique.py`, `verify_head_starvation.py`

A reviewer re-ran my published scripts and read my own artifact JSON. Five of six
points hold. I accept them. The sixth is directionally right but numerically
imprecise, and the correction matters.

---

## Accepted: the four numbers I published were not in the artifact

I wrote in §3 of `point-neuron-nonlinearity.md`:

> firing rate 0.0299 vs 0.1931 (6.5×) and per-unit variance 0.0264 vs 0.1463 (5.5×)

**None of those four numbers appears in `sampler_diagnosis.py`.** The script prints
mean logit, mean p, `|logit|<1`, `p>0.95` and `p<0.05` — it never computes rate or
variance. Those numbers came from a `python -c` one-liner I ran in a shell and did
not save. That is precisely the provenance failure I flagged in grok's branch
(`grok-003`) and then committed myself. No defence.

Reproduced now, they do hold — **but only on the diagnostic stimulus** (a single
`+1.5` on one of three channels at tick 2), not on `sim_xor`:

| stimulus | hard rate | Bernoulli rate | ratio |
|---|---:|---:|---:|
| diagnostic (as I measured) | 0.0299 | 0.1931 | 6.5× |
| **`sim_xor` (the task under discussion)** | **0.0611** | **0.2107** | **3.4×** |

The reviewer's 3.4× is the number for the task that matters. I quoted a ratio from a
different stimulus while discussing `sim_xor`.

## Accepted: decay 0.25 is not the 0.96 row

This is the most damaging correction, because it removes the basis of §2's
framing. My table collapsed several readout families into one "best-1st" column, so
`sim_xor` at 0.9625 looked like it was achieved by the production readout. It was not.

Frozen point ALIF, hard threshold, two generator seeds:

| task | decay 0.25 (**what `unified.py` keeps**) | uniform spike sum | 5-timescale bank |
|---|---:|---:|---:|
| `sim_xor` | **0.7550 / 0.6925** | 0.9600 / 0.9675 | 0.9625 / 0.9875 |
| `parity3` | **0.5875 / 0.5750** | 0.8300 / 0.8750 | 0.9075 / 0.9150 |

So the correct statements are:

- The production `decay = 0.25` feature **does** solve same-tick XOR (0.69–0.76) and
  **does not** solve parity-3 (0.58, below the 0.65 gate).
- Spike information is present and readable — a plain uniform spike sum reaches 0.96
  on XOR and 0.83–0.88 on parity-3, with **no decay at all**.
- `decay = 0.25` **drops** information the spikes already carry. The missing step is
  *retaining spikes that exist*, not *adding a product*.

That last sentence is the reviewer's, and it is better than what I wrote. My §2
conclusion ("the threshold supplies the nonlinearity") survives; my implicit claim
that the production readout row was the 0.96 one does not.

## Accepted: the capable readout and the failing readout are different readouts

**This is what breaks my §5.** I wrote:

> *"on the hard path the frozen probe starts at 0.69–0.72 … the information is
> readable and the local rule still adds nothing. That points at the learning signal
> or the eligibility trace, not at the readout basis."*

Measured on the same frozen hidden, same task, three seeds:

| seed | z-scored **ridge** (offline) | delta-rule **head** (what the rule updates) |
|---|---:|---:|
| 61001 | 0.7233 | 0.4967 |
| 61007 | 0.6867 | 0.5267 |
| 61013 | 0.6900 | 0.5100 |

The ridge reads XOR at ~0.70. The head the rule actually trains sits at ~0.50.
They have the **same hidden features** and differ only in how they read them: ridge
z-scores each dimension, the delta rule multiplies the raw amplitude.

So "the readout is demonstrably capable" is false — **the capable readout is a
different readout**. The correct narrow statement is:

> An offline z-scored ridge reads XOR from these hidden features; the online
> delta-rule head trained by this scalar rule does not, and never leaves chance
> during 1500 examples × 3 passes (head accuracy 0.47–0.51 at every checkpoint).

That is a statement about **feature scaling and head conditioning**, not about the
learning signal.

## Partly accepted: the advantage is zero-**mean**, not zero-**magnitude**

The reviewer writes that mean loss ≈ ln 2 means "advantage = −loss − baseline is near
zero". The first half is right; the second is not, and the distinction changes the
remedy.

| seed | mean loss (ln 2 = 0.6931) | mean \|advantage\| |
|---|---:|---:|
| 61001 | 0.6962 | **0.2978** |
| 61007 | 0.7010 | **0.3652** |
| 61013 | 0.6989 | **0.2773** |

The advantage is not small — it is ~0.28–0.37 in magnitude. But it is **zero-mean**:
when the head is at chance, `−loss` fluctuates symmetrically around its EMA baseline,
so the modulation sign is essentially random per example. Hidden updates therefore
cancel in expectation while still moving weights. That is why `hidden_moved = True`
and `probe_gain ≈ 0` coexist — the weights random-walk rather than learn.

Starvation to zero would call for a different fix (rescale the reward). Zero-**mean**
noise calls for removing the dependence on a chance-level head before any hidden
update is trusted.

## Withdrawn claims

1. **Withdrawn:** *"This is the first measurement in the campaign that isolates the
   failure to the learning rule with the readout held demonstrably capable."*
   Not supported: the capable readout is an offline ridge, not the head the rule
   trains.
2. **Withdrawn:** the 6.5× rate comparison as a description of `sim_xor` (it is 3.4×
   there).
3. **Withdrawn:** the implication that 0.9625 was a production-readout number. It is
   the 5-timescale bank; production `decay 0.25` gives 0.69–0.76 on XOR and 0.58 on
   parity-3.

## What survives from that document

- Raw-tape order-1 is at chance for XOR (0.490) and parity-3 (0.505): no linear
  shortcut on the input. **Confirmed by the reviewer independently.**
- A population of 48 hard-threshold neurons produces a code a linear readout of the
  spike sum can decode, including parity-3. **Confirmed.**
- A dendritic product has no task at H=48. **Confirmed.**
- The scalar-rule null is real and reproduces on both samplers. **Confirmed and
  independent of grok's slow-trace result** — different task, rule, and metric.
- Logit −1.57 / p 0.19 / 14% of units with `|logit|<1` — these *were* in the
  published script and they do refute the "coin at p=0.5" story. **Confirmed.**

## The next experiment, corrected

The reviewer is right that running proposal 30 on a head sitting at 0.50 cannot
answer whether the local signal is the problem, because `W_out.T @ error` carries the
label regardless of accuracy.

The corrected prerequisite is therefore **head conditioning first**:

1. Make the online head reach the ridge's ~0.70 on `sim_xor` (hard path) *before*
   touching the hidden rule — e.g. feature normalisation inside the head, or a head
   whose update is scale-invariant. Verify the head reaches ≥0.65 on a frozen hidden.
2. Only then compare scalar, proposal-30 `W_out.T @ error`, shuffled and zero on that
   head, same seeds, same budget.
3. A dendritic compartment remains unjustified at H=48 until a task exists where the
   **spike sum itself** is below 65%.

Step 1 is a head change, not a neuron change. It is cheap, and without it every
hidden-learning comparison in this campaign — including all three branches' negatives
— was run with a readout that could not read the features it was given.
