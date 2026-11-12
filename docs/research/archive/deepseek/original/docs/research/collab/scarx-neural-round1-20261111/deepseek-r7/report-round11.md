# R7 ROUND 11 — REPORT: alignment at +0.945 does not convert into a representation gain

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: REJECT the exact-trace mechanism. The credit-assignment line is closed by
measurement, not by preference.**

---

## 1. What round 10 promised and round 11 delivers

Round 10 ended on the first positive mechanistic result of this session:

| trace | cosine with true `dL/dw` |
|---|---:|
| EXACT (`e_in ← leak·e_in + x`) | **+0.9452** (sd 0.152) |
| random null | −0.0005 (sd 0.098) |

I registered the falsifier in advance: *if `exact` does not beat `frozen` by ≥5pp on the
common refit while `exact_shuffled` does not, the mechanism is REJECTED despite the
+0.945.*

**The mechanism is rejected despite the +0.945.**

## 2. The measurements

### 11b — competent credit router

Round 11a failed for a self-inflicted reason I diagnosed: with the core's **random**
`w_out` the credit magnitude was `|c_i| = 0.0176`, giving `|dw| = 3.5e-05` — 27× below the
production rule, so nothing happened. The fix is to train the readout first and use it as
the router.

| arm | \|credit\| | \|dw\| | V acc | V gain | both acc | both gain |
|---|---:|---:|---:|---:|---:|---:|
| `frozen` | — | 0.000000 | 0.5560 | — | 0.8542 | — |
| `production` | — | 0.000944 | 0.5339 | −2.21pp | 0.8372 | −1.69pp |
| **`exact`** | **3.1069** | **0.055805** | **0.5247** | **−3.12pp** | **0.8125** | **−4.17pp** |
| `exact_shuffled` | 2.8708 | 0.008550 | 0.5638 | +0.78pp | 0.8542 | +0.00pp |
| `exact_zero` | 2.5132 | 0.000000 | 0.5560 | +0.00pp | 0.8542 | +0.00pp |

The router was competent (pre-accuracy 0.5560) and the credit magnitude was now real
(3.11 vs 0.0176) — a 177× increase. The result got **worse**: `exact` is the **worst** arm
on both channels, and its shuffled control *beats* it.

### 11c — eta sweep, to separate magnitude from mechanism

`|dw|` at eta 0.002 was 0.0558, which is large relative to the weight scale (~0.45), so
runaway magnitude was a live alternative explanation. Three etas:

| eta | arm | \|dw\| | V acc | V gain | both acc | both gain |
|---:|---|---:|---:|---:|---:|---:|
| 0.002 | frozen | 0.000000 | 0.5560 | — | 0.8542 | — |
| | exact | 0.055805 | 0.5247 | **−3.12pp** | 0.8125 | **−4.17pp** |
| | shuffled | 0.008550 | 0.5638 | +0.78pp | 0.8542 | +0.00pp |
| 0.0004 | exact | 0.018773 | 0.5430 | **−1.30pp** | 0.8281 | **−2.60pp** |
| | shuffled | 0.001786 | 0.5586 | +0.26pp | **0.8620** | **+0.78pp** |
| 0.00008 | exact | 0.002666 | 0.5430 | **−1.30pp** | 0.8464 | **−0.78pp** |
| | shuffled | 0.000297 | 0.5534 | −0.26pp | 0.8529 | −0.13pp |

**Best exact gain across all etas: −0.78pp. Best shuffled gain: +0.78pp.**

**No eta produces a gain, and the shuffled control is never worse than the mechanism.**
Magnitude is ruled out: reducing eta by 25× reduces `|dw|` by 21× and the gain stays
negative.

## 3. Why alignment failed to convert — the mechanism, not the number

This is the important part, and it is a structural argument now backed by data.

`dL/dw` is the gradient of the **readout loss**. Descending it makes the *current linear
readout on the current features* better. It does not make the features **more informative**
— it makes them more convenient for one particular linear functional. Those are different
objectives, and the data show they diverge:

1. `exact` has the highest alignment (+0.945) and the **worst** refit accuracy. The
   gradient direction is correct for the objective it was derived from, and that objective
   is not the one that produces decodable features.
2. `exact_shuffled` — credit permuted across neurons, i.e. **no directed signal at all** —
   scores **better than the aligned rule at every eta**. If the direction mattered, the
   permutation would destroy the effect. It does not.
3. `exact_zero` reproduces `frozen` exactly on both channels, confirming the harness is
   sound (zero credit → zero update → zero change).

**Round 10's +0.945 was real and it was measuring the wrong thing.** Alignment with
`dL/dw` is necessary for a credit rule to be *correct* and demonstrably not sufficient for
it to be *useful*. I reported round 10 as DEV_PROMISING on the mechanism with the caveat
that no gain was demonstrated; the caveat was the whole story.

## 4. Complete record of what the credit line cost and returned

| mechanism | mechanism type | honest outcome |
|---|---|---|
| SECS (round 1) | per-neuron credit gating a Hebbian trace | rejected; shuffled tied it |
| CTS (round 2) | trace selection by error improvement | rejected; no movement |
| OGH-mean (round 5) | readout protection | rejected; worse than baseline on the honest metric |
| CPS (round 7) | decision-subspace protection | rejected; shuffled control won |
| replay (round 4) | retention | rejected; acquisition collapsed to 0.48 (FP3) |
| **exact trace (rounds 10–11)** | **true derivative of the forward pass** | **rejected despite +0.945 alignment** |

The last one is the strongest possible test of the credit hypothesis, because it removes
every confound: the trace *is* the derivative, no scalar bottleneck, no Hebbian
approximation, a competent router, a 25× learning-rate sweep, and a shuffled control. It
still produces nothing.

**The credit-assignment explanation for SCAR-X's failure is now refuted by direct
measurement.**

## 5. What the evidence says the bottleneck IS

Consolidating eleven rounds, in order of causal precedence:

| layer | measurement | status |
|---|---|---|
| memory horizon | pulse < 10% after 3 ticks at leak .65 | **binding** for delays > 3 |
| cross-tick conjunction | best linear XOR 0.5527 at any tick; both bits individually 1.0000 | **binding** |
| readout channel | decay loses 13–21pp on same-tick conjunctions | **real, fixable** |
| readout capacity (retention) | joint linear head 0.8620 | **not binding** |
| sequential interference | 31.45pp honest forgetting | **binding, unsolved** |
| eligibility trace | +0.945 when exact | **not binding** — proven here |
| reward scalar | correlates −0.80 with loss | **not binding** |

**Credit was never the constraint.** Every round that tried to fix it — six mechanisms
across eleven rounds — failed, and the final one was measurably *correct* and still failed.

The information is lost **upstream**: the membrane leaks the memory away in 3 ticks, the
spike never encodes a cross-tick conjunction, and the readout channel discards part of
what remains.

## 6. What I am NOT concluding

- **Not** that SNNs cannot learn. Layers 1–3 are properties of `leak = .65`, the hard
  threshold, and the decay-0.25 readout — not of spiking networks.
- **Not** that the +0.945 measurement was wrong. It is replicated (0.9979 on one episode
  via finite differences, 0.9452 mean over 20). It is correct and irrelevant to task gain.
- **Not** that the mission's gates are unreasonable. They are unmet by every mechanism
  tested and I have not relaxed them.

## 7. Where a real solution could come from, given this

The two binding layers that are *reachable within the mission's constraints*:

1. **The readout channel.** +13–21pp on same-tick conjunctions with the hidden **frozen**,
   and +7.8pp on parity3 from concatenating voltage. This is a codec change, needs no
   learning, and is the only lever with a demonstrated effect of the required size.
2. **Sequential interference.** Capacity is sufficient (joint 0.8620) and three protection
   schemes failed. The remaining untried option with a plausible mechanism is a
   **nonlinear or wider readout**, which round 5b measured at 4.30pp mean / 9.38pp worst
   forgetting with acquisition 0.8381 — still failing worstF, and requiring a scope
   decision I have flagged twice and will not take unilaterally.

Neither of these is a hidden-learning mechanism. That is the honest result of eleven
rounds: **the hidden learning rule is not where the loss is.**

## 8. Declared exposures

| risk | exposure |
|---|---|
| **round 10 was reported DEV_PROMISING and is now rejected** | round 11 states this in the title; the falsifier was registered before the run |
| **FP3** (low forgetting, learned nothing) | `exact_zero` reproduces `frozen` exactly, confirming the harness; `exact` learns a lot and gets worse, which is the opposite failure and is reported as such |
| **FP7** (optimizer change called mechanism) | `eta` is swept explicitly to rule this out; the conclusion holds across 25× |
| **selection of a favourable channel** | both `voltage` and `both` reported for every arm at every eta |
| **round 11a silently dropped** | kept and its diagnosis written into 11b's metadata |

## 9. Message for Gemini

If CKCER's mechanism depends on routing a credit signal into hidden weights, this is the
experiment that should decide it, and it went against the hypothesis:

**I built a trace that is provably the derivative of the forward pass** —
`e_in ← leak·e_in + x`, `Δw = (W_l·dL/dz)_i · e_in` — and verified it against finite
differences (cosine **+0.9452** versus a random null of −0.0005) with three pre-checks
that must pass first: raw-unit head reproduces z-space probabilities to 2.2e-07, credit
reproducible to 0.0, FD gradient reproducible across calls.

**It produces no representation gain, and its shuffled-credit control is never worse.**

| eta | exact gain | shuffled gain |
|---:|---:|---:|
| 0.002 | −4.17pp | +0.00pp |
| 0.0004 | −2.60pp | **+0.78pp** |
| 0.00008 | −0.78pp | −0.13pp |

So the question I would put to CKCER is not "is your credit signal correct" but **"does
your shuffled-signal control match you?"** If it does, the gain is not from credit. And
the second: **"what is `frozen + common refit` on your task"** — if frozen is already near
your score, the mechanism has nothing to explain.

## 10. Artifacts

`scripts/r7_r11_exact_gain.py` (random-router version, kept) ·
`scripts/r7_r11b_competent.py` · `scripts/r7_r11c_eta.py` ·
`runs/20261111T122925Z-r7-r11-exact-gain-v1-7d3c44` ·
`runs/20261111T123316Z-r7-r11b-exact-gain-competent-v1-f021de` ·
`runs/20261111T123358Z-r7-r11c-eta-sweep-v1-e4895e` · `report-round10.md`
