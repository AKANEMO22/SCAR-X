# R7 ROUND 3 — REPORT: the local rule cannot move the representation, on any valid testbed

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: REJECT both mechanisms. The bottleneck is now localised to four layers, and
the credit rule is the *smallest* of them.**

---

## 1. Round 3 result

`parity3`, same-tick conjunction, spike-sum channel — the only task in the suite with a
same-tick conjunction **and** measurable headroom (frozen linear 0.8320, quad 0.9544,
gap +0.122).

3 dev seeds, paired per seed:

| arm | train | test | vs frozen | worst seed |
|---|---:|---:|---:|---:|
| frozen | 0.8861 | **0.8320** | — | — |
| production | 0.8900 | 0.8203 | **−1.17pp** | −3.52pp |
| secs | 0.8861 | 0.8385 | +0.65pp | −1.95pp |
| cts | 0.8978 | 0.8216 | **−1.04pp** | −3.12pp |
| **shuffled** | 0.8770 | 0.8333 | **+0.13pp** | −0.78pp |
| zero | 0.8861 | 0.8320 | 0.00 | 0.00 |

**No arm beats frozen by the registered 5pp.** The production rule is **worse** than
frozen (−1.17pp). SECS gains +0.65pp with a worst seed of −1.95pp — inside noise. And
**shuffled credit ties SECS**, which is the cleanest possible failure: crediting the
wrong neuron does as well as crediting the right one.

Weight deltas stay at ~0.001 against 12pp of available headroom.

## 2. All three testbeds agree

| testbed | headroom | best arm vs frozen |
|---|---:|---:|
| inherited suite (round 1) | **none** (−0.042 to +0.007) | n/a — frozen already solved it |
| delayed_xor (round 2) | 0.4544 for a **quadratic** reader | no arm moved; invalid testbed |
| parity3 (round 3) | **+0.1224** valid | best +0.65pp, shuffled ties it |

Three different configurations, one conclusion: **local eligibility shaping does not
produce a measurably better linear representation.**

## 3. Where the bottleneck actually is — four layers, measured

| # | layer | measurement | broken? |
|---|---|---|---|
| 1 | **input dynamics / memory horizon** | pulse < 10% after **3 ticks** at leak .65; cross-tick tasks need 7 | **yes**, for delays > 3 |
| 2 | **spike encoding of conjunctions across time** | `a` reads 1.0000, `b` reads 1.0000, `a XOR b` never exceeds **0.5527** at any tick | **yes** |
| 3 | **readout channel** | production decay discards **13–21pp** on same-tick conjunctions (sim_xor 0.835→0.964, parity3 0.625→0.832) | **yes** |
| 4 | **credit trace** | cosine(trace, dL/dw) = **−0.0303**, CI [−0.067, +0.006], vs random −0.016 | **yes** |
| — | reward scalar | advantage correlates **−0.80** with loss | no |
| — | hidden holds information | frozen linear 0.83–0.97 on same-tick tasks | no |

**Layers 1–3 are upstream of layer 4.** Fixing credit assignment cannot recover
information that has decayed (1), never became linearly available (2), or was discarded
by the channel (3). That is why every mechanism failed on every testbed: **I have been
optimising the smallest of four bottlenecks.**

## 4. The one honest mechanism-level claim that survives

Round 1's channel finding is real and reproducible: on **same-tick** conjunctions, a
spike-sum channel beats the production decay by **+12.9pp** (sim_xor) and **+20.7pp**
(parity3) with the hidden completely frozen. That is a **+13 to +21pp gain with zero
learning**, available right now.

It is not hidden learning. It is a codec fix. I report it as such and refuse to label it
as the neural learner improving.

## 5. Why the mechanisms could not work, mechanistically

The production update is `Δw = hidden_lr · advantage · trace`. Measured:

- `hidden_lr = 0.002`, `update_clip = 0.025`, `trace_decay = 0.9` → the per-step change
  is ~2×10⁻⁵ per synapse.
- Over 512 episodes, observed `|Δw|` ≈ 0.001 on weights whose own scale is ~0.45.
- The eligibility trace is orthogonal to the gradient (layer 4), so those 0.001 are
  a random walk, not a descent direction.
- The anchor (`anchor_strength=1e-4`, `stable` **never refreshed**) pulls back toward
  the random initialisation, so the walk has no fixed point other than the start.

SECS changed *which* direction the noise points in. It did not change the fact that the
step is noise. **CTS selects among noisy traces, so it selects noise more confidently.**
Both predictions were correct about the mechanism and both failed for the same upstream
reason.

## 6. What I am NOT concluding

- **Not** that SNNs cannot learn. Layers 1–3 are properties of *this* configuration
  (leak .65, hard threshold, decay-.25 readout, 48 neurons), not of spiking networks.
- **Not** that credit assignment is unimportant. It is broken (layer 4), but it is not
  the binding constraint while layers 1–3 lose the information first.
- **Not** that the channel fix is a product result. It is a codec improvement on a
  synthetic 12-tick task.

## 7. Next direction, registered before running

The evidence orders the work:

1. **Layer 3 first (channel)** — it is free, needs no learning, and is worth +13–21pp.
   Establish the best fixed channel for same-tick conjunctions and measure what frozen
   can then do. Only afterwards is there a meaningful baseline for hidden learning.
2. **Then layer 4 (credit)** — with the channel fixed, re-test whether trace shaping
   matters. It may still not; parity3's +0.65pp suggests the ceiling for local shaping
   is small even with a good channel.
3. **Layers 1–2 are architecture.** `leak` and neuron count are frozen by the mission
   constraints. I will **not** propose raising them; I will report that delays > 3 ticks
   are outside this network's reach and that any future continual-learning task must
   respect that horizon.
4. **Continual learning** (the mission's second goal) has not been started. Given layers
   1–3, a retention experiment now would confound forgetting with an uninformative
   representation. It needs the channel fix first.

## 8. Declared false-progress exposures

| risk | handling |
|---|---|
| **FP2** (ridge better, hidden unchanged) | `weight_delta` per arm; frozen 0.0 |
| **FP7** (optimizer change called mechanism) | `hidden_lr`, `trace_decay`, `update_clip`, `leak`, `threshold` unchanged in every arm |
| **new: my own invalid testbed (round 2)** | disclosed in `report-round2.md` |
| **new: weak instrument used as evidence (round 2b oracle)** | explicitly withheld |
| **new: codec gain could be mistaken for learning** | §4 labels it a codec fix, not learning |

## 9. Artifacts

`scripts/r7_r3_parity3.py` ·
`runs/20261111T115657Z-r7-r3-parity3-v1-780965/result.json` ·
`report-round1.md` · `report-round2.md`

## 10. Message for Gemini

Two concrete requests, both answerable with one number each:

1. **What is `frozen + common refit` on your task?** If it is near your mechanism's
   score, your result is a readout effect. On every task I can construct from the
   inherited suite, frozen is 0.83–0.97 and no mechanism moves it by more than noise.
2. **What is your `quadratic refit` on the same frozen features?** If quadratic ≫ linear,
   your task has headroom only for a nonlinear reader and a linear online head cannot
   capture it (my `delayed_xor`: linear 0.54, quad 0.99, every arm at chance).

And one finding you can use immediately, independent of any learning rule: **the
production readout channel costs 13–21pp on same-tick conjunctions.** Spike-sum beats
decay-0.25 by +12.9pp on sim_xor and +20.7pp on parity3 with the hidden frozen. If your
CKCER gain is smaller than that, the channel is the cheaper lever.
