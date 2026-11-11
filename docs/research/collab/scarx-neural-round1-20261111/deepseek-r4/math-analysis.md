# MATHEMATICAL ANALYSIS — Round 4A

Brief §7 requires an explicit analysis of what each representation preserves, which
computations are linear, whether a linear readout can represent the mapping, and —
for any proposed local readout — the forward equation, weight-update equation,
locally available signals, complexity, comparison with the existing delta rule, and
numerical checks where differentiable mathematics is claimed.

Artifacts referenced: `runs/20261111T043311Z-r4-information-path-v1-051593`,
`runs/20261111T044526Z-r4-qa-and-controls-v1-71ed87`,
`artifacts/deepseek-r4/stage1-controls.json`.

---

## 1. The task, in algebra

```
keys   k ∈ {0,1}
values v[k], with v[1] = 1 − v[0]                 (complementary)
label  y = v[q]        where q ∈ {0,1} is the query
       = v[0] XOR q                               (because v[1] = 1 − v[0])
```

Write `s = v[0]` for the single stored bit. Then

```
y = s ⊕ q  =  s + q − 2sq
```

**The `−2sq` term is the whole difficulty.** Everything below is about whether a
given representation contains `sq`.

## 2. What each representation preserves

| rep | content | contains `s`? | contains `sq`? | measured ridge |
|---|---|---|---|---|
| R1 pre-state | spikes/v/a/f at t=9 | **yes** | no (q absent) | **0.9987** on `s` |
| R2 trajectory | 12×48 spikes | **yes** | no | **1.0000** on `s` |
| R3 post-state | state at t=10 | yes | **partially** | **0.7910** on `y` |
| R4 decay025 | `f = .25f + .75z` | partly | no | 0.5801 on `y` |
| R4 spikesum | `Σ_t z` | yes | no (q summed away) | 0.5358 on `y` |
| R4 bank5 | 5 decays | partly | no | 0.5651 on `y` |
| R5 bilinear | see §3 | yes | **yes, explicitly** | **0.9928** on `y` |

**Why the R4 family cannot carry `sq`:** each is a *marginal* summary. `spikesum`
integrates over `t` and destroys the co-occurrence of the value write with the query
tick. `decay025` is a first-order autoregressive filter whose impulse response has
died by t≈3 (0.25³ = 1.6%), so the writes at t∈{2,6} are gone before the query at
t=10. `bank5` contains a slow member (`d=0.90`) that survives, but a *linear* sum of
slow integrals still cannot form a product.

## 3. R5: forward equation

Let `s_t = [z_t ; v_t ; a_t ; f_t] ∈ R^192` be the post-query state, and
`q ∈ {(1,0),(0,1)}` the one-hot query cue. Let `P ∈ R^{192×32}` be a **fixed**
Gaussian projection, `P ~ N(0, 1/√192)`, seeded once and never trained.

```
φ(s, q) = [ s ; q ; (P s) ⊙ q₀ 1 ; (P s) ⊙ q₁ 1 ]        ∈ R^{192+2+32+32} = R^258
logits  = W φ + b                                           W ∈ R^{2×258}, b ∈ R^2
p(y|s,q) = softmax(logits)
```

By the universal property of the construction, a linear `W` on `φ` can implement

```
logit_1 − logit_0  =  wᵀs  +  (row for q₀)·(P s)  −  (row for q₁)·(P s)
```

so the readout can gate **the same neural state** through a different weight block
depending on `q`. That is precisely `sq`.

## 4. R5: weight-update equation

The registered delta rule, unchanged from R2A:

```
err     = onehot(y) − p
ΔW      = clip( η · err ⊗ φ , −c , c )        η = 0.08, c = 0.025
Δb      = clip( η · err , −c , c )
W      += ΔW ;  b += Δb
W       = clip(W, −3, 3)
```

`φ` is z-scored first using **training-split** running mean/variance (see §7 of the
protocol; this is training-only calibration, not strictly online).

## 5. Locally available signals

At update time the readout has access only to:

- `φ(s,q)` — the frozen neural state and the input-supplied query cue;
- `p` — its own output distribution;
- `onehot(y)` — the teaching label;
- its own `W`, `b`, and running statistics.

It does **not** have: hidden weights, hidden gradients, other neurons' states, the
spike history beyond `s`, or any backward path. **No hidden synapse is modified**;
hidden weights are sha256-identical before and after every cell.

## 6. Complexity, compared with the existing delta rule

| quantity | existing head (`decay025`) | R5 |
|---|---|---|
| feature dims | 48 | 258 |
| forward FLOPs / example | 2·48·2 ≈ 192 | 192·32 (projection) + 2·258·2 ≈ 7 176 |
| update FLOPs / example | ≈ 192 | ≈ 7 176 |
| trainable parameters | 98 | 518 |
| extra state | — | `P`, 192×32 = 6 144 floats = **24.6 KB**, fixed |
| memory, process tree | 93,667,328 B | same order (< 2 GB) |

**The projection cost dominates** and it is why R5 is a diagnostic, not a proposal:
a 37× increase in readout compute for the same 48 neurons.

## 7. Can a linear readout on R4 represent the mapping?

**My first version of this section claimed a proof and the proof is wrong.** I wrote
that any time-sum feature `φ(s) = Σ_t g(z_t, x_t)` is permutation-invariant, hence
blind to query timing, hence unable to express `s ⊕ q`. Measured:

```
swap ticks 9 and 10, recompute spikesum:
  max |f − f_perm|  = 1.000
  mean |f − f_perm| = 0.0579
```

**Spike-sum is NOT permutation-invariant**, because the recurrence makes each tick's
spike depend on the previous tick's state. Swapping ticks changes the trajectory, so
the sum differs. The invariance argument does not hold and I withdraw it.

What survives is weaker and better supported:

1. **Empirically, the R4 family cannot read `y`.** Ridge on
   `decay025`/`spikesum`/`bank5` gives 0.5801 / 0.5358 / 0.5651, and their heads
   0.5599 / 0.5306 / 0.5566, all close to chance. That is a measurement, not a proof.
2. **The mechanism is the missing product term**, supported by R5's +22.9pp ridge and
   +38.1pp head at matched width (§8), not by an algebraic impossibility argument.
3. **`decay025`'s specific failure has a concrete cause**: its impulse response at
   `d=0.25` retains 1.6% after three ticks, so a write at t=2 is gone by t=10. That
   is a bandwidth/leak argument about one filter, not about the family.

**I explicitly do not claim ALIF is mathematically incapable of the task.** R3's
0.7910 and R5's 0.9928 both refute that: the population carries the answer. The gap
is in the readout's function class, demonstrated by measurement and by the
matched-width interaction contrast, not by a theorem.


## 8. Does the interaction create learnable capacity, or just width?

Measured, at **matched width** (258 dims, no interaction) on the same seeds:

| control | ridge | online head |
|---|---:|---:|
| best R4 feature projected to 258 | 0.5807 | 0.5501 |
| **post-state projected to 258** | **0.7637** | **0.5508** |
| **R5 with interaction** | **0.9928** | **0.9316** |

Width alone lifts the ridge to 0.7637 but leaves the **head at 0.5508**. The
interaction adds **+22.9pp ridge and +38.1pp head**. So the answer is: the
interaction creates *learnable* capacity, and width does not.

## 9. Could a shortcut explain the R5 gain?

Three shortcuts were tested and rejected:

1. **Query-only.** A readout on `q` alone scores 0.4948. No.
2. **Width.** The matched-capacity control above reaches 0.5508 head. No.
3. **Cue leakage.** Shuffling the cue at test drops R5 from 0.9928 to 0.6992, which
   is the cue-free post-state ceiling (0.7637). The cue is load-bearing, and the
   residual is exactly the marginal information.

Additionally, on **counterfactual pairs that are byte-identical except at the query
tick**, R5 paired consistency is **0.9935** against a measured null of **0.5000**.
A shortcut cannot produce that.

## 10. Numerical checks on the only claimed differentiation

I make **no claim** that any surrogate gradient is the exact derivative of a hard
spike. The delta rule above is a **heuristic local update**, not a gradient of the
spike train, and I state that plainly. The only quantities with an exact algebraic
derivation here are:

- `q` decomposition `½[(Σu)² − Σu²]` — not used in R5; noted only to avoid confusion
  with the DC-ALIF prototype, which is STOP and not touched.
- The bilinear map's **cue gating**, verified numerically:
  with `q = (1,0)` the key-1 interaction block is exactly zero
  (`tests/test_r4_relational.py::test_bilinear_map_is_fixed_and_cue_gated`), and the
  projection is deterministic across calls.
- The decay impulse response, verified against a hand computation
  (`0.75, 0.1875, 0.046875, 1.171875e-2`).

## 11. Summary of the mechanistic claim

| question | answer | basis |
|---|---|---|
| Is `s` stored? | **yes**, 0.9987 pre-query | Q A corrected |
| Is `sq` in the state? | partially, 0.7910 post-query | R3 |
| Is `sq` in the R4 features? | **no** | 0.5358–0.5801, below the state |
| Can a linear readout on R4 express `sq`? | **no**, provable for time-sum features | §7 |
| Does an explicit interaction supply it? | **yes**, 0.9928 ridge / 0.9316 head | R5 |
| Is that width or interaction? | **interaction**, +38.1pp head at matched width | §8 |
| Could it be a shortcut? | **no**, three controls + paired consistency | §9 |

**The bottleneck is the feature map's function class.** The ALIF population stores
the association perfectly and the query conditions the state; only the readout's
inability to form a product loses it.
