# MATHEMATICAL ANALYSIS — Round 5B

Brief §XIV requires ten specific items. Each is answered below, and the claims that
could be broken by a counterexample are checked numerically.

---

## 1. The label when vA and vB are independent

```
vA, vB ~ Uniform{0,1}, independent
q      ~ Uniform{0,1}          (0 = key A, 1 = key B)
y      = (1 − q)·vA + q·vB
```

## 2. Why this does not reduce to a one-bit XOR

R4A imposed `vB = 1 − vA`, which gives

```
y = (1−q)·s + q·(1−s) = s ⊕ q          with s = vA
```

so the label is a function of **one** latent bit plus the query. Here no such
substitution exists: `(vA, vB)` ranges over all four points of `{0,1}²`, and

```
y = vA + q·(vB − vA)
```

is a genuine function of **two** independent bits. The interaction term `q·(vB − vA)`
is nonzero exactly when `q = 1` and `vA ≠ vB`. This is the term a query-free readout
cannot form, and it is why the first-key baseline saturates at **0.75**, measured.

## 3. Independent information content of the memory state

The two writes carry `H(vA) + H(vB) = 2` bits. The full episode also carries the
query `H(q) = 1` bit, for **3 bits** total, of which the label consumes
`H(y) = 1` bit. Measured uniformities (Stage 0): four memory states at
0.2529 / 0.2373 / 0.2524 / 0.2573, query at 0.4927 / 0.5073, `corr(vA,vB) = +0.0207`.

## 4. Representational capacity of a linear readout

A linear readout on a **marginal** feature `φ(s) = Σ_t g(z_t, x_t)` computes

```
logit = W·φ(s)
```

There is no `q` in `φ`, so it cannot compute `q·(vB − vA)`. Its ceiling is the best
query-free predictor, which for this distribution is **0.75** — either "always say vA"
or "say vA, flipping when vA = vB", both 0.75. Measured: 0.7522 and 0.7473, and every
un-conditioned head landed 0.7480–0.7773.

I **do not** claim linear readouts are impossible in general — only that on a
*q-free* feature they saturate at 0.75. Adding the query linearly raises the ceiling
to `P(vA = vB) + P(vA ≠ vB)·(query used)` but still cannot exceed the joint structure
without an interaction: a linear head on `[φ(s), q]` can only shift its threshold with
`q`, not swap which stored value it reads.

## 5. Representational capacity of the bilinear interaction

R5 supplies

```
φ_R5(s,q) = [ s ; q ; (P s) ⊙ q₀ 1 ; (P s) ⊙ q₁ 1 ],   P fixed, 192×32
```

`(P s) ⊙ qₖ` is nonzero only when the query names key k, so a linear head on `φ_R5`
can implement "read the value of the queried key": it has a **separate weight block
per query value** applied to the same state. That is the interaction. Measured effect
at matched width 258: interaction 0.9635 vs no interaction 0.7637, and on the harder
`needB` subset 0.9254 vs 0.5643.

## 6. Role of the legal input query cue

The cue is read from the input at the query tick (`query_cue` locates the marker; it
does not assume tick 10). It is **not** knowledge the hidden learner discovered, and
the report says so. Its role is isolated by three arms:

| arm | acc | reading |
|---|---:|---|
| true cue | 0.9635 | baseline |
| cue shuffled in train | 0.7604 | cue must be informative during learning |
| cue shuffled at test | 0.8021 | cue must be correct at evaluation |
| query zeroed at core + cue | **0.9876** | the core need not see the query at all |

The last row is the important one: retrieval **improves** when the query never enters
the population, because the cue then selects without perturbing the stored state.

## 7. Limits of a fixed random projection

`P` is drawn once (`seed + 1234`) and never trained. Consequences:

- It is not adapted to the task, so the head must find the useful directions in a
  random 32-dim image of a 192-dim state. It works here (0.9635) because the state is
  low-dimensional in practice.
- It is tied to the **absolute** tick at which the state was captured. Under a shift a
  caller must pass the shifted `query_tick`; R4A's confirmation failed precisely
  because its harness did not. Here `capture(..., query_tick=...)` is a parameter and
  `locate_query_tick()` reads it from the data.
- It adds 192×32 = 6 144 floats = **24.6 KB** of fixed state.

**A caveat I proved against myself:** the naive "move events later, pad with zeros"
transform is a **no-op** for a zero-initialised recurrent net (`v = 0.65·0 + 0 = 0`).
Measured `max |Δ| = 0.0` on the post-query state. Any temporal claim built on it is
vacuous. The corrected transform pads with noise (`max |Δ| = 1.0`).

## 8. Local update equation

Unchanged registered delta rule:

```
err  = onehot(y) − p(·|φ)
ΔW   = clip(η · err ⊗ φ, −c, c)          η = 0.08, c = 0.025
Δb   = clip(η · err, −c, c)
W   += ΔW ;  b += Δb ;  W = clip(W, −3, 3)
```

Signals available: `φ(s,q)`, the head's own `p`, the teaching label, and its own
`W, b` and running statistics. **Not** available: hidden weights, hidden gradients,
other neurons, spike history beyond `s`, or any backward path. Hidden weights are
sha256-identical before and after (Stage 0 F).

## 9. Complexity

| quantity | value |
|---|---|
| state `s` | 192 |
| projection FLOPs / example | 192·32 ≈ 6 144 |
| readout parameters | 2 × 258 + 2 = **518** |
| update FLOPs / example | ≈ 7 176 |
| fixed extra state | 24.6 KB |
| process-tree commit (all runs) | 93.7 MB peak, target 2 GB |
| screen wall clock | < 60 s, budget 600 s |

## 10. Limits as the number of keys grows

With `K` keys of `V` values:

- stored bits scale as `K · log₂ V`;
- a first-order substrate's shortcut ceiling rises toward
  `1/K + (1 − 1/K)/V`, i.e. toward 1 as `V` grows, so the gate must be recomputed per
  configuration rather than reused;
- R5's projection must allocate one gated block per query value, so its dimension
  grows `O(d · V)`; at `V` large this dominates and a fixed random `P` will not suffice.

**This round measured only K=2, V=2.** Nothing here supports a claim beyond that.

## Counterexample checks run

| proposition | how checked | result |
|---|---|---|
| "first-key = 0.75" | measured on 4096 episodes | **0.7522** ✓ |
| "bag probe cannot separate 01 from 10" | measured `mixed_only` | **0.9947** — so binding is real, and the bag claim holds only *within* the collapsed class |
| "zero-pad shift is a no-op" | `max |Δ|` on post-state | **0.0** — confirmed, transform replaced |
| "query is not needed at the core" | query-zeroed + cue | **0.9876 > 0.9635** ✓ |
| "un-conditioned heads cannot exceed the ceiling" | 5 substrate arms | 0.7480–0.7773 ✓ |

## What I explicitly do NOT claim

- That ALIF is mathematically incapable of anything, based on a ridge result.
- That reading a frozen state is learning.
- That a surrogate gradient is the exact derivative of a hard spike (no gradient is
  used; the delta rule is a heuristic local update and is described as such).
- That R5 is deployable: the projection is fixed random and the cue is external.
