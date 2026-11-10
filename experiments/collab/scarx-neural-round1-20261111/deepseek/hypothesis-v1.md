# HYPOTHESIS v1 — deepseek — campaign `scarx-neural-round1-20261111`

**Branch:** `research/deepseek/scarx-neural-round1-20261111`
**BASE_SHA:** `1735ad66ea9e7b416f5df15df99d9c13cea46b4e`
**Environment:** DeepSeek Harness (DSH) Web, Windows 11
**Model actually used:** `deepseek-ai/deepseek-v4.1-flash-1017`

## ⚠️ INDEPENDENCE DISCLOSURE — read this first

**This branch was authored by the same agent session that authored
`research/antigravity/scarx-neural-round1-20261111`**, at the explicit direction
of the user because no independent `deepseek` session existed to produce it.

Consequences a reviewer must apply:

1. **This is NOT an independent third party.** Any agreement between `deepseek` and
   `antigravity` is one agent agreeing with itself and must **not** be counted as
   convergent evidence.
2. This branch was written **after** seeing both `antigravity`'s and `grok`'s
   results. It is therefore not blind, and its hypothesis is necessarily informed
   by their findings.
3. What it *can* still legitimately offer: a **new and testable claim** that neither
   prior branch made, verified by its own registered experiment. Novelty of the
   claim is checkable even when independence of the author is not.
4. A reviewer should weight this branch's *evidence* normally and its
   *independence* at zero.

## 1. The specific problem

Both prior branches measured a large, CI-secure accuracy gain on the `order` task
and both ended at the same wall: **a frozen hidden was as good as a learned one.**

- grok: `slow-zero == slow-frozen` on all 8 seeds.
- antigravity: `mtb − mtb_frozen = +0.43pp [−0.91, +1.87]`.

So neither branch demonstrated hidden learning. But there is a deeper problem that
both branches *touched and neither resolved*:

**`strict_binding` — the only task in the suite that is supposed to require
binding — is unsolvable by construction.**

`strict_binding` computes `label = values[query] XOR query`. The target is an
**order-3** interaction (key marker × value marker × query marker). An additive
point neuron followed by a linear readout is order-1 by construction.

Measured on the raw input tape with fixed-feature ridge probes (no synapse trained,
label-shuffled control in the same table):

| basis | seed 12101 | seed 12109 |
|---|---:|---:|
| order-1 (additive channels) | 0.5175 | 0.5025 |
| order-2 (key × value at one position) | 0.5350 | 0.5225 |
| **order-3 (key × value × query)** | **1.0000** | **1.0000** |
| order-3, **label-shuffled control** | 0.4850 | 0.4500 |

The task *is* solvable — order-3 reaches 1.00 — but it is not solvable **by this
architecture at any learning-rule quality**. That is why the repository's own
isolated BPTT control also stalled at 63,54% (report 35): the ceiling is the
function class, not the credit assignment.

**This invalidates `strict_binding` as a gate.** Every "binding failure" in this
repository — reports 18, 20, 35, and both round-1 branches — is partly measuring
the task, not the learner.

## 2. The proposed fix

Replace the unsolvable gate with a **rank-2 binding task** that preserves what
binding means (a key/value conjunction that a bag-of-events cannot answer) while
being expressible by an additive neuron + linear readout:

```
pair_binding(seed, count):
  for each example:
    2 positions p1 ∈ {1,2}, p2 ∈ {4,5,6}
    for each position: key marker on channel key, value marker on channel 2+value
    label  = value[key=0]  XOR  value[key=1]        # order-2, no query triple
```

The conjunction `value[key]` is order-2 (key marker × value marker at the same
position). XOR-ing two order-2 terms is still order-2 in the joint basis. A bag of
all writes cannot answer it, because both keys and both values appear exactly once
in every example — so the label is determined only by the *pairing*.

**This is the point.** `pair_binding` is a real binding test that the architecture
can actually attempt. It gives the campaign, for the first time, a binding gate
whose failure would mean something about the learner.

## 3. What is claimed, and what would refute it

**Claim C1 (task validity):** `pair_binding` is solvable by an additive point
neuron + linear readout. Refuted if an order-2 fixed-feature probe on the raw tape
does not exceed chance substantially, or if its label-shuffled control is not near
chance.

**Claim C2 (nothing-to-learn check):** on `pair_binding`, a *frozen* hidden is not
already sufficient — i.e. unlike `order`, the readout alone cannot clear the gate.
Refuted if `frozen` ≈ `learned`, which would make the task useless as a learning
gate in the other direction (too easy for the readout).

**Claim C3 (learning is measurable):** if C1 and C2 hold, hidden-representation
gain from local score×advantage learning is reportable with a meaningful sign on
`pair_binding`, and the frozen/readout-only controls isolate it.

**C3 is allowed to fail.** If it fails, the honest conclusion is that this
architecture still cannot learn binding, but the campaign now has a *valid* task
to say so — which is itself the deliverable, and is strictly better than the
current state where the gate cannot be passed or interpreted at all.

## 4. Equations and signal path

Unchanged from the repository's approved local rule. Neuron:

```
v[t] = leak·v[t-1] + W_in·x[t] + Σ W_rec·z[t-1][idx]
limit = threshold + b·a[t]
z[t] ~ Bernoulli(σ((v[t]−limit)/0.4))
a[t] = adaptation_decay·a[t-1] + z[t]
f[t] = feature_decay·f[t-1] + (1-feature_decay)·z[t]
```

Hidden learning (approved proposal-30 boundary, unchanged):

```
score[i]  = (z[i] − p[i]) / 0.4
elig_in  += score[i] · V_pre[x];  elig_rec += score[i] · V_pre[prev]
advantage = clip(reward − reward_mean, −2, 2)
ΔW_hidden = clip(hidden_lr · advantage · elig)
```

No autograd, no BPTT, no RTRL, no hidden→hidden error transport. Readout uses its
own local delta rule. The one-step `W_out.T @ error` factor is disclosed as not
fully biologically local, as proposal 30 requires.

**No new mechanism is introduced.** This branch changes the *measurement
instrument*, not the learner. That is deliberate: the diagnosis is that the
instrument was broken.

## 5. Auxiliary state, RAM, compute

No new state beyond the generator. `pair_binding` is a pure function returning
`(x, y)` with the same shapes as `strict_binding` (12 ticks × 8 channels, 2
classes). RAM and compute are therefore identical to the existing suite; the gate
re-runs inside the same 2 GB envelope with the same monitor.

## 6. Already present in prior research

- The generator style, tick counts, channels and ridge-probe helper are the
  repository's own (`tasks21.representation`, `binding_audit21.strict_binding`,
  `tasks21.ridge_probe`). This branch adds one generator and does not modify them.
- Report 20 established that local paired-score passes `order`/`xor` and fails
  `strict_binding`. This branch explains *why* that failure was uninformative and
  supplies a test that can distinguish the explanations.
- It does **not** reopen the STOPPED families (22 timescale views, 26/27 word
  score, 35/36/37 bridges). It changes no learning rule, no encoder, no history.

## 7. Scope of claims

Only: *"`strict_binding` is order-3 and therefore not a valid gate for this
architecture; `pair_binding` is an order-2 replacement whose validity is measured,
and it does/does not reveal measurable hidden learning."*

Explicitly **not** claimed: normal chat, general vision, reasoning, retention,
lifelong learning, or that SCAR-X now binds generally. Synthetic tasks at H=48 on
CPU.
