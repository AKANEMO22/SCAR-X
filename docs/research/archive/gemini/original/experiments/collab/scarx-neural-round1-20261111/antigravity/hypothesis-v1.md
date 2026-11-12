# HYPOTHESIS v1 — Antigravity — campaign `scarx-neural-round1-20261111`

**Agent:** antigravity
**Environment actually used:** DeepSeek Harness (DSH) Web, Lead session, Windows 11 host
**Model actually used:** `deepseek-ai/deepseek-v4.1-flash-1017`
*(Antigravity is the AGENT_ID / environment label; the executing model is recorded
above and must not be conflated with the environment name.)*
**BASE_SHA:** `1735ad66ea9e7b416f5df15df99d9c13cea46b4e`
**Branch:** `research/antigravity/scarx-neural-round1-20261111`
**Committed before reading any peer's unpublished results.** At the time of this
commit I had observed only peer *branch names and HEAD SHAs* via
`git ls-remote`; no peer protocol, report, code or result was opened.

---

## 1. The specific problem

Every recent SCAR-X representation round reports the same shape of failure:

- report 26 (word-score v2): dense hidden dev **7,53%**, frozen **8,60%** — no gain;
  family STOPPED.
- report 27 (isolated learnability): local score **7,53%** vs global-BPTT control
  **44,09%**.
- report 34 (information-path audit): head feature probe **8,30%** frozen /
  **11,60%** control; raw upstream probe **50,22%**. Information is present
  upstream and largely unreadable downstream.
- report 35 / 36 / 37: only "frozen gain" passes; accuracy, novel-composition and
  causal-swap gates fail.
- report 20 (P1 paired-score): **did** pass on `order` (+18,33) and `xor`
  (+20,00) but **failed `strict_binding`** at 49,56%, and report 35's isolated
  BPTT control also reached only **63,54%** on the same task.

The repository's reading of these results is that the *learning signal* is the
problem (scalar-reward noise, credit assignment, local vs global). Report 30's
approved local+`W_out.T @ error` boundary is the response to that reading.

**I do not think that reading is complete, and I will test the alternative
reading.** Two upstream facts are being attributed to credit assignment without
being isolated:

1. the readout integrates spikes through **one** scalar decay
   (`feature_decay=0.25` in the shipping `Config`), i.e. a first-order temporal
   basis — it is blind to everything outside that single timescale;
2. `strict_binding` **cannot** be expressed by an additive point neuron plus a
   linear readout **at any learning-rule quality**.

## 2. Root-cause evidence (measured this session, exploratory scratch only)

Scratch scripts: `local/antigravity-r1/scratch/` (git-ignored, not a result).
Every number below is a **fixed-feature ridge probe on frozen weights**; no
synapse was trained. These are design-time diagnostics that justify the
hypothesis — they are **not** results and are **not** claims.

### 2.1 `strict_binding` requires a third-order feature (task-validity finding)

`strict_binding` computes `label = values[query] XOR query`. `values[query]` is an
order-2 term (key marker × value marker at one position); XOR with the query
marker at position 10 makes the target **order-3**. Ridge on the raw tape:

| Basis | seed 12101 | seed 12109 |
|---|---:|---:|
| order-1 (additive channels) | 0,5175 | 0,5025 |
| order-2 (key × value) | 0,5350 | 0,5225 |
| **order-3 (key × value × query)** | **1,0000** | **1,0000** |
| order-3, **label-shuffled control** | 0,4850 | 0,4500 |

An additive integrator followed by a linear readout is first-order. So on this
task **no** learning rule can pass the 65% gate — which is exactly why the
isolated BPTT control also stalled at 63,54%. The binding failure is at least
partly a **task/expressivity** artifact, not evidence against local learning.

### 2.2 The single-decay readout is a temporal bottleneck, and it is not parameters

Task `order` (a task that *did* pass the representation gate), same untrained ALIF,
same tick budget:

| Readout substrate | seed 12101 | seed 12109 | seed 12113 |
|---|---:|---:|---:|
| production single decay `d=0.25` (48d) | 0,6594 | 0,5906 | 0,6813 |
| **dimension-matched** single decay, fixed random projection to 240d | **0,6594** | **0,5906** | **0,6813** |
| **5-decay bank** `d∈{0,1;0,25;0,5;0,75;0,9}` (240d) | **0,9469** | **0,9563** | **0,9500** |
| different 5-decay bank `d∈{0,15;0,35;0,55;0,7;0,85}` (240d) | 0,9437 | 0,9563 | 0,9531 |

The dimension-matched control is **bit-identical** to the single-decay row. So the
gain is **not** extra readout parameters and **not** a lucky decay set: it is the
*mutliplicity of timescales* that recovers label information the single integral
destroys. On `xor` the same bank gives no gain (0,49→0,55; 0,55→0,45), so this is
task-conditional, not a universal improvement — the hypothesis must predict
*where* it helps, and I will report the tasks where it does not.

## 3. Hypothesis

> **H1.** A substantial part of SCAR-X's measured "representation failure" is a
> **readout temporal-basis bottleneck**, not a credit-assignment failure. Giving
> the *readout* a bank of leaky integrators spanning multiple timescales — with
> neuron dynamics, spiking, recurrence and the hidden learning rule held exactly
> identical — will (a) raise acquisition on tasks whose label needs multi-timescale
> integration, and (b) *increase the measured hidden-representation gain*,
> because the readout stops masking what the hidden weights learned.

> **H1-null.** Any apparent gain is explained by added readout parameters, extra
> compute, or a changed evaluation basis. Predicted null: the dimension-matched
> single-timescale control closes the gap (my probe says it will not, but the
> registered experiment must be able to show that it does).

**Falsifiers — the hypothesis is rejected if any of these hold:**
1. MTB's gain over the dimension-matched control has CI95% lower bound ≤ 0 (dev),
   i.e. it is parameter/compute, not timescale multiplicity.
2. Hidden-representation gain (refit head, learned vs initial hidden, same basis
   and budget) fails the registered gate on ≥2 of 3 tasks.
3. `shuffled-signal` control performs as well as MTB (no dependence on a real
   learning signal).
4. Acquisition/final <65%, or mean forgetting >3pp / worst >5pp.
5. Spike trains differ between MTB and dimension-matched control (would prove the
   condition is not readout-only and invalidate the attribution).

## 4. Equations and signal path

Neuron and hidden learning are **unchanged** from `Config`/`SpikingCore`:

```
v[t]   = leak·v[t-1] + W_in·x[t] + Σ_j W_rec[i,j]·z[t-1][idx]
limit  = threshold + b·a[t]
p[t]   = σ((v[t]-limit)/0.4);   z[t] ~ Bernoulli(p[t])
v[t]  -= z[t]·limit;   a[t] = adaptation_decay·a[t-1] + z[t]
```

**Only** the readout substrate changes. Production (K=1):

```
f[t] = feature_decay·f[t-1] + (1-feature_decay)·z[t]
```

Candidate MTB (K decays, one shared weight matrix, `W_out ∈ R^{O×KH}`):

```
f_k[t] = d_k·f_k[t-1] + (1-d_k)·z[t],   d_k ∈ {0.10,0.25,0.50,0.75,0.90}
φ[t]   = [f_1[t] ; … ; f_K[t]]           (K·H)
```

Readout learning (readout-local delta rule — unchanged in form, wider input):

```
err    = onehot(y) - softmax(W_out·φ + bias)
ΔW_out = readout_lr · err ⊗ φ        (clipped)
Δbias  = readout_lr · err            (clipped)
```

Hidden learning (unchanged, the approved-forward-only path):

```
score[i]  = (z[i] - p[i]) / 0.4
elig_in  += score[i] · V_pre[x]      elig_rec += score[i] · V_pre[prev]
advantage = clip(reward - reward_mean, -2, 2)
ΔW_hidden = hidden_lr · advantage · elig        (clipped, active rows only)
```

No autograd, no BPTT, no RTRL, no hidden→hidden error transport, no reverse
traversal. The only cross-neuron signal is the readout-local error used to update
`W_out`, which is exactly the boundary approved in proposal 30; MTB does not
widen it.

## 5. Auxiliary state, RAM and compute

- Extra state: `(K-1)·H` float32 feature channels = `4·48·4 = 768 B` at H=48, K=5.
- Extra weights: `O·(K-1)·H` float32; at O=2, H=48, K=5 → `2·4·48 = 384 floats
  = 1 536 B`. **Both deltas are reported, and the dimension-matched control spends
  exactly the same bytes** so parameter count is not the confound.
- Compute: readout forward `O·K·H` vs `O·H`; readout update likewise. Hidden
  forward/update cost is **identical**. Equal-update and equal-*time* comparisons
  are both reported.
- RAM: projected well under the 2 000 000 000 B ceiling; peak RSS/commit measured
  in-process **and** across the process tree for learn/save/load/rollback.

## 6. What already exists in prior research — and why this is not a re-tune

Report 22 proposed **per-neuron heterogeneous time constants** (membrane /
adaptation / feature decay *per neuron*) and screened it as an inference view on
immutable weights: `STOP_TIMESCALE_VIEW_SCREEN_V1` (leak-spread 47,47%,
all-spread 47,28%, mean-tau control 54,11%). That family is STOPPED and I am **not**
reopening it. The mechanical differences, all of which the protocol tests:

| | report 22 (STOP) | this hypothesis (MTB) |
|---|---|---|
| what changes | neuron dynamics: `leak`, `adaptation_decay`, `feature_decay` become **per-neuron vectors** | **nothing in the neuron**; single scalar `leak` and adaptation retained |
| spike trains | differ (dynamics differ) | **bit-identical** by construction — asserted in P0 |
| learning | none (immutable-weight view screen) | real hidden + readout learning under the approved rule |
| what is measured | refit-head accuracy of decay *views* | acquisition, hidden-representation gain, retention, with matched-dimension control |
| control | mean-tau | **dimension-matched single timescale** (my probe shows it is bit-equal to production, so the confound is isolated to timescale multiplicity) |

This is a **readout receptive-field** question, not a neuron-heterogeneity
question. It is also not a re-tune of reports 26/27/35/36/37: those changed the
encoder, the task adapter, the head objective or the learning signal; MTB changes
only the readout's temporal basis and keeps the hidden rule they used.

Also distinct from report 24 (beam search: inference-only, no learning) and
report 38 (readonly objective audit: no training).

## 7. Which result would reject the hypothesis

See §3 falsifiers. Concretely, I will report **NO-GO** and say so plainly if
MTB does not beat the dimension-matched control, or if it beats the control but
fails to raise the hidden-representation gain, or if it raises `order` while
leaving `xor`/`binding` unchanged and I therefore cannot claim a general
improvement. A gain confined to one synthetic task with no hidden-side effect
would be reported as **PIVOT**, not GO.

## 8. Scope of claims I will NOT make

No claim of normal chat, general vision, OCR, VQA, reasoning, or lifelong skill
retention. No claim that SCAR-X now "understands" anything. Any accuracy figure is
inference on a registered synthetic task at H=48. The 65% / ≤3pp / ≤5pp gates and
the 8-seed confirmation requirement are kept exactly as registered in the master
plan; I will not lower them after seeing results.
