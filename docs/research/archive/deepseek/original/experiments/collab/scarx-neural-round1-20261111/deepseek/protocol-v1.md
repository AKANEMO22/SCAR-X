# PROTOCOL v1 — deepseek — `strict_binding` is an invalid gate; `pair_binding` replaces it

**Campaign:** `scarx-neural-round1-20261111`
**Agent:** deepseek
**Environment:** DeepSeek Harness (DSH) Web, Windows 11
**Model actually used:** `deepseek-ai/deepseek-v4.1-flash-1017`
**BASE_SHA:** `1735ad66ea9e7b416f5df15df99d9c13cea46b4e`
**Hypothesis commit:** `82f4a5fc131d3647cb23f970f917127b19449c1f`
**Branch:** `research/deepseek/scarx-neural-round1-20261111`

**Independence:** disclosed in `hypothesis-v1.md` — same agent session as
`research/antigravity`. Agreement between the two is one agent agreeing with
itself. Reviewers must weight the evidence normally and independence at zero.

This protocol is committed **before** any formal run.

---

## 1. Authority and limits

Granted by the round-1 prompt: private branch/worktree, self-chosen hypothesis,
protocol first, private research prototype, push, Draft PR, cross-review both
peers, amend own branch on evidence.

Forbidden and not done here: merging to `main`, self-approving release, touching
Hermes weights/database, editing peer branches, lowering criteria after results,
paid APIs, overwriting historical runs. Proposal-30's approved signal boundary is
used unchanged and not widened.

## 2. Invariants

One Spiking RNN, LIF/ALIF, real synaptic learning, CPU-first, < 2 GB. No
Transformer/LLM/pretrained vision core, no proxy. No autograd/BPTT/RTRL in the
candidate. BPTT exists only in `src/scarx/evaluation/controls.py` as an isolated
control and is **not used in this branch at all**. Fast/Deep/Idle stay three modes
of one core. More neurons or data do not by themselves prove intelligence.

## 3. The problem being solved

`strict_binding` (`src/scarx/evaluation/binding_audit21.py:15`) computes
`label = values[query] XOR query`. That target is an **order-3** interaction
(key × value × query). An additive point neuron followed by a linear readout is
**order-1 by construction**, so the task is unreachable at *any* learning-rule
quality.

Measured on the raw input tape, fixed-feature ridge, no synapse trained:

| basis | seed 31001 | seed 31009 |
|---|---:|---:|
| order-1 | 0.4720 | 0.5220 |
| order-2 (key×value) | 0.4760 | 0.5180 |
| **order-3 (key×value×query)** | **1.0000** | **1.0000** |
| order-3, label-shuffled | 0.4850 | 0.4500 |

This is why the repository's isolated BPTT control also stalled at 63,54%
(report 35): the ceiling is the function class, not credit assignment. Every
binding failure in reports 18/20/35 and in both round-1 branches is therefore
partly a measurement artifact.

## 4. The candidate: `pair_binding` (order-2, no bag shortcut)

```
positions p1 ∈ {1,2}, p2 ∈ {4,5,6}
at each position: key marker on channel key, value marker on channel 2+value
values = {0: v, 1: 1−v}   with v = label      # values ALWAYS differ
label = value[key=0]
```

The two values always differ, so the multiset of writes is **identical for both
labels** (exactly one `0` marker and one `1` marker). Only the **pairing** of key
to value carries the label. This is what makes it a binding task rather than a bag
read.

**A first version of this generator was defective and is discarded.** It drew the
two values independently; when they coincided the multiset collapsed to `{0,0}` or
`{1,1}`, and an order-1 bag readout scored **0.70–0.77** — above the 0.65 gate.
That defect is recorded, the version is not used, and the gate below exists to
catch exactly this class of error.

Verified after the fix:

| | seed 31001 | 31009 | 31013 |
|---|---:|---:|---:|
| order-1 (must be ≈ chance) | 0.4700 | 0.4860 | 0.5160 |
| order-2 (must be solvable) | 1.0000 | 1.0000 | 1.0000 |
| order-2 label-shuffled control | 0.5120 | 0.5460 | 0.5060 |

## 5. Registered claims

- **C1 (task validity):** `pair_binding` is order-2 solvable with **no order-1
  shortcut**. Gate: order-2 probe ≥ 0.90 **and** order-1 probe ≤ 0.60, on every
  registered generator seed, with the shuffled control ≤ 0.60.
- **C2 (the task is not trivially readable):** a **frozen** hidden does not already
  clear the 65% gate on `pair_binding`. Refuted if frozen ≥ 0.65, which would make
  the task useless as a learning gate in the other direction.
- **C3 (learning is measurable):** the learned hidden beats the frozen hidden with
  a CI95% lower bound > 0 on `pair_binding`, while `order` (a known-positive
  control) shows the same, and `strict_binding` shows nothing because it is
  unsolvable.

**C2 and C3 are allowed to fail** and will be reported as failures. If C3 fails,
the deliverable is still valid: the campaign gains a *correct* binding gate and a
truthful statement that this architecture cannot learn binding within budget —
strictly better than a gate that cannot be passed or interpreted.

## 6. Equations — unchanged from the repository

Neuron (`Config`/`PlasticCore`, verbatim):

```
v[t]   = leak·v[t-1] + W_in·x[t] + Σ W_rec·z[t-1][idx]
limit  = threshold + b·a[t]
p[t]   = σ((v[t] − limit)/0.4);   z[t] ~ Bernoulli(p[t])
v[t]  -= z[t]·limit;  a[t] = adaptation_decay·a[t-1] + z[t]
f[t]   = feature_decay·f[t-1] + (1 − feature_decay)·z[t]
```

Hidden learning (approved proposal-30 path):

```
score[i]  = (z[i] − p[i]) / 0.4
elig_in  += score[i]·V_pre[x];  elig_rec += score[i]·V_pre[prev]
advantage = clip(reward − reward_mean, −2, 2)
ΔW_hidden = clip(hidden_lr·advantage·elig)
```

Readout: local delta rule `ΔW_out = readout_lr·err⊗f`.

**No new learning mechanism.** This branch changes the measurement instrument,
not the learner. The signal `W_out.T@error` is disclosed as not fully
biologically local, as proposal 30 requires.

## 7. Data and splits

Generators are repository-authored and deterministic. Splits use the repository's
established offset convention (`scripts/run_p1_paired_score.py`):

| role | offset | dev size | confirmation size |
|---|---|---|---|
| calibration | `+6 200 000` | 128 | 128 |
| training | `+6 300 000` | 2048 | 2048 |
| probe fit | `+6 400 000` | 128 | 128 |
| dev / test | `+6 500 000` | 256 | 512 |

Tasks: `pair_binding` (candidate), `strict_binding` (invalid-gate reference),
`order` (known-positive control).

Dataset SHA-256 per split is recorded in each run manifest.

**No leakage:** labels enter only the readout target; a P0 test asserts the input
tensor hash is unchanged under label permutation and that hidden updates are
label-independent when the reward is held equal.

## 8. Seeds

- **Development:** `41001, 41007, 41013` — fresh, verified unused in the repo.
- **Confirmation:** `42001, 42007, 42013, 42019, 42023, 42029, 42031, 42037` —
  8 fresh, disjoint from development and from all seeds in reports 20/21/27/35/36/37.
- One seed = one fresh initialisation. Three checkpoints of one model are never
  three seeds. No best-seed selection.

## 9. Arms

| arm | hidden learning | purpose |
|---|---|---|
| `score` | full local score×advantage | candidate |
| `frozen` | none | no-learning control (**the load-bearing control**) |
| `readout` | hidden frozen, head trained | readout-only |
| `shuffle` | sign-randomised modulation | signal-must-matter |

## 10. Budget

H=48, K=8, R=12, 2048 examples, 12 ticks, `blocks=4`. Screen: 3 tasks × 4 arms ×
3 seeds × 3 `readout_lr` = **108 trials**. Confirmation: same grid at the dev-locked
lr on 8 seeds. Hard cap **3 hours** per formal run. Peak process-tree RSS/commit
recorded. Two adjustment rounds maximum.

## 11. Gates and statistics

- Acquisition/final ≥ 65% (master value, not lowered).
- C1 as in §5.
- C2: `frozen < 0.65` on `pair_binding`.
- **C3 (primary):** CI95% lower bound of `score − frozen` > 0 on `pair_binding`.
- Secondary: `score − readout` > 0, and `score > shuffle`.
- Statistics: paired by seed; 95% intervals; sign-flip exact p; Holm across the
  three tasks. **Interval method named explicitly: seed-level bootstrap, 10 000
  resamples, `numpy.random.default_rng(2101)`** — the repository's registered
  convention (`scarx.evaluation.phase21.interval`). This is stated because the
  campaign already had one case where the choice of interval determined a verdict.
- Hyperparameters (`readout_lr ∈ {0.025, 0.05, 0.10}`) chosen on **development**
  only and locked before confirmation.

## 12. P0 — required before any capability number

1. Generator determinism: same seed → bit-identical `(x, y)`.
2. **No bag shortcut:** order-1 probe ≤ 0.60 on every dev generator seed (this is
   the check that caught the discarded first version).
3. No target leakage: input hash unchanged under label permutation; hidden update
   label-independent at equal reward.
4. `frozen` arm provably does not change hidden weights.
5. Checkpoint save/load round-trip bit-exact; cold-process resume reproduces the
   next learning step.
6. Rows are interchangeable: `pair_binding` returns shapes identical to
   `strict_binding` (12×8, 2 classes) so the harness cannot accidentally read the
   wrong task.
7. Peak RSS **and** commit measured in-process and across the process tree.
8. `python -m pytest -q` passes (pre-existing failures are recorded, with proof
   they pre-exist on the clean baseline).

If P0 fails, fix the implementation — never report a capability score from it.

## 13. Stop rule

- If C1 fails → the task is invalid; STOP, report as a design failure.
- If C2 fails (frozen already ≥ 0.65) → task is too easy for the head; STOP and
  report, do not re-tune the generator to make it harder.
- If C3 fails → **NO-GO for learning on binding**, reported honestly. No retune,
  no extra seeds, no longer budget, no renaming.
- Confirmation opens only if the dev screen passes and C1/C2 hold.

## 14. Claim scope

Only: *"`strict_binding` is order-3 and therefore an invalid gate for this
architecture; `pair_binding` is a measured order-2 replacement; on it, hidden
learning does/does not produce a measurable representation gain."*

Not claimed: normal chat, general vision, OCR, VQA, reasoning, retention,
lifelong learning, or general binding ability. Synthetic tasks, H=48, CPU.

## 15. Artifacts

```
runs/<unique-run-id>/                          result.json, report.md, manifest.json
artifacts/deepseek-r1/                         P0 receipt, locked lr
experiments/collab/<campaign>/deepseek/        this protocol
docs/research/collab/<campaign>/deepseek/      report, lessons-learned
```

No shared `latest.json`. Prior runs never rewritten.
