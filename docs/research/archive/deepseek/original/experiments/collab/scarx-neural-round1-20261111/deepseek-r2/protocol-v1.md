# PROTOCOL v1 — Round 2A: substrate × reader (readout learnability)

**Campaign:** `scarx-neural-round1-20261111` → Round 2A
**Agent:** deepseek
**Environment:** DeepSeek Harness (DSH) Web, Windows 11
**Model:** `deepseek-ai/deepseek-v4.1-flash-1017`
**Branch:** `research/deepseek/r2-readout-learnability`
**Base:** `61eeb77cff2b371ec314d3e5bcc9e6fd29d00eef` (deepseek round-1 tip)
**Frozen ALIF:** H=48, fan-in=8, `plastic_rows=12`, `leak=.65`, `threshold=.5`,
`feature_decay=.25`, `adaptation_strength=.02`. Hidden weights never updated.

---

## 0. Disclosure of pre-protocol exploration

Three exploratory scripts were run **before** this protocol was committed:
`r2_stage0_reproduce.py`, `r2_diagnose_fit.py`, `r2_head_capacity.py`.

They are disclosed because their results shaped the design:

| exploration | result | how it shaped this protocol |
|---|---|---|
| Stage 0 reproduction | ridge decay 0.6348 / spike-sum 0.9733 / online head 0.4837 (means over grok's 3 seeds) | confirms the brief's contrast reproduces; **all numbers agree within seed variance** |
| fit diagnostic | head train 0.53 ≈ test 0.50 on decay-0.25 | rules out overfitting; points to underfitting or substrate poverty |
| head capacity | on **spike-sum**, head reaches **0.9759** vs oracle 0.9753 with **unchanged** hyperparameters | makes the substrate the primary variable |

**The registered hypothesis below was fixed after seeing these**, and the report
states this plainly. It is a pre-registration of the *confirmation and screen*,
not of the exploration.

## 1. Hypothesis

**H2A-1 (primary).** The online delta-rule head is **not** the binding constraint.
Given a frozen ALIF population and a readout substrate that carries the label, the
online head learns to exploit it at or near the offline ridge oracle. The observed
failure on production `feature_decay = 0.25` is caused by the **substrate**, not by
the head.

**H2A-2 (falsifier of H2A-1).** If, on the spike-sum substrate where the oracle is
≥ 0.80, the online head still stays below 0.65 with the registered
hyperparameters, then the head/rule *is* a binding constraint and H2A-1 is refuted.

**H2A-3 (control).** Changing substrate must not be confounded with capacity: the
spike-sum substrate has the same 48 dimensions as the production feature, so **no
matched-capacity control is needed**. A 5-decay bank (240 dims) is included and
*is* capacity-controlled by a fixed random projection down to 48 dims.

## 2. Conditions — substrate × reader

Substrates (all from the **same** frozen ALIF, hard-threshold sampler):

| id | substrate | dims |
|---|---|---|
| `decay025` | production `f[t]=0.25f+0.75z` | 48 |
| `spikesum` | Σₜ z[t] over the 12 ticks | 48 |
| `bank5` | concatenated decays {0.10,0.25,0.50,0.75,0.90} | 240 |
| `bank5_proj48` | `bank5` mapped through a fixed random projection | 48 |

Readers:

| id | reader |
|---|---|
| `oracle` | offline ridge, z-scored, α=0.01 (`tasks21.ridge_probe`) — **information ceiling only** |
| `head_default` | online delta rule, lr=0.08, clip=0.025, z-scored basis (grok's published config) |
| `head_budget` | online delta rule, **identical in every way to `head_default`** — the only registered alternative is not used in the screen unless `head_default` fails |

The brief allows "at most two head conditions beyond baseline". `head_default` is
the baseline-plus nothing; if it clears the screen, no second head is run. This is
deliberate: **the point of the experiment is to test the substrate, not to tune a
head.**

## 3. Data and splits — sim_xor, hard-threshold

Same generator as `point-neuron-nonlinearity.md`: two bits on one tick at
t ∈ {1,2,3}, label = XOR, response gate at t=10, raw-tape order-1 = 0.49.

| role | seed offset | count |
|---|---|---|
| calibration (feature statistics only) | `+100 000` | 128 |
| **training** | `+200 000` | **1024** |
| **dev / test** | `+300 000` | **512** |

`n_train = 1024` is registered because the exploration showed 256 underfits and
2048 changes nothing; 1024 is chosen once, on development, and is not revisited.

## 4. Seeds

- **Screen (development):** `81001, 81007, 81013` — fresh, unused in the repository.
- **Confirmation:** `82001, 82007, 82013, 82019, 82023, 82029, 82031, 82037` —
  8 fresh seeds, **not run** unless the screen passes.

Seeds are ALIF weight seeds. Each seed is one independent initialisation. The three
head randomisations per seed are **not** additional seeds and will not be counted
as such.

## 5. Budget

1 epoch for the screen (1024 updates per condition per seed), 4 epochs used only
in the exploration. Hard cap **600 s wall-clock** for the whole screen; if exceeded,
STOP and report. Peak process-tree RSS and commit measured.

Estimated: 4 substrates × 2 readers × 3 seeds = 24 cells; the exploration ran
similar work in well under 60 s.

## 6. Screen gate — registered exactly, not changed after seeing results

The screen passes if **all** of:

| gate | threshold |
|---|---|
| S1 oracle headroom | `oracle(spikesum) ≥ 0.80` on every dev seed |
| S2 online head clears | `head_default(spikesum) ≥ 0.65` on every dev seed |
| S3 improvement | `head_default(spikesum) − head_default(decay025) ≥ +0.10` |
| S4 gap narrowed | `head_default(spikesum) − oracle(spikesum) ≥ −0.10` |
| S5 no leakage | calibration statistics computed from training features only |
| S6 reproducibility | two runs with identical seeds produce bit-identical head weights |

If S1–S6 hold, confirmation (8 fresh seeds) opens with the same gates plus a paired
CI95% lower bound > 0 on `head(spikesum) − head(decay025)` and no regression on
`decay025` itself.

**No threshold may be changed after seeing the screen.** Two adjustment rounds
maximum, and any adjustment requires a new committed protocol version.

## 7. Metrics per cell

`direct` (online head test accuracy), `train_acc`, `mean_loss`, `clip_saturation`,
feature sparsity, feature rank (effective dimensionality), CPU seconds, peak RSS.
Every cell records its full per-seed row to JSON.

## 8. Stop rules

- S1 fails → the task cannot carry the experiment; STOP.
- S2 fails with S1 holding → **H2A-1 is refuted**; report that the head/rule is the
  binding constraint. Do **not** tune lr/clip further in this round.
- S5/S6 fail → technical failure; fix and re-run, do not report capability.
- Wall clock > 600 s → STOP with reason.

## 9. Identity and constraints

No core-architecture change, no neuron-count change, no P2 training, no Hermes
readout change, no live-checkpoint edit, no BPTT, no autograd, no pretrained
encoder, no Qwen, no answer lookup. Hidden weights are asserted unchanged.
Hard-threshold and Bernoulli **never appear in the same statistic**; Bernoulli is
not used in this protocol at all.

## 10. Claim scope

Only: *on `sim_xor` at H=48 with a hard-threshold ALIF population, whether an online
delta-rule head can exploit a substrate the offline ridge oracle can exploit.*

**Not claimed:** that SCAR-X "learns", that any chat/vision/reasoning capability
exists, that retention holds, or that the production readout should be replaced.
A ridge oracle is an information ceiling, never a skill.
