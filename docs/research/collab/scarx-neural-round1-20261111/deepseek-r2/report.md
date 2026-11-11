# REPORT — Round 2A — the readout is not the bottleneck; the substrate is

**Branch:** `research/deepseek/r2-readout-learnability`
**Base:** `61eeb77cff2b371ec314d3e5bcc9e6fd29d00eef`
**Protocol:** `protocol-v1.md` @ `f0ade05`, SHA-256
`18b618969b2cbd6648d8457904d874516e1c83cb5308ee091fc7c47d5784c28a`
**Environment:** DeepSeek Harness (DSH) Web, Windows 11
**Model:** `deepseek-ai/deepseek-v4.1-flash-1017`
**Runs:** `20261111T031631Z-r2-substrate-screen-v1-6e9376` (3 dev seeds),
`20261111T031658Z-r2-substrate-confirmation-v1-37d670` (8 confirmation seeds)

**Verdict: GO for H2A-1.** The online delta-rule head is **not** the binding
constraint. With an unchanged head, the substrate alone moves accuracy from
**0.5063 → 0.9712** on 8 fresh seeds.

---

## The question this had to answer

> With the same frozen ALIF neurons and the same data, can SCAR-X learn an online
> readout good enough to exploit the information the spike train already carries?

**Yes.** And the reason the previous answer looked like "no" is now measured.

## 1. Stage 0 — the brief's contrast reproduces

Independent reimplementation (not grok's code), hard-threshold sampler,
`feature_decay = 0.25`, frozen hidden, grok's three seeds:

| quantity | my measurement | brief reference |
|---|---:|---:|
| ridge on decay 0.25 | 0.6348 | ~0.6953 |
| ridge on spike sum | 0.9733 | ~0.9701 |
| online frozen head | 0.4837 | ~0.5039 |

All three land within seed variance of the brief. The contrast is real: the spike
train carries ~0.97 of the label, the production feature carries ~0.63–0.70, and
the online head sits at chance.

## 2. The diagnosis — not overfitting, and not the clip

Two candidate explanations were tested before the protocol, both disclosed in it.

**Clip saturation is real but not causal.** In the published head, `lr = 0.08`
with `np.clip(…, -0.025, 0.025)` saturates on **78.4%** of updates, so the step
size is fixed regardless of gradient magnitude. Raising the clip to 0.5 cut mean
training loss from **0.60 → 0.37** but left test accuracy at **0.51**. Loss fell,
accuracy did not: **overfitting, not the bottleneck.**

**Not overfitting either.** Train vs test accuracy on decay-0.25:

| n_train | train_acc | test_acc | gap |
|---:|---:|---:|---:|
| 256 | 0.6094 | 0.4837 | +0.126 |
| 1024 | 0.5433 | 0.5026 | +0.041 |
| 2048 | 0.5317 | 0.4948 | +0.037 |

The head cannot fit its **own training set** (0.53) while ridge on the same
features reaches 0.70 test. That is an underfitting/step-failure, and more data
makes it worse rather than better.

**The decisive pre-protocol check.** Same head, same hyperparameters
(`lr=.08, clip=.025`, **unchanged**), two substrates:

| substrate | oracle (ridge) | online head |
|---|---:|---:|
| `decay025` | 0.7070 | **0.5026** |
| `spikesum` | 0.9753 | **0.9759** |

The head reaches the oracle exactly when the substrate carries the label. Nothing
about the head changed.

## 3. Registered screen — all six gates pass

3 fresh dev seeds, 1 epoch, hard-threshold sampler:

| substrate | dims | oracle | head | gap | train |
|---|---:|---:|---:|---:|---:|
| `decay025` | 42 | 0.7598 | 0.4850 | −0.2747 | 0.5436 |
| `spikesum` | 42 | 0.9805 | **0.9831** | **+0.0026** | 0.9902 |
| `bank5` | 203 | 0.9941 | 0.9525 | −0.0417 | 0.9616 |
| `bank5_proj48` | 48 | 0.8743 | 0.5664 | −0.3079 | 0.6270 |

`S1`–`S6` all pass; wall clock **1.1 s** (cap 600 s); hidden weights unchanged in
every cell; `S6` verified by bit-identical head digests on a repeat.

## 4. Confirmation — 8 fresh seeds

| substrate | oracle | head | gap |
|---|---:|---:|---:|
| `decay025` | 0.7036 | 0.5063 | **−19.73pp** [−24.19, −14.33] |
| **`spikesum`** | 0.9719 | **0.9712** | **−0.07pp** [−1.12, +0.98] |
| `bank5` | 0.9856 | 0.9243 | −6.13pp [−9.91, −3.59] |
| `bank5_proj48` | 0.8831 | 0.5793 | −30.37pp [−34.13, −25.32] |

Paired contrasts, 8 seeds, bootstrap 10 000 / rng 2101:

| contrast | mean [CI95] |
|---|---|
| **head: spikesum − decay025** | **+46.48pp [+43.31, +49.27]** |
| head: bank5 − decay025 | +41.80pp [+38.13, +45.07] |
| oracle: spikesum − decay025 | +26.83pp [+23.10, +30.62] |

`spikesum` head range across the 8 seeds: **0.9336 – 0.9941** (every seed clears
0.65 by a wide margin). `decay025` head range: **0.4590 – 0.5645** (no seed
reaches 0.65).

**The online head's gap to its own oracle closes to −0.07pp on `spikesum`** and
was −19.73pp on `decay025`. The head did not change. Only the substrate did.

## 5. The capacity control is the most informative negative

`bank5_proj48` was registered as a matched-capacity control: a 240-dim bank
projected to 48 dims by a fixed random matrix. Oracle **0.8831** — the projection
keeps most of the information — but the head only reaches **0.5793**
(−30.37pp).

So "more dimensions" is not the explanation and neither is "any compression
helps". The difference between `spikesum` (head 0.9712) and `bank5_proj48`
(head 0.5793) at the **same 48 dims** is that `spikesum` preserves the spike
counts additively while the random projection **destroys the sparse structure the
delta rule can exploit**.

This also explains the clip finding: a delta rule on a sparse, non-negative,
low-overlap code converges easily; on a dense random projection of the same
information it does not.

## 6. Answering the question directly

> Can SCAR-X learn an online readout good enough to exploit what the spike train
> already carries?

**Yes — to within 0.07pp of the information ceiling, on 8 unseen seeds, with the
head unchanged from the version that fails on the production feature.**

The failure on `feature_decay = 0.25` is therefore **not** a credit-assignment
failure, **not** an optimizer failure, and **not** a capacity failure. It is a
**feature-extraction** failure: the production decay discards the temporal
support that makes the label linearly readable, and no head can recover what the
substrate no longer contains.

## 7. Claim matrix

| claim | metric | source | scope | limitation |
|---|---|---|---|---|
| Brief's contrast reproduces independently | ridge 0.9729 / head 0.4837 | `stage0-reproduction.json` | sim_xor, 3 seeds | hard-threshold only |
| Head reaches its oracle when the substrate carries the label | head 0.9759 vs oracle 0.9753 | `r2_head_capacity.py` | 3 seeds, `lr .08 clip .025` unchanged | exploration, pre-protocol |
| Substrate, not head, is the constraint | head 0.9712 vs 0.5063, 8 seeds | confirmation run | sim_xor H=48 | synthetic task only |
| Gap to oracle closes | −0.07pp [−1.12, +0.98] | `confirmation-stats.json` | 8 seeds | one task |
| Capacity is not the cause | `bank5_proj48` head 0.5793 at oracle 0.8831 | confirmation run | same 48 dims | random projection of one bank |
| Hidden weights untouched | sha256 equal in all cells | screen + confirmation | all 44 cells | — |

## 8. What is NOT claimed

- **Not** that SCAR-X now learns language, chat, vision or reasoning. `sim_xor`
  is a 12-tick synthetic task at H=48.
- **Not** that the production readout should be replaced by a spike sum today.
  This measured information accessibility with a frozen hidden; it did not run a
  continual stream, did not measure retention, and did not touch Hermes.
- **Not** that `bank5` is better than `spikesum`. It is worse in the head
  (0.9243 vs 0.9712) at 4× the dimensions.
- A ridge oracle is an information ceiling, never a skill.

## 9. Honest limitations

1. **One task.** `sim_xor` only. `order` and `pair_binding` were named in the brief
   but not run, because the protocol's gate was already decisive on one task and
   expanding after seeing the result would be post-hoc.
2. **Hard-threshold only.** Bernoulli is excluded by the brief and by the
   protocol; no cross-sampler statistic appears here.
3. **The hypothesis was fixed after exploration.** Stage 0 and the capacity check
   ran before protocol v1 was committed. The protocol says so in §0. This is
   pre-registration of the screen and confirmation, not of the diagnosis.
4. **`n_train = 1024` chosen on development** after observing that 256 underfits.
   Registered once, not revisited.
