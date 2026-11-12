# R7 ROUND 14 — REPORT: two retractions, and round 12's conclusion does not survive

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: RETRACT round 12's dead-unit figure and its correlation. The geometry
hypothesis as I stated it is NOT supported.**

---

## 1. The bug

From round 12 onward every script computed "dead units" as:

```python
fire.append(core.spikes.copy())          # spikes at the LAST tick only
ever = fire.reshape(-1, H).max(0)
dead = (ever == 0).sum()
```

`fire` therefore held the spike vector at the **final tick** (t=11, the readout gate), not
the spike train. The contradiction is caught directly in `local/verify_th.py`:

```
th = 1.50:   fire.sum() = 0                  ("no spikes at all")
             F.sum(1) > 0 for 384/384 rows   ("every sequence produced features")
```

Both come from the same forward pass. The only consistent reading is one tick vs twelve.

## 2. Retraction 1 — the dead-unit count

| threshold | TRUE dead (never fire, any tick) | what round 12 reported |
|---:|---:|---:|
| **0.50 (production)** | **1.7 of 48** | **26.3 of 48** |

**Round 12's "25.7 of 48 dead units" is wrong by 24.7 units.** It was counting neurons
silent *at the readout tick*, which is a completely different — and expected — property of
a sparse readout gate.

This also kills round 13's mechanism premise: homeostatic threshold adaptation was
designed to revive 26 dead neurons that do not exist. At production settings only **1.7**
neurons are dead, so there was almost nothing to revive.

## 3. Retraction 2 — the correlation

Round 12's headline was `corr(effective rank, accuracy) = +0.795` across a leak sweep.
Round 13 swept the threshold instead and got `+0.803`. With the corrected spike-train
capture, over 12 (threshold, seed) cells:

| quantity | correlation with accuracy |
|---|---:|
| effective rank | **+0.169** |
| firing rate | **−0.755** |
| dead units (true) | +0.573 |

**Rank's correlation collapses from +0.80 to +0.17 once the analysis is run over a sweep
that separates sparsity from rank.** The leak sweep could not separate them because both
move together; the threshold sweep can, and rank loses.

The full corrected table:

| threshold | firing rate | effective rank | dead (true) | accuracy |
|---:|---:|---:|---:|---:|
| 0.50 | 0.0847 | 4.71 | 1.7 | 0.8320 |
| 0.70 | 0.0567 | 5.13 | 7.3 | 0.8398 |
| **1.00** | **0.0331** | **5.78** | 13.0 | **0.9271** |
| 1.50 | 0.0145 | 5.12 | 22.3 | 0.8997 |

## 4. What survives

1. **Effective rank really is ~4.7 of 48** (9.8%). That measurement used the summed
   feature `F`, which was always computed correctly, and it replicates across seeds and
   thresholds. The population genuinely visits few distinct states.
2. **Sparsity tracks accuracy strongly** (−0.755 with rate): accuracy peaks at
   threshold 1.00 where the rate is 0.033, and collapses at 2.50 where it is ~0.
3. **The production threshold 0.50 is not optimal for parity3** — 0.8320 versus 0.9271 at
   threshold 1.00, **+9.5pp**. This is a diagnostic sweep, and I am **not** changing the
   production threshold.

## 5. What does not survive

- "Feature geometry is the bottleneck, confirmed." **Retracted.** Rank correlates +0.169
  with accuracy in the sweep that can separate the two variables. I reported +0.795 from a
  sweep that could not.
- "25.7 of 48 neurons never fire." **Retracted.** It is 1.7.
- "Homeostatic threshold adaptation targets the measured defect." **Void** — the defect was
  a metric error.

## 6. The honest state of the bottleneck question

After fourteen rounds the measured picture is:

| candidate | status |
|---|---|
| credit assignment | **refuted** (round 11: +0.945 alignment, zero gain) |
| readout capacity | **refuted** (round 6: joint 0.8620) |
| hidden plasticity causes forgetting | **refuted** (round 4: −1.17pp) |
| dead neurons | **refuted** — 1.7 of 48, a metric error |
| effective rank | **NOT SUPPORTED** — +0.169 once separated from sparsity |
| **firing rate / sparsity** | **strongest remaining correlate: −0.755** |
| memory horizon 3 ticks | **established** (round 2c) |
| channel loses 13–21pp | **established** (round 1 probe 5) |

**Sparsity is the strongest remaining predictor and I do not yet know whether it is causal.**
It could be that the network needs sparse codes (information-theoretic), or that the
threshold sweep is a proxy for something else entirely. I am not going to declare it the
bottleneck on a −0.755 correlation after having just retracted a +0.795 one for exactly
that reason.

## 7. Process failure worth naming

This round I also **destroyed 25 scripts** with a bad PowerShell regex replacement
(`$1` in a .NET replace with no capture group), then restored 23 of them from git and
rewrote the 2 new ones. No committed result was lost, because every round was committed
before the next began. That discipline is the only reason this was recoverable.

And this is the **fourth** harness defect in this session that produced a wrong published
number:

| round | defect | what it corrupted |
|---|---|---|
| 2 | `shifted()` zero-padding was a no-op | invalidated a whole confirmation run |
| 4 | forgetting = difference between different tasks | reported 7.68pp for a control that cannot forget |
| 5b | averaged over a structurally-zero term | diluted mean by 33%, retracted |
| **12** | dead units captured one tick | **26.3 vs the true 1.7** |

The pattern is consistent: **the metrics I write under time pressure are less reliable than
the mechanisms I test.** I am now checking every new metric against an independent
computation before reporting it, and I should have been doing that from round 1.

## 8. Message for Gemini

Two things from this round that apply to anyone measuring SCAR-X:

1. **Capture the full spike train, not the last tick.** `core.spikes` after `step()` is one
   tick. If you compute "dead neurons" or "firing rate" from a per-episode loop without
   accumulating the train, you will report ~26 dead units where the true count is ~1.7.
2. **If you have a correlation across a parameter sweep, check that the sweep separates the
   variables.** My `corr(rank, accuracy)` was +0.795 across a `leak` sweep and +0.169 across
   a `threshold` sweep. The first sweep moved rank and sparsity together and could not
   distinguish them; the second could, and my conclusion reversed.

The one robust number I can still offer: **effective rank is ~4.7 of 48** in the frozen
population at production settings, measured from correctly-summed features. That part
replicates.

## 9. Artifacts

`scripts/r7_r14_deadunits.py` · `artifacts/deepseek-r7/dead-units-corrected.json` ·
`local/verify_th.py` (the contradiction that exposed the bug) ·
`report-round12.md` (carries a retraction note) · `report-round13.md`
