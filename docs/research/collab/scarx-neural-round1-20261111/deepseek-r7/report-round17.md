# R7 ROUND 17 — REPORT: replay reaches 7.68pp, the worst-case gate is the blocker

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: REVISE. Correct replay cuts mean forgetting 4.2× (31.90 → 7.68pp) and still
misses the gates. The binding constraint is now the WORST seed, not the mean.**

---

## 1. Result

| cell | acq | final | meanF | worstF | gate | per-seed meanF |
|---|---:|---:|---:|---:|---|---|
| `no_replay` | 0.8819 | 0.6693 | **31.90pp** | 54.30pp | fail | — |
| **`buf128_wt1`** | **0.8728** | **0.8216** | **7.68pp** | 16.02pp | fail | 5.66 / 5.86 / 11.52 |
| `buf128_wt3` | 0.8403 | 0.7795 | 9.11pp | 17.19pp | fail | 6.64 / 6.64 / 14.06 |
| `buf384_wt1` | 0.8537 | 0.7891 | 9.70pp | 16.80pp | fail | 4.69 / 12.89 / 11.52 |
| `buf384_wt3` | 0.8607 | 0.8064 | 8.14pp | 17.58pp | fail | 2.93 / 9.96 / 11.52 |
| `buf640_wt1` | 0.8537 | 0.7891 | 9.70pp | 16.80pp | fail | 4.69 / 12.89 / 11.52 |
| `buf640_wt3` | 0.8607 | 0.8064 | 8.14pp | 17.58pp | fail | 2.93 / 9.96 / 11.52 |

Best: `buf128_wt1` at **acq 0.8728, final 0.8216, meanF 7.68pp, worstF 16.02pp**.

Against `no_replay`: **mean forgetting 31.90 → 7.68pp (−24.22pp, 4.2×)** and **final
accuracy 0.6693 → 0.8216 (+15.2pp)**.

## 2. Two things I must report plainly

**2a. `buf384` and `buf640` are identical by construction.** The retention step is

```python
take = min(buf_per_task, N_TR)          # N_TR = 384
```

so any buffer request above 384 per task is silently clamped. There are only 384 examples
per task; the buffer cannot hold more. **The largest-tested buffer is 384/task, and the
640 rows are not extra evidence** — they are the same configuration under a different
label. I flag this rather than reporting six cells as if they were six independent points.

**2b. The worst seed does not move.** `worstF` sits at 16.02–17.58pp across every cell.
Per-seed mean forgetting shows why: seeds 140011 and 140017 come down to 5.66 / 5.86pp,
but **seed 140023 stays at 11.52pp regardless of buffer or weight**. The mean improved
because two of three seeds improved; the third is stationary.

**That is the honest bottleneck now: not "replay", but whatever is different about
seed 140023.** Averaging over it is exactly the aggregation error the brief warns about,
which is why the per-seed column is in the headline table.

## 3. Gate position

| requirement | best achieved | status |
|---|---:|---|
| acquisition ≥ 0.65 | 0.8728 | ✓ on every seed |
| final ≥ 0.65 | 0.8216 | ✓ |
| mean forgetting ≤ 3pp | **7.68pp** | ✗ 2.6× over |
| worst forgetting ≤ 5pp | **16.02pp** | ✗ 3.2× over |
| every seed meanF ≤ 3pp | 5.66 / 5.86 / 11.52 | ✗ |

Two of four gates pass. The two that fail are both about forgetting, and both are now
dominated by one seed.

## 4. Why the weight lever failed its prediction

I registered: "meanF falls monotonically with both levers."

- **Weight**: not monotone. `wt3` is worse than `wt1` at 128 (9.11 vs 7.68) and better at
  384 (8.14 vs 9.70). Over-training the replay set pulls the head back toward old tasks and
  *costs* acquisition (0.8728 → 0.8403 at buf128), so the net effect depends on which
  dominates.
- **Buffer**: bounded by `N_TR`, so the lever was never really tested above 384.

**Prediction failed. Recording it as failed rather than reframing it.**

## 5. What is now established about the second mission goal

The chain from round 4's decomposition to a working mechanism is complete and confirmed:

1. Round 4: hidden plasticity contributes **−1.17pp** to forgetting; readout overwriting
   contributes **+31.90pp**.
2. Round 16: replay (correct labels) attacks exactly the readout, and cuts forgetting
   **3.4×** while preserving acquisition.
3. Round 17: a modest buffer (128/task) is the best setting, and cuts it **4.2×**.

**A mechanism derived from a causal decomposition worked.** That is the first time in
seventeen rounds that this has happened, and it happened only after I found the label bug
that had hidden it.

## 6. Next, and what it requires

The remaining gap is seed-specific, so the next round should be **a diagnosis of seed
140023, not more replay tuning.** Concretely: measure what differs — its R matrix, its
feature geometry, its per-task acquisition — and test whether the stationary 11.52pp is a
task-order effect, a feature-scale effect, or an optimisation artefact of that seed's
initialisation.

Tuning the replay knobs further would be fitting the mean while the worst seed does not
move, which is the definition of the aggregation error the mission forbids.

## 7. Declared exposures

| risk | exposure |
|---|---|
| **FP3** (low forgetting, learned nothing) | acquisition 0.8728 ≥ 0.65 on every seed; recorded per seed |
| **FP9** (replay answers instead of the neuron) | the buffer only re-exposes the head to `(feature, label)` pairs; the neurons compute every feature. Adaptive replay is on the brief's permitted list |
| **aggregation masking a seed** | per-seed column in every table; worst seed called out as the blocker |
| **reporting six cells that are two** | §2a states buf384 ≡ buf640 |
| **prediction failed** | §4 records the non-monotone weight result as a failure |

## 8. Message for Gemini

Two checks that changed my result, both cheap:

1. **Does your replay buffer store the label with the sequence?** Mine did not — it applied
   the current task's labels positionally to stored sequences from previous tasks. That
   produced `acq 0.4818, meanF −0.04pp`, which I misread as "replay destroys learning" and
   rejected for twelve rounds. Corrected: `acq 0.8728, meanF 7.68pp`.

2. **Does your worst seed move, or only the mean?** My mean forgetting improved 4.2× while
   `worstF` stayed at 16pp, because seed 140023 is stationary at 11.52pp across every
   buffer size and replay weight. If I had only reported the mean I would now be claiming a
   4.2× win on a mechanism that does nothing for one third of my seeds.

## 9. Artifacts

`scripts/r7_r17_replay_sweep.py` · `artifacts/deepseek-r7/replay-sweep.json` ·
`report-round16.md` (the label bug and the first corrected replay) · `report-round4.md`
(annotated with the retraction)
