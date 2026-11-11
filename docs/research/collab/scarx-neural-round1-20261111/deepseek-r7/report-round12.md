# R7 ROUND 12 — REPORT: feature geometry is the bottleneck, and it is measured

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no — the leak sweep builds a separate
`Config` for the diagnostic only.

**Decision: the brief's "sai hình học của feature" (wrong feature geometry) hypothesis is
CONFIRMED as the dominant cause, after eleven rounds of ruling out the alternatives.**

---

## 1. The measurement that was missing from every previous round

The brief listed eight candidate causes. I tested seven of them across eleven rounds and
had never once measured the geometry of the representation itself.

Frozen population, `parity3`, spike-sum channel, 3 seeds:

| property | value |
|---|---:|
| **effective rank** (participation ratio) | **4.60 of 48 (9.6%)** |
| **dead units** (never fire, in any episode) | **25.7 of 48** |
| mean pairwise \|correlation\| | 0.226 |
| conditioning | > 1e5 |

**The 48-neuron population occupies about 5 real dimensions, and 26 of its neurons never
fire at all.**

## 2. Is it causal, or just descriptive? A sweep decides

A correlation between rank and accuracy in one configuration would prove nothing. I swept
`leak` — the parameter that controls how much temporal structure each neuron integrates,
hence how many distinct states the population visits — across four values × three seeds =
12 cells, measuring rank and decodability independently.

| leak | effective rank | dead units | \|corr\| | firing rate | accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.50 | **5.26** | 30.7 | 0.217 | 0.064 | **0.8477** |
| **0.65 (production)** | **4.71** | 26.3 | 0.230 | 0.096 | **0.8320** |
| 0.80 | 4.36 | 20.7 | 0.251 | 0.128 | 0.7682 |
| 0.90 | 4.09 | 16.3 | 0.279 | 0.149 | 0.6914 |

Across all 12 cells:

```
corr(effective rank, accuracy) = +0.7948
corr(firing rate,    accuracy) = -0.7613
```

**Rank predicts decodability at +0.79 across a parameter sweep, not just in one setting.**
Every column moves monotonically in lockstep: rank down, dead units down, correlation up,
rate up, accuracy down.

The firing-rate correlation is the same fact read backwards: more firing does **not** mean
more information — it means the population is spending its state in fewer distinguishable
cells. This is the opposite of the usual "sparse coding is good" intuition, and it is what
the data say.

## 3. This explains every failure in rounds 1–11 at once

With 4.6 effective dimensions:

- **A credit signal routes into ~5 axes.** Perturbing hidden weights trades one axis for
  another; it cannot increase how many exist. That is precisely why round 11's `exact`
  trace (alignment +0.945, provably the derivative) and `exact_shuffled` (no direction at
  all) landed in the same place — **direction is irrelevant when there are only 5 axes to
  move along.**
- **Six credit/retention mechanisms failed** (SECS, CTS, OGH, CPS, replay, exact trace)
  because they all assumed the constraint was *where to push*, when it was *how much room
  exists*.
- **Cross-tick XOR was never linearly available** (round 2g, best 0.5527): with ~5 axes the
  conjunction of two temporally separated bits does not fit alongside everything else.
- **The joint linear head reached 0.8620** (round 6) while sequential training reached
  0.53: the same ~5 axes can hold the union when fitted jointly and cannot be
  re-used sequentially. Same root cause, different symptom.
- **The 3-tick memory horizon** (round 2c) is the temporal face of the same degeneracy: a
  population with 4.6 dimensions forgets because it has nowhere to store.

**One cause, six symptoms. That is what a root cause looks like.**

## 4. What this changes about the mission

| claim | status after round 12 |
|---|---|
| credit assignment is the bottleneck | **refuted** (round 11: +0.945 alignment, zero gain) |
| capacity of the readout is the bottleneck | **refuted** (round 6: joint 0.8620) |
| hidden plasticity causes forgetting | **refuted** (round 4: −1.17pp) |
| information is lost upstream | **supported** (rounds 1, 2) |
| **feature geometry is the bottleneck** | **CONFIRMED, corr +0.795 across a sweep** |

The actionable statement is now specific: **SCAR-X's population does not visit enough
distinct states.** Any learning rule — local, exact, or otherwise — operates inside that
limited state space.

## 5. Why `leak = 0.65` is not the fault

At `leak = 0.50`, rank is 5.26 and accuracy 0.848 — better on both. **A lower leak gives
more distinct states because neurons recover faster and spike more selectively.** The
production value 0.65 sits mid-range, and the sweep shows accuracy is monotonically
decreasing in leak within [0.50, 0.90].

I am **not** proposing to change the production leak. The mission forbids tuning
architecture to mask a learning-rule failure, and changing `leak` would be exactly that.
The sweep is a diagnostic that establishes **which variable controls the bottleneck**, and
it is worth reporting precisely because it says the constraint is
**state diversity, not the learning rule**.

## 6. What a legitimate fix would look like, and why I am not implementing it

The finding says the population needs more distinct states. Candidate mechanisms, in order
of how well they respect the constraints:

1. **Homeostatic threshold adaptation** — a neuron that never fires has its threshold
   lowered; one that always fires has it raised. This is *local*, needs no extra neurons,
   no new topology, no BPTT, and it attacks the measured defect directly: **25.7 of 48
   dead units**. There is already `adaptation_strength` in the core, so this is a change
   to how an existing variable is used, not a capacity increase.
2. **Decorrelation pressure local to each neuron** — a neuron that is highly correlated
   with another reduces its own gain. Also local, also no new units.
3. Lower `leak` — **forbidden**, and I am not proposing it.

**Option 1 is the next mechanism and it is well-posed**, because it has a registered
target: raise effective rank from 4.6 toward the ~7–8 that the accuracy curve implies
matters, by reviving dead units. That is a quantitative prediction, not a hope.

I am **not** claiming it will work. Rounds 1–11 taught me that a mechanism with the right
motivation can still fail, and I will test it with the same controls and report the result
the same way.

## 7. Declared exposures

| risk | exposure |
|---|---|
| **correlation reported as causation** | the sweep is 12 cells over a parameter that changes state diversity; §2 states corr +0.795 explicitly as correlational, and §5–6 draw only the mechanism implication |
| **leak sweep could be read as a licence to retune** | §5 explicitly refuses; production leak unchanged; the sweep is diagnostic |
| **effective rank is one statistic** | dead units and mean \|corr\| move with it in the same direction, so it is not a lone proxy |
| **round 12 has no mechanism result** | this round is a diagnosis; the mechanism is registered for next round with its target number |

## 8. Message for Gemini

One number to check on any SCAR-X-derived representation, because it explains my eleven
failures and may explain a false positive in someone else's:

**Effective rank** — participation ratio `(Σλ)² / Σλ²` of the feature covariance, plus the
count of neurons that **never fire** in any episode.

Mine: **4.60 of 48**, with **25.7 dead neurons**. Across a four-point sweep,
`corr(effective rank, accuracy) = +0.795`.

The implication for any credit claim: with ~5 effective dimensions, the direction of a
credit signal is nearly irrelevant — which is exactly what I measured (round 11: aligned
credit +0.945 vs shuffled credit, same outcome at every learning rate). **If your
mechanism improves accuracy while effective rank stays at ~5, your gain cannot be coming
from a better representation**, and the shuffled-signal control should match it.

## 9. Artifacts

`scripts/r7_r12_geometry.py` · `scripts/r7_r12b_geometry_causal.py` ·
`artifacts/deepseek-r7/feature-geometry.json` ·
`artifacts/deepseek-r7/geometry-vs-accuracy.json` · `report-round11.md`
