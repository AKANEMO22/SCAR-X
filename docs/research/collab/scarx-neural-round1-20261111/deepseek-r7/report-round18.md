# R7 ROUND 18 — REPORT: the seed gap is real but small, and my diagnosis could not separate the hypotheses

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: INCONCLUSIVE. The three hypotheses were not separated by these measurements,
and I will not report the automatic verdict as a finding.**

---

## 1. Why this round existed

Round 17 left the retention mean replaced by a single seed:

| cell | acq | final | meanF | per-seed meanF |
|---|---:|---:|---:|---|
| `no_replay` | 0.8819 | 0.6693 | 31.90pp | — |
| `buf128_wt1` | 0.8728 | 0.8216 | **7.68pp** | **5.66 / 5.86 / 11.52** |

Seeds 140011 and 140017 fall to ~5.7pp; **seed 140023 sits at 11.52pp** across every
buffer size and replay weight. Tuning further would fit the mean while one third of the
seeds does not move, so this round diagnosed the seed instead.

## 2. Measurements, per seed

| seed | joint ceiling | joint train | seq acq | seq final | meanF | joint−seq gap | head train @ last | between-task \|corr\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 140011 | **0.8867** | 0.9106 | 0.8893 | 0.8515 | **5.66pp** | 0.0352 | 0.9766 | 0.7107 |
| 140017 | 0.8685 | 0.8898 | 0.8620 | 0.8229 | **5.86pp** | 0.0456 | 0.8620 | 0.5847 |
| 140023 | **0.8281** | 0.8550 | 0.8672 | 0.7904 | **11.52pp** | 0.0378 | 0.8516 | 0.6831 |

## 3. The automatic verdict said H1 — and I do not believe it

My script printed "H1 REPRESENTATION SUPPORTED" because `joint(140023) = 0.8281` is
0.0586 below `joint(140011) = 0.8867`, and I had set the threshold at 0.05.

**That threshold was arbitrary and the margin clears it by 0.0086.** Three reasons the
verdict is not trustworthy:

1. **Effect size.** A 5.9pp joint-ceiling spread across three seeds is within the
   seed-to-seed variation I have measured elsewhere in this session on the same tasks.
2. **It does not predict the right thing.** If representation were the cause, the
   *sequential* deficit should track the joint ceiling. It does not: seed 140023's
   joint−sequential gap (0.0378) is *smaller* than seed 140017's (0.0456), yet 140023
   forgets twice as much.
3. **The between-task correlation points the wrong way.** H1 predicts the constrained seed
   has *more* overlapping task means. Seed 140023 is at 0.6831, between 140017 (0.5847,
   the best-forgetting second seed) and 140011 (0.7107, the best-forgetting first seed).
   **The ordering does not match the forgetting ordering at all.**

## 4. H3 was not ruled out either

H3 (optimisation — the head cannot fit its own task) required `train_acc_last > 0.90` to be
ruled out. Values: **0.9766 / 0.8620 / 0.8516**. The worst-forgetting seed has the second
*lowest* train accuracy (0.8516), and the best-forgetting seed has by far the highest
(0.9766).

That is weak but real support for H3: **the seed that forgets most is also the seed whose
head converges least on its own training data.**

## 5. What is actually established, and it is narrower

**One clean fact:** seed 140023's joint ceiling is 0.8281, the lowest of the three, and its
per-task joint scores are uniformly lowest (sim_xor 0.879 / parity3 0.770 / order_short
0.836 versus 140011's 0.887 — i.e. it is a *general* deficit, not task-specific).

**One clean non-fact:** that deficit does not explain its forgetting. A 5.9pp lower ceiling
does not produce 5.7pp more forgetting when the joint−sequential gap is *smaller* for that
seed.

**The honest conclusion: I found a correlate (lower joint ceiling, lower head train
accuracy) but neither accounts for the 11.52pp in the way the hypotheses predicted.**
Reporting "H1 supported" on a 0.0086 margin over an arbitrary threshold would be exactly the
error I retracted in rounds 5b and 12.

## 6. What I will do next instead

The measurement that would actually discriminate is one I did not run: **train the
sequential sequence with a DIFFERENT task order for each seed.** If seed 140023's 11.52pp
follows the *order* rather than the *seed*, it is a task-interaction effect and not a
property of that initialisation at all. That is a cheap, decisive test and it is what
should have been in this round.

I am also going to stop treating "which seed is worst" as the framing. With three seeds and
an 11.5pp / 5.7pp spread, the more useful question is whether the harness's seed-to-seed
variance is itself larger than the gate the mission sets (3pp). If it is, then a 3pp mean
forgetting gate **cannot be measured** with three seeds at all, and that is a fact about the
protocol the mission should see before any more mechanism work.

## 7. Declared exposures

| risk | exposure |
|---|---|
| **arbitrary threshold produced a verdict** | §3 states the 0.05 threshold was arbitrary and the margin is 0.0086 |
| **correlate reported as cause** | §5 explicitly refuses; both correlations point in inconsistent directions |
| **three seeds is too few to resolve a 3pp gate** | raised as a protocol question in §6, not buried |
| **no mechanism this round** | this round is a diagnosis that failed to separate its hypotheses, reported as such |

## 8. Message for Gemini

The check that matters more than any mechanism result at this sample size:

**What is your seed-to-seed spread in mean forgetting, and is it smaller than your gate?**

Mine: three seeds at **5.66 / 5.86 / 11.52pp** under one identical protocol and mechanism.
The spread (5.86pp) is **twice the mission's 3pp mean-forgetting gate.** With three seeds I
cannot distinguish a 3pp result from a 9pp result — the noise floor is above the threshold.

Before either of us claims a retention number, the protocol needs enough seeds that the
gate is resolvable. Otherwise a "PASS" is a draw from a distribution wider than the gate.

## 9. Artifacts

`scripts/r7_r18_seed_diag.py` · `artifacts/deepseek-r7/seed-diagnosis.json` ·
`report-round17.md` (the per-seed table this round followed up)

---

> ## VOID by round 19: the premise was an order artefact
>
> This report diagnosed "seed 140023's 11.52pp" as possibly representation-driven (H1) or
> optimisation-driven (H3), and its script printed "H1 SUPPORTED".
>
> **The premise does not hold.** Round 19 shows the same seed reports **5.08pp to 16.80pp**
> across the six task orders under the identical mechanism. The 11.52pp was largely an
> order effect, not a property of that initialisation.
>
> I did not report H1 as a finding at the time, because the automatic verdict cleared an
> arbitrary 0.05 threshold by only 0.0086. **That caution was correct, and the reason I gave
> for distrusting it was not the real reason.**
>
> Also note this report's own raised question — whether a 3pp gate is resolvable at three
> seeds — was answered by round 19: **it is not** (pooled sd 5.36pp against a 3pp gate).
