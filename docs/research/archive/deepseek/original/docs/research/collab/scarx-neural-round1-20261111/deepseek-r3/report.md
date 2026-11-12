# REPORT — Round 3A — a temporal substrate helps `order`, and nothing generalises

**Branch:** `research/deepseek/r3-temporal-substrate`
**Base:** `7d590f768b36eb9d60e5c9433f1966e4e620cff3`
**Protocol:** SHA-256 `ac0aa85d6c00318e3e5b2dcb831a10ee7d8e5ac6d3d81eb89eed6d472e507532` @ `89738e7`
**Environment:** DeepSeek Harness (DSH) Web, Windows 11
**Model:** `deepseek-ai/deepseek-v4.1-flash-1017`
**Runs:** screen `20261111T035039Z-r3-temporal-screen-v1-c4ff37` (3 dev seeds),
confirmation `20261111T035124Z-r3-temporal-confirmation-v1-7fe1b9` (8 fresh seeds)

**Verdict: STOP.** The screen passed; **confirmation failed.** `temporal_bins_3`
lost its `xor` advantage and regressed on `strict_binding`.

---

## 1. Headline

`temporal_bins_3` is a genuine improvement on **`order`** (**+19.63pp
[+16.21, +22.85]** over the best baseline, 8 seeds) and on **`sim_xor`** (+1.64pp
[+0.27, +3.83], a trivial margin). It **fails to improve `xor`** (+0.34pp
[−3.78, +4.10]) and **regresses `strict_binding`** (−4.71pp [−7.23, −2.12]).

The screen said `xor` improved by +7.4pp. Confirmation said +0.34pp. **Three seeds
did not predict eight.** This is the round's most important methodological finding
about this campaign, not just about this substrate.

## 2. Screen (3 fresh dev seeds) — passed, and it misled

| substrate | dims | sim_xor | xor | order | pair_binding | strict_binding |
|---|---:|---:|---:|---:|---:|---:|
| `decay025` | 48 | 0.4811 | 0.4902 | 0.7956 | 0.5208 | 0.5534 |
| `spikesum` | 48 | 0.9701 | 0.6497 | 0.8385 | 1.0000 | 0.5423 |
| `bank5` | 240 | 0.9362 | 0.5716 | 0.9954 | 0.9993 | 0.5827 |
| `count_plus_fast` | 96 | 0.9753 | 0.6660 | 0.9818 | 1.0000 | 0.5658 |
| **`temporal_bins_3`** | 144 | **0.9850** | **0.7233** | **1.0000** | 1.0000 | 0.5612 |

`temporal_bins_3` passed S1 (4/5 tasks ≥ 0.65), S2 (beats baseline on `xor` +7.4
and `order` +16.2, both time-dependent), S3 (no regression), S5 (deterministic),
S6 (15.2 s, cap 600 s).

**It looked like a clean win. It was not.**

## 3. Confirmation (8 fresh seeds) — failed

| substrate | dims | sim_xor | xor | order | pair_binding | strict_binding |
|---|---:|---:|---:|---:|---:|---:|
| `decay025` | 48 | 0.5061 | 0.4907 | 0.7742 | 0.5085 | 0.5286 |
| `spikesum` | 48 | 0.9783 | 0.6702 | 0.7781 | 0.9998 | 0.5356 |
| `bank5` | 240 | 0.9473 | 0.5833 | 0.9885 | 0.9980 | 0.5271 |
| `count_plus_fast` | 96 | 0.9902 | 0.6487 | 0.9719 | 0.9998 | 0.5356 |
| `temporal_bins_3` | 144 | 0.9946 | 0.6736 | **1.0000** | 1.0000 | **0.5007** |

Paired contrasts vs the better baseline, 8 seeds, bootstrap 10 000 / rng 2101:

| task | `temporal_bins_3` | `count_plus_fast` |
|---|---|---|
| `sim_xor` | +1.64 [+0.27, +3.83] **WIN** | +1.20 [−0.15, +3.34] n.s. |
| `xor` | **+0.34 [−3.78, +4.10] n.s.** | −2.15 [−4.61, +0.07] n.s. |
| `order` | **+19.63 [+16.21, +22.85] WIN** | +16.82 [+13.67, +19.75] WIN |
| `pair_binding` | +0.02 [+0.00, +0.07] n.s. | +0.00 n.s. |
| `strict_binding` | **−4.71 [−7.23, −2.12] loss** | −1.22 [−2.59, −0.12] loss |

Gates: S1 pass, **S2 fail** (one qualifying win, not two), **S3 fail**
(`strict_binding` regression exceeds 3pp). `screen_pass: False`.

## 4. Answering the five brief questions

**Q1. Does one fixed substrate learn across several temporal structures?**
**Partly, and not where it matters.** `temporal_bins_3` reaches ≥0.65 on four of
five tasks and 1.000 on `order`. But its advantage over the baselines exists on
**one** task. `spikesum` at **48 dims** already reaches ≥0.65 on three tasks
(`sim_xor` 0.978, `xor` 0.670, `pair_binding` 1.000). The 144-dim candidate buys
`order`; it does not buy breadth.

**Q2. Does spike-sum lose order information?**
**Yes, specifically on `order`, and not on `xor`.**

| contrast | mean [CI95] |
|---|---|
| `bins3 − spikesum` on `order` | **+22.19pp [+17.11, +27.42]** |
| `bins3 − spikesum` on `xor` | +0.34pp [−3.78, +4.10] |

Spike-sum collapses the window, so it cannot tell which event came first —
`order` falls to 0.778 while `temporal_bins_3` reaches 1.000. But `xor` puts the
two bits on *different* ticks and spike-sum still reaches 0.670 because the
*counts* suffice when the bits are on separate channels. **Order loss is real but
task-specific.**

**Q3. Do multi-timescale features help the online head, or only the offline ridge?**
Help the **head**, and the oracle comparison proves it. Head-minus-oracle gap,
8-seed mean (pp):

| substrate | sim_xor | xor | order | pair_binding | strict_binding |
|---|---:|---:|---:|---:|---:|
| `decay025` | −20.3 | −3.8 | −9.9 | **−37.1** | −2.0 |
| `spikesum` | −0.2 | −3.9 | −1.8 | +0.0 | +0.2 |
| `bank5` | −4.2 | **−13.8** | −1.0 | +0.8 | −4.0 |
| `count_plus_fast` | +0.0 | −8.0 | −1.5 | +0.0 | −2.4 |
| `temporal_bins_3` | +0.4 | −4.5 | +0.0 | +0.2 | −4.0 |

The online head reaches the ridge diagnostic on `sim_xor`, `order` and
`pair_binding`. **`bank5` (240 dims) is 13.8pp *worse* than its own ridge on `xor`**
— more timescales made the online head's job harder, not easier. That is the
clearest available answer: multi-timescale features did **not** help the head
beyond what `spikesum` already achieved, and at 240 dims they actively hurt one
task.

**Q4. Can all of this be done in H=48 without adding neurons?**
**Yes.** Every substrate uses the same frozen 48-neuron population; the extra 96
and 144 numbers are feature channels from bounded streaming state (96 B and
576 B). Hidden weights are sha256-identical before and after every cell. No
neuron was added, and no substrate needed a second population.

**Q5. What deserves an independent streaming test?**
**`spikesum` at 48 dimensions** — not the candidates. It matches the ridge
diagnostic on three of five tasks (gap ≤ 0.2pp), needs only 192 B of state, and
survives every task except the order-3 `strict_binding` that no substrate carries.
`temporal_bins_3` earns a place only if `order` matters enough to justify 3× the
state and a `strict_binding` regression.

## 5. Claim matrix

| claim | metric | source | scope | limitation |
|---|---|---|---|---|
| Temporal bins help `order` | +19.63pp [+16.21, +22.85] | confirmation, 8 seeds | `order_task`, H=48 | **PASS**, single task |
| Temporal bins do not help `xor` | +0.34pp [−3.78, +4.10] | same | `xor_separated` | **FAIL** |
| Temporal bins regress `strict_binding` | −4.71pp [−7.23, −2.12] | same | `strict_binding` | **FAIL**, exceeds S3 |
| Spike-sum loses order information | bins3 − spikesum +22.19pp | same | `order` | real, task-specific |
| `spikesum` matches the ridge diagnostic | head − oracle ≤ 0.2pp on 3 tasks | same | 48 dims | diagnostic only |
| `bank5` hurts the online head on `xor` | head − oracle −13.8pp | same | 240 dims | capacity did not help |
| Everything stays in H=48 | hidden sha256 unchanged, all cells | screen + confirmation | all substrates | — |
| Screen of 3 seeds did not predict 8 | `xor` +7.4pp → +0.34pp | screen vs confirmation | this round | **main methodological finding** |

## 6. What went wrong, honestly

1. **The screen passed on a task that did not replicate.** `xor` showed +7.4pp on
   three seeds and +0.34pp on eight. My screen gate required two qualifying tasks;
   `order` was rock-solid and `xor` was noise. A looser S2 would have caught it; a
   stricter one would have caught it too. What actually failed is that I read a
   two-task pass as evidence when one of the two was within noise on its own.
2. **`strict_binding` regression is a real cost, not a rounding error.** −4.71pp
   with a CI excluding zero. The bin boundaries (4, 8) put the query tick 10 in
   the third bin together with the second write, which plausibly dilutes the
   query signal.
3. **I predicted `strict_binding` would fail for every substrate and it did**
   (head 0.50–0.58 for all five, oracle 0.53). That prediction was right and it is
   the only part of the round that behaved as registered.
4. **Two of my own bugs were caught by the tests, not by me**: `count_plus_fast`
   returned the decayed trace where the batch returned raw counts, and its count
   accumulator was never updated. Both fixed before any capability run.

## 7. STOP declaration

Per protocol §8: **STOP.** Confirmation failed S2 and S3. No further tuning, no
new candidate beyond the registered two, no extension of the seed budget. The
registered conclusion is:

> **`spikesum` (48 dims) is the best general-purpose substrate this round could
> find.** No fixed temporal feature map beat it across task families. One
> candidate beat it on `order` and lost on `strict_binding`.

## 8. Not claimed

No chat, vision, reasoning, retention, or multitask transfer. Each task has its own
readout; this measures **substrate learnability**, not a shared skill. A ridge
diagnostic is a linear reference, **not** an information ceiling. Nothing here
should replace the product readout, and no branch was merged.
