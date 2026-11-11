# REPORT — Round 6A — the bottleneck is data, not updates and not representation

**Branch:** `research/deepseek/r6-learning-scaling`
**Base:** `495dab2f4bac0180e4d1cd28674391f055ccf09d`
**Environment:** DeepSeek Harness (DSH) Web, Windows 11
**Model:** `deepseek-ai/deepseek-v4.1-flash-1017`
**Protocol:** `protocol-v1.md`, committed **after** the exploratory screen — stated in
its §0 and repeated here.

**Verdict: GO on the primary scientific question, STOP as a gated confirmation**
(G6 fails on one seed).

---

## 1. The answer, up front

> *Does SCAR-X need more unique experience, more updates, or a different learnable
> representation? Is there a reproducible delayed improvement?*

**More unique experience. Not more updates. Not a different representation.**

| factor | verdict | evidence |
|---|---|---|
| **data** | **binding constraint** | unique-16384 beats repeated-256 by **+13.25pp [+6.99, +19.40]** at *identical* 16 384 updates |
| **updates** | **saturates, then hurts** | epoch scan peaks at 4 epochs (0.8318), falls to 0.8036 at 16 |
| **representation** | **not the limit** | ridge gap closes from **0.1152 → 0.0228** as data grows |
| **delayed transition** | **not supported** | per-curve second differences land at chance (18/36 = 0.50) |

**The 0.6676 confirmation failure in R5B was data-limited.** At 16 384 unique examples
the same frozen network and the same local delta rule reach **both_A = 0.9237** on 8
untouched seeds, against R5B's 0.6676 at 1 024 examples.

## 2. The scaling ladder — confirmation, 8 untouched seeds

| n_unique | direct | both_A | both_B | needB | ridge | **gap** |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.8425 | 0.5407 | 0.9887 | 0.6915 | 0.9578 | **0.1152** |
| 1 024 | 0.9138 | 0.7176 | 0.9904 | 0.8170 | 0.9924 | **0.0786** |
| 4 096 | 0.9532 | 0.8445 | 0.9903 | 0.9221 | 0.9958 | **0.0426** |
| **16 384** | **0.9736** | **0.9237** | **0.9980** | **0.9557** | **0.9965** | **0.0228** |

`both_A` min per seed at 16 384: **0.7984** — every seed clears the 0.75 gate.

**The ridge column is the mechanistic key.** Ridge is already at 0.9578 when the head
is at 0.5407 (gap 0.1152). As data grows the head catches up to ridge while ridge
barely moves. That is **optimization-under-data-scarcity**, not representational
insufficiency: the information was always there, and the local rule needed enough
distinct episodes to find it.

## 3. The matched-update contrast — the decisive control

All four arms perform **exactly 16 384 updates**:

| arm | n_unique | epochs | both_A | direct | train | clip |
|---|---:|---:|---:|---:|---:|---:|
| repeated-256 | 256 | 64 | **0.7920** | 0.9354 | **1.0000** | low |
| repeated-1024 | 1 024 | 16 | 0.9014 | 0.9634 | 0.9816 | |
| repeated-4096 | 4 096 | 4 | 0.8948 | 0.9706 | 0.9758 | |
| **unique-16384** | 16 384 | 1 | **0.9244** | 0.9771 | 0.9762 | |

**`unique − repeated = +13.25pp [+6.99, +19.40]`.**

This is the finding the brief demanded. **16 384 updates over 256 examples do not buy
what 16 384 distinct examples buy.** Note the repeated-256 arm reaches **train
accuracy 1.0000** and still scores 0.7920 on `both_A` — it has memorised 256 episodes
perfectly and cannot generalise the relational structure. Repetition is not learning.

## 4. Updates alone: saturate, then degrade

At fixed `n = 1024`, varying epochs only (screen, 3 seeds):

| epochs | updates | direct | train | both_A | gap |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 024 | 0.9268 | 0.9541 | 0.7338 | 0.0664 |
| 2 | 2 048 | 0.9437 | 0.9648 | 0.7945 | 0.0495 |
| **4** | 4 096 | **0.9548** | 0.9775 | **0.8318** | 0.0384 |
| 8 | 8 192 | 0.9411 | 0.9574 | 0.8269 | 0.0521 |
| 16 | 16 384 | 0.9398 | 0.9473 | **0.8036** | 0.0534 |

**H2 is refuted as a growth mechanism.** More updates help up to ~4 epochs and then
degrade, while the training fit also degrades (0.9775 → 0.9473). That is not
overfitting in the usual sense; it is the delta rule drifting under a saturating
`clip` once the useful signal is exhausted.

## 5. No delayed transition

The fine sweep (7 budgets × 3 seeds × 2 orders) produced:

```
first differences : [0.086, 0.052, 0.136, 0.088, 0.043, 0.080, 0.023]
second differences: [-0.035, +0.085, -0.049, -0.045, +0.037, -0.057]
```

An aggregated curve with two interior accelerations — which my first check read as a
transition. **It is not.** The four tests I registered:

| test | result | verdict |
|---|---|---|
| interior acceleration | 2 present | passes |
| saturating law misfits | rmse 0.0339, max residual 0.0739 | passes |
| **per-curve d2 not at chance** | **18/36 = 0.500** | **fails — exactly chance** |
| **wiggle > seed sd** | |d2| 0.0512 < sd 0.0631 | **fails** |

All **6/6** per-seed curves contain a dip. `transition_claimed: False`.

**H4 is refuted.** The curve is ordinary saturating growth; the apparent jump was
between-seed noise aggregated into a smooth-looking bump.

## 6. Held-out stress — where the round fails its own gate

| condition | mean | min seed | ceiling |
|---|---:|---:|---:|
| temporal shift (+1 tick, no retrain) | 0.8256 | **0.5557** | 0.7419 |
| distractors (no retrain) | 0.7726 | **0.5713** | 0.7419 |

Both means exceed the first-key ceiling, but **seed 121017 falls to 0.5557 and 0.5713**
— below chance-plus-shortcut. G6 as I registered it ("every stress row above the
ceiling") therefore **fails**, and `confirmation_pass: False`.

I am not relaxing the gate. The honest reading: the data-scaling result is robust in
aggregate and in every seed at clean evaluation, but **temporal robustness is
seed-fragile**, colliding with the same fragility R4A found.

## 7. Null controls (screen, 3 seeds at n=16384)

| control | value | reading |
|---|---:|---|
| reference direct | 0.9756 | |
| **random training labels** | **0.5251** | no leakage — labels must be real |
| first-key shortcut | 0.7419 | the 0.75 ceiling holds |
| query-only shortcut | 0.4987 | the query alone is useless |

## 8. Hidden plasticity (H5) — **BLOCKED**, not fabricated

The brief permits one secondary arm using the existing local hidden rule. I did not run
it. The repository's hidden update lives behind `PlasticCore`/`UnifiedCore` in
`src/scarx/neural/`, which the brief forbids modifying, and reproducing that rule
faithfully inside research-only code would require reimplementing eligibility traces
and their state lifecycle. R5B's `sampler-learning-comparison` already showed the
hidden rule does not help on a related task, but **that is not this test** and I will
not present it as one.

**H5 is untested. BLOCKED, with the reason stated.**

## 9. What this does and does not settle

**Settled:** on this generator at H=48 with a frozen hidden, the R5B figure was limited
by **number of distinct episodes**. Data diversity is causally necessary
(+13.25pp at matched updates), update count saturates and then harms, and the ridge gap
closing to 0.0228 shows the representation was never the constraint.

**Not settled:** whether more data helps *other* tasks; whether hidden plasticity would
beat this ceiling (untested); whether the seed-121017 fragility is a population property
or a harness interaction. And nothing here is a language, vision or reasoning claim.

## 10. Answer to the final question, in one paragraph

SCAR-X needs **more unique experience**, not more updates and not a different
learnable representation, to overcome this relational-retrieval bottleneck. There is
**no reproducible delayed improvement**: the learning curve is ordinary saturating
growth whose apparent internal bump is seed noise (per-curve second differences land at
exactly chance, 18/36). The precise mechanism is that the frozen ALIF features were
always linearly decodable (ridge 0.9578 at n=256) while the local delta rule needed
enough *distinct* episodes to climb from 0.5407 to 0.9237; repetition to the same update
budget produces perfect training fit (1.0000) and no transfer (+13.25pp deficit).

## 11. Limitations

1. **Protocol committed after the screen.** Stated in §0. Only the 8 confirmation seeds
   count as preregistered.
2. **Budgets are not nested.** `independent_query_set_v1` reshuffles at the end, so
   `count=256` is not a prefix of `count=16384` (measured overlap 21/256). Each budget
   is an independent draw; the protocol asked for nesting "where possible" and it is
   not possible without changing the generator, which is forbidden.
3. **One generator, one architecture.** No second relational task.
4. **H5 untested.** 
5. **Seed-121017 stress fragility** unexplained.
