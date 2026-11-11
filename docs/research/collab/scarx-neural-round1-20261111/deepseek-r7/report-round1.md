# R7 ROUND 1 — REPORT: three causes located, one mechanism rejected, one benchmark invalidated

**Branch:** `research/deepseek/r7-causal-hidden`
**Base:** `495dab2f4bac0180e4d1cd28674391f055ccf09d` (main)
**Environment:** DSH Web · `deepseek-ai/deepseek-v4.1-flash-1017`
**Production `src/scarx/neural/` MODIFIED:** no — all work is in
`src/scarx/evaluation/r7_secs.py` and `scripts/r7_*`.

**Decision: REJECT (SECS) + REVISE (the benchmark).**

---

## 1. What was asked, and what happened

Task: find why hidden learning produces no useful general representation, propose at
least two competing mechanisms, implement and DEV-test the best one.

I did the diagnosis, implemented mechanism A (SECS), ran it, and **it failed**. More
importantly, the failure exposed that **the benchmark I inherited cannot answer the
question** — `frozen` already reaches 0.827 with a common refit, and no task family
tested has any readout headroom. I report that as the round's main result rather than
tuning SECS until a number moves.

## 2. Three measured causes (probes 0–3, production code instrumented only)

### C1 — the eligibility trace is directionless (dominant)

| measurement | value |
|---|---|
| cosine(trace, dL/dw), trained readout, soft threshold | **−0.0303**, CI95 **[−0.0667, +0.0060]** |
| same for a random vector (the null) | −0.0159 |
| episodes with \|cos\| > 0.2 | 10% |

The CI **contains zero**. The update is `hidden_lr · advantage · (a vector
uncorrelated with the loss gradient)` — scalar-modulated noise.

*My own error, corrected:* the script printed "the trace carries real directional
information" because 0.0303 > 1.5 × 0.0159. Comparing two point estimates whose CIs both
span zero is meaningless. The honest reading is "indistinguishable from noise".

### C2 — the hard threshold zeroes the gradient on 83% of episodes

| condition | mean \|dL\| | zero-gradient episodes |
|---|---:|---:|
| hard threshold | 5.39e-04 | **25/30** |
| soft threshold (β=8) | 8.22e-04 | **0/30** |

**Softening is not free**: ridge on soft features falls to 0.5156 from hard 0.6562. The
hard threshold carries *more information* while admitting *less gradient*. "Use a
surrogate gradient" is not a solution here; it trades away the information.

### C3 — the anchor spring is permanent

`stable_in` / `stable_rec` are **never refreshed** (measured unchanged after 40 updates).
Every update is opposed by a pull back toward the *initial random* weights.

Context: 41 spikes over 12 ticks × 48 neurons (rate 0.071); 28/48 neurons ever fire;
9/12 plastic rows fire, so C1 is not "no activity".

### What this rules out

| candidate | verdict |
|---|---|
| hidden holds no information | **rejected** — frozen ridge 0.6953–0.8346 |
| reward scalar is wrong | **rejected** — advantage correlates −0.80 with loss |
| trace loses causal influence | **CONFIRMED**, cosine ≈ 0 |
| overwrite after learning | partially — anchor spring + noise walk |

## 3. Mechanism A: SECS — implemented, tested, **REJECTED**

Equations (full detail in `hypothesis-report.md`):

```
c_i   = W_out[:, i] · error                     per-neuron credit (local, 2×H matvec)
s_i   = sign(c_i) · min(|c_i| / mean|c|, 3)      normalised, bounded
elig  <- trace_decay·elig + s_i · (spike_i·pre − .5·post_i·x)
Δw    = hidden_lr·advantage·elig − anchor·(w − stable)     [unchanged form]
```

DEV, 3 seeds, same population, common refit head:

| arm | online | refit | \|Δw\| | trace cosine |
|---|---:|---:|---:|---:|
| frozen | 0.4831 | **0.8268** | 0.00000 | −0.105 |
| production (scalar) | 0.4857 | 0.8242 | 0.00088 | −0.131 |
| **secs** | 0.4818 | **0.8216** | 0.00110 | **+0.011** |
| **shuffled (control)** | 0.4792 | 0.8268 | 0.00088 | **+0.076** |
| zero | 0.4831 | 0.8268 | 0.00000 | n/a |

**SECS fails on its own stated purpose.** The gating criterion I registered was
"cosine(trace, dL/dw) must exceed the random-null CI". SECS gives +0.011; the
**shuffled-credit control gives +0.076** — better. Credit shaping did not shape credit.

And no arm beats frozen refit (0.8268). There is nothing to win.

## 4. The benchmark is invalid as a hidden-learning testbed

This is the round's most important finding and it came from the SECS failure.

**Probe 4 — headroom per task** (linear refit vs second-order refit on the *same* frozen
features; headroom is what a more expressive reader could extract):

| task | linear | quadratic | headroom |
|---|---:|---:|---:|
| sim_xor | 0.8346 | 0.7930 | **−0.042** |
| order | 0.9688 | 0.9753 | +0.007 |
| parity3 | 0.6107 | 0.5664 | **−0.044** |

**No task has headroom.** The frozen representation is already linearly sufficient for
every task in the inherited suite. Brief §IX Q7 applies exactly: *"if hidden does not
beat frozen, is the problem the algorithm or has the test already been nearly solved by
frozen?"* — **it has.**

## 5. What is actually broken: the readout channel, not the hidden weights

**Probe 5 — information per readout channel, same frozen population, 3 seeds:**

| task | decay features | spike sum | trajectory | spike-sum − features |
|---|---:|---:|---:|---:|
| sim_xor | 0.8346 | **0.9635** | 0.9883 | **+0.129** |
| order | 0.9688 | 0.7956 | 1.0000 | −0.173 |
| parity3 | 0.6250 | **0.8320** | 0.8932 | **+0.207** |
| delayed_xor | 0.5365 | 0.5404 | 0.5716 | +0.004 |

For `sim_xor` and `parity3`, the production readout's own feature (decay-0.25) throws
away **13–21 percentage points** of information that the *same frozen neurons* already
produced. `order` is the opposite: the decay preserves order information better than the
sum, which is why the production feature was chosen.

**Consequence.** On `sim_xor` and `parity3`, hidden plasticity is being asked to
rearrange weights so that a **decayed marginal summary** carries a conjunction the decay
removes. That is the wrong target: the information is already in the spikes and the
readout channel is what discards it. No local credit rule can fix a channel.

This **reframes cause C1**: the trace is directionless *partly because the loss it is
being graded against is computed on a feature that has already lost the signal*. The
gradient `dL/dw` on 83% of episodes is exactly zero (C2) and near-zero elsewhere because
the decayed feature barely depends on the plastic rows.

## 6. Mechanism B: CTS — proposed, **not run**, and now questionable

Contrastive Trace Selection (select the trace whose sign agrees with the observed
improvement in readout error). Stated in `hypothesis-report.md`.

**I am not running it next in its current form.** Probe 5 says the receiver is broken;
optimising the sender while the receiver discards 13–21pp is wasted effort. Running CTS
now would be spending a DEV budget on a question the evidence already says is
mis-ordered.

## 7. Next decision: REVISE the testbed, then re-ask the learning question

**Round 2 will do, in this order:**

1. **Build a readout channel that does not discard the spikes** (spike-sum or a small
   fixed multi-channel bank), and re-measure headroom. A task is only usable if
   `frozen_linear` is well below the trajectory ceiling.
2. **Construct a relational task with genuine headroom** — one where frozen linear is
   near chance and only a hidden-weight change can help, and verify that by measurement
   *before* training any mechanism.
3. **Then re-run the mechanism comparison** (production / SECS / CTS / controls) on that
   task. Only then is "hidden learning creates a useful representation" a testable
   claim.

Registering this now so it cannot be retrofitted after seeing round-2 numbers.

## 8. Declared false-progress exposures in this round

| risk | exposure |
|---|---|
| **FP2** (ridge better, hidden unchanged) | `weight_delta` reported per arm; frozen has 0.0 by construction |
| **FP7** (optimizer change called architecture) | `hidden_lr`, `trace_decay`, `update_clip` unchanged in every arm |
| **FP4** (one benchmark 100%, related at chance) | protected `sim_xor` arm present; `order` measured separately and shows the opposite channel effect |
| **new: invalid testbed** | no task had headroom; disclosed as the primary result rather than hidden |

## 9. Limitations

- **3 dev seeds.** DEV only; no confirmation was run and none is claimed.
- **`sim_xor` frozen refit 0.8268** is well above R6A's reported ~0.69 for a different
  feature channel; the number is not comparable across channels and I do not compare them.
- **Probe 4's quadratic proxy** is a readout-capacity proxy, not an information ceiling.
  A headroom conclusion drawn from it is about *linear readers*, which is the relevant
  question for an online delta-rule head.
- **SECS was rejected on trace alignment**, its own registered criterion. Accuracy alone
  would have looked like a tie, and I would have been tempted to call that "neutral"
  rather than "failed".

## 10. Message for Gemini

Your CKCER result and my SECS result should be compared on one specific question before
either is believed: **what is `frozen + common refit` on the task you used?**

On the inherited SCAR-X suite I measure frozen refit at **0.8268** on sim_xor, **0.9688**
on order and **0.6107** on parity3, with **zero or negative** headroom from a
second-order reader. If CKCER's task also has frozen near ceiling, then its gain is a
readout effect, not a representation effect, and the comparison arm must be
frozen-plus-same-refit, not frozen-plus-production-head.

The second question: **does your mechanism change `cosine(eligibility, dL/dw)`?** That is
the quantity my probe 3 shows to be at noise (−0.0303, CI [−0.067, +0.006]) in the
production rule. A mechanism that raises accuracy without raising that alignment is not
supplying credit; it is supplying a lucky perturbation, and the shuffled-control will
match it.

## 11. Artifacts

| item | path |
|---|---|
| hypothesis report | `docs/.../deepseek-r7/hypothesis-report.md` |
| probe 0 (three causes) | `scripts/r7_probe0_causes.py` |
| probe 1 (zero gradient) | `scripts/r7_probe1_gradient.py` |
| probe 2 (threshold isolation) | `scripts/r7_probe2_threshold.py` |
| probe 3 (trace alignment) | `scripts/r7_probe3_alignment.py` |
| probe 4 (headroom) | `scripts/r7_probe4_headroom.py` |
| probe 5 (channels) | `scripts/r7_probe5_channels.py` |
| SECS implementation | `src/scarx/evaluation/r7_secs.py` |
| DEV runner | `scripts/r7_dev_secs.py` |
| DEV result | `runs/20261111T114428Z-r7-secs-dev-v1-3a8240/result.json` |
