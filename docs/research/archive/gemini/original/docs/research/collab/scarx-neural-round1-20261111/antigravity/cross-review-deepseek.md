# Cross-review of `deepseek` â€” `pair_binding`

Reviewer: antigravity. Environment: Antigravity (Windows). Model: Claude Sonnet 5.5 (High) for this review; the branch itself was authored under Gemini 4 Argon (High).
BASE_SHA: `1735ad66ea9e7b416f5df15df99d9c13cea46b4e`
Reviewed: `research/deepseek/scarx-neural-round1-20261111` @ `72aca4e` (the report was written against `bd74d98`; HEAD adds the corrected-claim commit). Read-only: bytecode disabled, nothing written in that worktree.
Reproduction driver: `scripts/review/repro_deepseek.py` (imports deepseek's `pair_binding`, `PlasticCore`).

## Independence note

The deepseek report discloses that an earlier session produced both it and `research/antigravity/scarx-neural-round1-20261111` (`3e5d85f`). My branch `research/antigravity-gemini/...` is a separate Gemini-authored implementation with a different mechanism (DC-ALIF), different seeds and code. I read the earlier same-session branch only as a peer artifact. I am not reviewing my own work here: the author of the reviewed branch is the deepseek-labelled session, not the model that wrote DC-ALIF. I still weight its independence from `antigravity/â€¦3e5d85f` at zero, as it asks.

## Reproduction

| Check | Reported | Reproduced | Match |
|---|---|---|---|
| `strict_binding` ridge: order-1 / order-2 / order-3 (seed 31001) | â‰ˆ0.47â€“0.52 / 0.48â€“0.52 / 1.00 | 0.465 / 0.4775 / **1.0000** | yes |
| `pair_binding`: order-1 / order-2 | â‰¤0.56 / 1.00 | 0.480 / **1.0000** | yes |
| Negative control, seed 41001, `readout_lr` 0.025, direct dev accuracy `score` vs `frozen` | 0.48828 vs 0.5000 | 0.48828 vs 0.5000 | exact |
| Checkpoint â†’ cold process â†’ next learning step, all 23 arrays | pass | SHA-256 of every array identical | pass |
| P0 receipt | 14/14 | not re-run end to end; 4 of its checks reproduced above independently | partial |

Positive finding (the information is only reachable at order-3 for `strict_binding`) reproduced. Negative control (no learning gain on `pair_binding`) reproduced. Cold resume reproduced.

## Assessment

1. **The `strict_binding` diagnosis is right and independently confirmed by my branch.** An additive point neuron plus linear head is order-1 in its inputs; the label is order-3. My own analysis reaches the same conclusion by a different route (phantom cross-tick coincidence ceiling of 5/8 = 62.5% for a 1-compartment linear voltage) and the repository BPTT control's 63.54% sits on that ceiling. I count this as two independent agreements, not three.
2. **The registered C1 was falsified honestly and the screen was a correct NO-GO.** `score âˆ’ frozen = âˆ’2.47 pp [âˆ’2.73, âˆ’1.95]`, confirmation never opened. Process is exemplary.
3. **Concern with the corrected Â§4 claim.** The report says the task "needs no product term" because a uniform sum of the *spike train* over ticks scores 0.752. That sum is not an order-1 function of the *input*: the raw tape reaches only 0.528 linearly (their own table), and spiking, reset and the adaptation threshold are nonlinear in the inputs. So 0.752 shows a long-memory readout of ALIF output extracts more than the raw linear tape, not that no multiplicative structure was used. The two statements "order-1 in the raw tape: 0.53" and "order-1 in spikes: 0.75" must not be read as the same order. The co-occurrence gap (0.990 vs 0.752 at matched width) still shows a product basis adds +23.6 pp. The headline "memory, not nonlinearity" is therefore too strong; "memory is necessary, product basis is additionally useful" is what the numbers support.
4. **Post-hoc status.** Â§4 was derived after the screen failed, which the report states. It is a hypothesis, not a result.
5. **Generator defect handling.** The discarded first generator (independent value draws â†’ bag shortcut 0.70â€“0.77) was caught by P0 and documented; that is the right practice.
6. **Docstring contradicts code.** `pair_binding.py` states `label = value[key=0] XOR value[key=1]`, but the code forces the two values to differ, so that XOR would be constant 1; the implemented label is `value[key=0]` (as the function's own docstring says later). Harmless to results, misleading to readers.
7. **Independent cross-check from my branch.** I ran my DC-ALIF (locked hyperparameters, 8 new seeds, post-hoc) on this generator: gated mechanism 50.5%, its own linear `no_gate` ablation 78.4%, frozen 50.0%. That agrees with the report that long memory solves `pair_binding` and that a gate fitted to `strict_binding` does not transfer. Details in my `report.md` Â§5.
8. **Interval method.** Three seeds with a bootstrap CI have very low resolution (the interval collapses toward min/max of three gains); the âˆ’2.47 pp CI is therefore narrow by construction. This does not change the NO-GO.

## What I would not accept

- "Every readout in the campaign fails because it lacks a product term" (withdrawn by the author).
- "The neuron already carries the signal" in the report title, if read as a learned capability; it is a probe of untrained spikes.

## Verdict on the claims

| Claim | Reviewer status |
|---|---|
| `strict_binding` is order-3 â†’ invalid gate for additive point neuron | supported, reproduced |
| `pair_binding` (order-2) solvable by oracle product basis | supported, reproduced |
| `pair_binding` learnable by hidden learning in this architecture | not supported (reproduced negative control) |
| The task needs no product term | not supported as phrased (see 3) |
| Temporal memory is the bottleneck | partially supported; necessary, not sufficient |

No blocking technical defect found.
