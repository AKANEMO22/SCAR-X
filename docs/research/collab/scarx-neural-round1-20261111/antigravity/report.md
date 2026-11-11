# antigravity report — DC-ALIF (dendritic coincidence-gated ALIF), round 1

Campaign `scarx-neural-round1-20261111` · BASE_SHA `1735ad66ea9e7b416f5df15df99d9c13cea46b4e`
Environment: Antigravity (Windows, CPU, Python 3.11.9). Models: the prototype, protocol and formal run were produced under **Gemini 4 Argon (High)**; the peer cross-review and this report's final analysis were done under **Claude Sonnet 5.5 (High)**. `ponytail` is not installed, so it was not used.
Protocol commit `4919cf880e0c48792bfa57d96ed22ffd78cbf650` (SHA-256 `c16fe28b…ca61`), plus [Amendment 1](../../../../../experiments/collab/scarx-neural-round1-20261111/antigravity/protocol-v1-amendment-1.md) (post-hoc).
Formal run: `runs/20261111T005708Z-collab-antigravity-dc-alif-v1-81d4b2` (`result.json`, `manifest.json`, `source.zip`). Interrupted earlier run `…-934cff` is preserved and marked superseded (memory cap, orchestration only).

## Verdict

**PIVOT / scoped, with a deviation.** DC-ALIF solves the order-3 `strict_binding` task that the 1-compartment baseline cannot (98.78% vs 63.41%) and the dendritic gate is what does it. It does **not** improve learning in general: it makes `order` much worse than the baseline, it fails on a different binding task (`pair_binding`, chance level), and it fails the P2 retention gate. The executed run also departs from the registered protocol (Amendment 1), so none of this is a protocol-v1 pass. No chat, vision, reasoning or retention claim is made.

## 1. Hypothesis and mechanism (as registered)

Why a 1-compartment ALIF fails `strict_binding` (label = value[query], needs key × value × query): the voltage is linear in the inputs, so subthreshold coincidence of the two writes and the query is an order-1 function; spike reset removes the stored write. Report 35's own BPTT control also stalled at 63.54% on this task, so the ceiling is the function class, not the credit rule.

DC-ALIF adds, inside each of the 48 ALIF neurons, a slow non-resetting dendritic trace `d_j` driven by zero-diagonal cross-synaptic coincidence, and a bilinear dendritic–somatic gate that multiplies `d_j` by the somatic query drive. Credit uses the approved one-step readout error `W_outᵀ·error` times forward eligibility computed at each neuron's own synapses (`e_f, e_v, e_a, e_d`). No hidden→hidden error transport, no backward-in-time pass, no autograd in the candidate. `bptt_control` is an isolated control and its weights are never imported.
State cost: `+192 B` neuron state, `+3,072 B` eligibility; candidate arrays 33,954 B (P1), 66,036 B (P2). Disclosure (required): `W_outᵀ·error` is not biologically local.

## 2. P0 engineering — passed

27 tests pass (`tests/test_dc_alif_p0.py`, `test_readout_eligibility.py`, `test_readout_usefulness_p0.py`; re-run 2026-11-11, 6.9 s). Not the full repository suite.
In the formal run: anchored-surrogate finite-difference check over 64 synapses per task (4 neurons × 16), max abs error `e_f` ≤ 8.1e-9, `e_d` ≤ 1.3e-8, forward state exact; zero == frozen bit-for-bit; matched signal RNG; mid-segment resume exact; fault-injection rollback exact; foreign loaders rejected. The oracle is an anchored smooth own-neuron surrogate, not a derivative of hard spikes. All 54 + 144 P1 checkpoints and the 9 P2 checkpoints resumed exactly in a fresh process.

## 3. P1 results (executed harness; direct accuracy, mean [95% CI], n = 8 confirmation seeds)

| Task | correct | frozen | shuffled | no_gate | BPTT control | correct − frozen (direct) |
|---|---:|---:|---:|---:|---:|---:|
| `strict_binding` | **98.78** [98.61, 98.93] | 75.22 | 85.57 | 62.43 | 98.85 | **+23.56** [18.87, 29.27] |
| `xor` | 98.14 [98.00, 98.32] | 85.84 | 91.82 | 91.19 | 97.83 | +12.30 [7.57, 17.87] |
| `order` | 64.04 [57.98, 71.68] | 82.52 | 82.84 | 69.56 | 89.36 | **−18.48** [−32.28, −1.61] |

Probe gain (refit ridge head, unseen data): `strict_binding` +24.19 [19.12, 30.05]; `xor` +12.33 [7.81, 17.65]; `order` −8.28 [−18.09, 3.74]. `zero` equals `frozen` bit-for-bit. State-reset probe ≈ 0.50 on all tasks, so the representation depends on carried state. Bag baseline ≈ 0.48–0.49.

Reading:
- `strict_binding`: the effect is large and consistent across 8 seeds and all controls (zero, frozen, shuffled, no_gate). `no_gate` falls to the old baseline (≈62%), which is the direct evidence that the bilinear dendritic gate, not extra state, carries the result.
- `shuffled` still reaches 85.6%: about 10 points of the gain over frozen is not from directing the signal to the right neurons.
- Frozen DC-ALIF already scores 75.2% (above the 65% gate): part of the benefit is architecture, not learning. The learning contribution is the +23.6.
- `order`: **fails.** Learning degrades the hidden layer below frozen. The development screen (3 seeds) did not predict this; its CI on correct−frozen already spanned −10.9..+23.4. Frozen order has a wide seed spread (69.95–93.46%).
- Regression against the earlier baseline: Report 35 (same harness, dev seeds, 1-compartment `ReadoutEligibility`) reports `order` correct 98.44% / frozen 94.92%. DC-ALIF is far below that on the same task. Report 35 `xor` was 91.80% (dev), here 98.05% (dev). These baseline numbers are from a committed earlier run, not a same-run control.
- BPTT control reaches 98.85% on `strict_binding`: the architecture can express the solution; the local one-step credit finds essentially the same performance there.

Gates measured on `strict_binding` from the executed data, with the repository interval rather than the registered bootstrap (Amendment 1): correct−frozen ≥ 5 pp with CI lower > 0 ✔; correct−zero ✔ (identical to frozen); correct−shuffled direct +13.2 pp, probe +12.2 [8.8, 15.4] ✔; correct−no_gate +36.4 [32.0, 40.9] ✔; probe gain ≥ 5 pp with CI lower > 0 ✔. Registered but unmeasured: `delayed_cue`, `order_sensitive_vision`, same-run `baseline_elig`.

## 4. P2 — failed

`mixed_stream`, 20 contexts, 3 seeds (`15217, 15227, 15241`), acquisition / final / mean forgetting / worst forgetting:

| Condition | Acq | Final | Mean forgetting | Worst |
|---|---:|---:|---:|---:|
| DC-ALIF + usefulness (β = 4) | 71.55 | 56.13 | 21.89 | 54.95 |
| DC-ALIF unprotected | 77.64 | 56.80 | 25.25 | 60.16 |
| DC-ALIF frozen reservoir | 30.83 | 26.11 | 10.25 | 34.38 |

Gate (final ≥ 65%, mean forgetting ≤ 3 pp, worst ≤ 5 pp): **fails** by a large margin on every seed. Usefulness protection changes final accuracy by −0.7 pp and mean forgetting by −3.4 pp: small. Hidden-only ablation drops are ≈ 0 (−0.6 pp), head-only drops 1.8 pp, joint refit probe 48.4%, so the evidence that hidden representations carry the skill on this stream is weak. The frozen-reservoir arm's low forgetting is because it learned little (acquisition 30.8%); it is not retention.
Cross-run context only: Report 21 reservoir reported final 34.74% / forgetting 29.09 pp on the same stream family; DC-ALIF is higher (56%) but this is a different run, and 22 points is not the gate.

## 5. Post-hoc out-of-design check — negative

`scripts/explore_dc_alif_pair_binding.py`, locked hyperparameters, `deepseek`'s `pair_binding` (order-2 pairing, constant bag), 8 seeds `36011..36061`, 512 test episodes:

| correct | frozen | shuffled | no_gate |
|---:|---:|---:|---:|
| 50.51 | 50.00 | 50.02 | **78.39** |

correct − no_gate = −27.9 pp [−34.4, −22.7]. On this task the dendritic coincidence gate **destroys** what the linear slow dendrite (`no_gate`) captures (78%, consistent with `deepseek`'s finding that long memory suffices for `pair_binding`). So the DC mechanism is a task-specific inductive bias fitted to the `strict_binding` structure, not a general binding mechanism. I have not diagnosed why the gated path drowns the signal; that would need its own registered test. This run is post-hoc and not used to select anything.

## 6. Resources

Formal run wall-clock 1,288 s (< 3 h). Peak process-tree RSS 608,903,168 B, peak tree commit 1,080,418,304 B (BPTT children included, one at a time), per-process commit ≤ ~78 MB for NumPy trials; all < 2,000,000,000 B. No iGPU, energy or API cost. Windows per-process cap enforced.

## 7. Claim matrix

| Claim | Metric | Source | Scope | Limit / status |
|---|---|---|---|---|
| Gated dendritic coincidence enables order-3 conjunction | `strict_binding` 98.78 vs no_gate 62.43, frozen 75.22 | formal run, 8 seeds | 12-tick, 8-channel, H = 48, synthetic | supported under deviation; task-specific |
| Own-neuron forward eligibility + one-step readout error trains it | correct − frozen +23.6 pp; zero == frozen | same | same | supported on `strict_binding`, `xor`; 10 pp of gain survives shuffling |
| Matches isolated BPTT | 98.78 vs 98.85 | same | same | supported on `strict_binding` |
| Improves learning in general | `order` −18.5 pp; `pair_binding` 50.5 | same; post-hoc | — | **not supported** |
| Retention at 20 contexts | final 56.1, forgetting 21.9 | P2 | `mixed_stream`, 3 seeds | **failed gate** |
| protocol-v1 satisfied | — | Amendment 1 | — | **no** |
| Chat / vision / reasoning | not measured | — | — | no claim |

## 8. What would change my mind / next step

Not executed and not claimed: (a) diagnose why the gated path fails on `pair_binding` and `order` (which of the gate, the quadratic term or `κ_d` dominates) and whether a gate learned or scheduled per task avoids the regression; (b) run the registered `delayed_cue` / `baseline_elig` arms; (c) a P2 test with the registered stream. Any of these is a new registration, not a retune of this run (Report 35's STOP family is not reopened: the mechanism here is a within-neuron product path, not a learning-rate or epoch change).
