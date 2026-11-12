# Round 2b handoff — grok — dendritic transfer

Code measured: `727f55e` on `research/grok/r2-dendritic-transfer`, base `81641f773145f5c441b5ad3b0f5771f645a5cbd6`.
Protocol committed before the run: `experiments/collab/scarx-neural-round1-20261111/grok/protocol-r2b-v1.md`.
Numbers: `runs/grok-r2b-transfer-v1/result.json`.
Compute 46.5 s, sampled RSS 47,005,696 bytes, no child process. Verdict: **STOP**.

DC-ALIF does not learn a representation that transfers. The coincidence path is necessary for `strict_binding` and it is the same path that holds `pair_binding` at chance and that lets hidden learning damage `order`.

## Stage 0 — what the code actually is

The surrogate finite-difference check passes. Maximum absolute error is 3.0e-9 on `e_f` and 6.7e-8 on `e_d`. That oracle is an anchored smooth surrogate on a fixed presynaptic tape. It is not a derivative of the hard spike `u >= theta`.

Executed values, from the constructor the Round-1 runner used:

| quantity | prose | code that ran |
|---|---:|---:|
| leak λ | 0.88 | **0.85** |
| feature decay | not the G1 value | **0.92** |
| α_s, α_d, γ_q, β_g, β_d, κ_d | 0.35, 0.25, 2.20, 2.00, 1.15, 0.96 | **0.32, 0.48, 0.95, 1.35, 1.65, 0.96** |

`coact` multiplies α_s, α_d and β_g and does not depend on synaptic weights, so it is absent from the synapse derivative. It is also absent from the documented forward equation. Mean absolute gap between the code's `q` and the documented all-pairs `q` is 0.35 (max 1.09) on 32 random inputs. The documented derivative uses `(I − b) − w·pre`. The code has no hidden bias, and its `q` keeps products between channel-pairs after subtracting within-pair sums. Findings `ds-003`, `ds-004` and `ds-005` stay open as documentation errors. They do not break the code's own surrogate check, so the screen was allowed to run the code as it is. The equations were not rewritten.

`zero` matched `frozen` on every Stage-2 seed. Hidden weights moved only on arms that apply a hidden update.

## Stage 1 — one confirmation checkpoint, seed 35201, 96 fresh sequences

This is a causal lesion, not a new accuracy claim. The 8-seed Round-1 table is unchanged.

| task | correct | frozen | shuffled | no_gate |
|---|---:|---:|---:|---:|
| strict_binding | 1.00 | 0.771 | 0.781 | 0.531 |
| xor | 0.979 | 0.875 | 0.958 | 0.896 |
| order | 0.562 | 0.969 | 0.969 | 0.792 |

Lesions on the trained `strict_binding` checkpoint: removing the coincidence current, the quadratic term, or the plateau each drops accuracy to 0.50. Forcing the somatic gate to 1 leaves accuracy at 1.00. Shuffling the gate across neurons leaves 0.865. The same three removals drop the frozen checkpoint from 0.771 to about 0.51, and forcing its gate open leaves 0.771. The architecture gain is the quadratic plateau current. It does not require a learned somatic gate at readout time.

A linear probe of the plateau at tick 9, before the query, scores 0.406 on that trained checkpoint. The plateau at tick 10 scores 0.469. End features score 0.844. The label is not sitting in the plateau as a linear code. The quadratic current changes the spikes the head reads.

On `order`, the frozen checkpoint scores 0.969 and none of the lesions move it. The trained checkpoint scores 0.562. Hidden learning destroyed a code the frozen net already had. Its end-feature probe falls from 0.969 to 0.625.

On `xor`, lesions of the trained checkpoint do not move 0.979, and the shuffled checkpoint already scores 0.958. That gain does not require the coincidence current at inference and does not require the signal to reach the matching neuron.

## Stage 2 — locked screen, 3 seeds, 128 examples, one epoch

Status **STOP**. Confirmation was not opened. No task-specific gate was added.

| task | correct | frozen | shuffled | zero | no_gate | one-compartment |
|---|---:|---:|---:|---:|---:|---:|
| order | 0.586 | 0.747 | 0.779 | 0.747 | 0.630 | 0.875 |
| xor | 0.945 | 0.831 | 0.831 | 0.831 | 0.885 | 0.721 |
| strict_binding | 0.815 | 0.685 | 0.677 | 0.685 | 0.534 | 0.609 |
| pair_binding | 0.500 | 0.500 | 0.500 | 0.500 | 0.810 | 0.966 |

`correct − frozen` is −0.161 on `order` (lower −0.219) and 0 on `pair_binding`. `pair_binding` with the gate is exactly 0.500 on all three seeds while `no_gate` is 0.810 and the one-compartment learner is 0.966. `order` and `pair_binding` are both more than 3 points below the matched one-compartment baseline.

`strict_binding` is the only task where `correct − shuffled` has a positive lower bound (0.102). `xor` gains 0.115 over frozen, but the shuffled arm ties frozen and the lower bound against shuffled is 0. Two tasks clear 65% direct accuracy and two clear the frozen-gain line. The specificity line and the no-regression line do not.

Per-seed rows are in `result.json`.

## How the errors separate

- Architecture: quadratic coincidence current. Required for `strict_binding`. Sufficient to pin `pair_binding` at chance. Not required, at inference, for the frozen `order` solution.
- Readout: end features, not the plateau state. Tick-9 plateau probe on trained `strict_binding` is 0.406.
- Sampler: hard threshold. The eligibility check does not claim to differentiate that step.
- Learning signal: neuron-specific on this screen's `strict_binding` only. On the historical `xor` checkpoint, shuffled already has the gain. `order` learning is actively harmful.

## Answer

DC-ALIF is using a strong inductive bias that fits `strict_binding`. It is not learning a hidden representation that transfers to `order` or `pair_binding`. Round 3 was not opened and the learner was not changed.
