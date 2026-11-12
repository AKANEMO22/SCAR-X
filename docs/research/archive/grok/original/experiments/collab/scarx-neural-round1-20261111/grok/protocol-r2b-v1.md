# Protocol r2b-v1 — grok — dendritic transfer

Campaign question: does the executed DC-ALIF learn a representation that transfers, or is the gain a task-specific inductive bias?

This protocol is locked before the r2b measurement. It does not change any Round-1 gate or result. It does not add a neuron architecture. The forward, eligibility, sampler and class constants are the code at base `81641f773145f5c441b5ad3b0f5771f645a5cbd6`, plus inference-only diagnostic lesions that are off unless named.

## Executed constants, not the prose constants

The runner constructs `Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0)` and does not override leak or feature decay. The code constants are the class attributes on `DCALIFEligibility`. Stage 0 must record:

- `Config.leak` (default 0.85) against the protocol sentence λ = 0.88
- `Config.feature_decay` (default 0.92)
- class `alpha_s=0.32`, `alpha_d=0.48`, `gamma_q=0.95`, `beta_g=1.35`, `beta_d=1.65`, `d_decay=0.96` against the protocol table that lists other numbers
- hard threshold `u >= theta`. The finite-difference oracle is the anchored smooth surrogate already in `dc_alif_finite_difference_check`. It is not a derivative of the hard spike.
- code `q_t = 0.5 * (I_in^2 - sum_m (u_pair_m)^2)` against the documented all-pairs formula `0.5 * ((sum w pre)^2 - sum (w pre)^2)`. The numerical gap is a finding. The screen still runs the code, not a repaired formula.

If the code's own surrogate finite-difference check fails at `1e-5`, stop. Do not train Stage 2.

## Hypotheses, kept separate

H1. The dendritic gate helps when a later query must select a value written earlier, and it can damage a task that only needs low-order integration.

H2. Part of `correct − frozen` requires the per-neuron signal, but a shuffled signal may keep a large share of the gain.

## Stage 1

Read-only checkpoints from the antigravity-gemini confirmation seed `35201`, tasks `order`, `xor`, `strict_binding`. Held-out data seed `94004`, 96 sequences, not used to train those checkpoints. Lesions, applied only at inference: intact, `drop_coincidence`, `drop_quadratic`, `zero_plateau`, `open_gate`, `shuffle_gate`. Also the direct accuracy of the frozen, shuffled, no_gate and zero checkpoints. A ridge on the plateau at tick 9 and on the final features is a probe of stored information, not a skill score. No hyperparameter is chosen from these numbers.

## Stage 2

One configuration. No per-task branch. No change after scores.

- Model seeds `93011`, `93017`, `93023`.
- Tasks `order`, `xor`, `strict_binding`, `pair_binding`, same data seed per task for every arm.
- Arms: `correct`, `frozen`, `shuffled`, `zero`, `no_gate`, and `baseline_elig` (`ReadoutEligibility`, same `Config`).
- Calibration 128, train 128, one epoch, development 128.
- Before any scored trial, time 16 unscored learns. If that timing projects Stage 2 above 540 seconds, stop and mark `BLOCKED`. Do not reduce the training count after a scored result.
- Data seeds: calibration `93003`, train `93002`, development `93004`.
- Interval: `phase21.interval` (bootstrap 10000, rng 2101).
- Screen budget 600 seconds. If the run would exceed it, stop and mark `BLOCKED`. Do not shrink the budget after seeing accuracy.
- Confirmation is not opened by this protocol unless every line below passes. A pass here is a 3-seed screen, not a confirmation.

Screen passes only if all of these hold:

1. Direct `correct` mean ≥ 0.65 on at least two tasks.
2. On at least two tasks, `correct − frozen` mean ≥ 0.05 and bootstrap lower bound > 0.
3. On those same two tasks, `correct − shuffled` bootstrap lower bound > 0.
4. No task has `correct` mean more than 0.03 below `baseline_elig`.

Otherwise STOP. Do not start another round and do not edit the learner.
