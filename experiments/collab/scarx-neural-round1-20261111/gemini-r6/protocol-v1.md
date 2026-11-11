# Pre-Registered Protocol v1 — SCAR-X Round 6C (`gemini-r6`)
## Context-Keyed Competitive Eligibility Readout (`CKCER`) for Local Associative Query Retrieval

- **Campaign**: `scarx-neural-round1-20261111` (Round 6C — First-Principles Local Associative Learning and Robust Query Retrieval)
- **Agent / Environment / Model**: `antigravity` / `Antigravity` (Windows 11, CPU-first, Python 3.11.9) / `Gemini 4 Argon (High)`
- **Branch**: `research/gemini/r6-local-associative-learning`
- **Base Commit (`main`)**: `495dab2f4bac0180e4d1cd28674391f055ccf09d`
- **Historical Baselines Referenced**:
  - Grok R5A: `b0480c9cc02a3f35ba994480df2a7d984b6f89f1`
  - DeepSeek R5B: `d6c609df6f90fc8df96fb3de424dd54388f8dc59`
- **Protocol Status**: **LOCKED PRIOR TO ANY SCORED DEV OR CONFIRMATION RUNS**

---

## 1. Scope, Permissions, and Non-Negotiable Constraints

1. **Research-Only Evaluation Scope**:
   - All Round 6C code lives in `src/scarx/evaluation/r6_associative.py`, `scripts/r6_*.py`, and `tests/test_r6_associative.py`.
   - Zero modifications to `src/scarx/neural/core.py`, `src/scarx/neural/unified.py`, Hermes integration, or any serving checkpoint.
   - No merge to `main`, no P2 continual-learning claims, and no revival of DC-ALIF.
2. **Core Architecture Invariants**:
   - Single recurrent ALIF core (`SpikingCore`, $H = 48$ hidden neurons, hard binary spikes $z_j(t) \in \{0,1\}$, adaptive threshold $a_j(t)$ with $\gamma_a = 0.95, \beta = 0.08$, membrane leak $\lambda = 0.90$).
   - CPU-first execution (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).
   - Process-tree committed memory $\le 2,000,000,000$ bytes (`2.0 GB`).
   - No pretrained Transformer/LLM, no BPTT through the recurrent core, no external answer database or lookup table, and no human-coded key-to-value output mapping.

---

## 2. Locked Candidate Mechanism (`CKCER`)

### 2.1 Pre-Query Dual-Timescale Eligibility State
At each tick $t \in \{0, \dots, t_q - 1\}$, each hidden neuron $j \in \{1, \dots, 48\}$ maintains two bounded local spike traces:
1. **Cumulative Spike-Count Eligibility Trace** ($\gamma_c = 1.0$, 0-th temporal moment):
   $$c_j(t) = c_j(t - 1) + z_j(t), \quad c_j(-1) = 0$$
2. **ALIF Adaptation Trace** ($\gamma_a = 0.95$, native to `SpikingCore.adaptation`, encoding the 1-st temporal moment via $c_j(t) - a_j(t) \approx 0.05 \sum_{s=0}^t (t - s) z_j(s)$):
   $$a_j(t) = 0.95\, a_j(t - 1) + z_j(t), \quad a_j(-1) = 0$$

At query onset $t_q$, the pre-query eligibility state is latched:
$$m(t_q - 1) = \begin{bmatrix} c(t_q - 1) \\ a(t_q - 1) \end{bmatrix} \in \mathbb{R}^{96}$$

### 2.2 Pre-Gating Variance-Safe Homeostatic Standardization
During the unsupervised observation phase (`obs_steps = 1024`, labels ignored), `CKCER` computes the coordinate-wise mean $\mu_m \in \mathbb{R}^{96}$ and standard deviation $\sigma_m \in \mathbb{R}^{96}$ of $m(t_q - 1)$, with active-coordinate mask $I_r = \mathbb{I}[\sigma_{m,r} > \epsilon_{\text{dead}}]$ ($\epsilon_{\text{dead}} = 10^{-4}$):
$$\hat{m}_r(t_q - 1) = \begin{cases} \dfrac{m_r(t_q - 1) - \mu_{m,r}}{\sigma_{m,r}} & \text{if } \sigma_{m,r} > 10^{-4} \\ 0 & \text{otherwise} \end{cases}$$

### 2.3 Context Gate Formation (`Mode A` vs `Mode B`)
- **`CKCER-ModeA` (Explicit Query Cue)**:
  Uses the legal 2-channel query cue at the query tick:
  $$g(t_q) = \begin{bmatrix} g_0(t_q) \\ g_1(t_q) \end{bmatrix} = \begin{bmatrix} x_{\text{ch4}}(t_q) \\ x_{\text{ch5}}(t_q) \end{bmatrix} \in \{0, 1\}^2$$
- **`CKCER-ModeB` (100% Neural-Derived Query via Unsupervised Local Oja Gate)**:
  Never reads input channels 4 or 5 at the readout. Instead, it extracts the query-tick somatic transient of the 48 ALIF neurons:
  $$\delta_q(t_q) = \begin{bmatrix} z(t_q) \\ v(t_q) - \lambda v(t_q - 1) \end{bmatrix} \in \mathbb{R}^{96}, \quad \lambda = 0.90$$
  Standardizes $\delta_q(t_q)$ to $\hat{\delta}_q(t_q)$ using unsupervised observation statistics $(\mu_\delta, \sigma_\delta)$ with $\epsilon_{\text{dead}} = 10^{-4}$, and learns a single principal axis $w_q \in \mathbb{R}^{96}$ ($\|w_q\|_2 = 1$) via Oja's unsupervised local Hebbian rule during calibration (`10` lightweight epochs over the calibration somatic transients at $\eta_{\text{oja}} = 0.02$, zero access to labels $y$ or input cues):
  $$u_q = w_q^\top \hat{\delta}_q, \quad w_q \leftarrow \frac{w_q + \eta_{\text{oja}}\, u_q (\hat{\delta}_q - u_q w_q)}{\|w_q + \eta_{\text{oja}}\, u_q (\hat{\delta}_q - u_q w_q)\|_2}$$
  WTA lateral inhibition yields the binary dendritic gate:
  $$\hat{g}(t_q) = \begin{bmatrix} \mathbb{I}[w_q^\top \hat{\delta}_q(t_q) < \bar{u}_q] \\ \mathbb{I}[w_q^\top \hat{\delta}_q(t_q) \ge \bar{u}_q] \end{bmatrix} \in \{0, 1\}^2$$
  where $\bar{u}_q$ is the unsupervised calibration median of $u_q$.

### 2.4 Forward Computation and Local Associative Synaptic Update
The context-keyed feature representation has **zero un-gated shared trunk** and **zero random projection matrices**:
$$\psi(\hat{m}, g) = \begin{bmatrix} g_0(t_q)\, \hat{m}(t_q - 1) \\ g_1(t_q)\, \hat{m}(t_q - 1) \\ g_0(t_q) \\ g_1(t_q) \end{bmatrix} \in \mathbb{R}^{194}$$
Class scores for $c \in \{0, 1\}$:
$$s(x, q) = W \psi(\hat{m}, g) + b \in \mathbb{R}^2, \quad \hat{y} = \arg\max_{c \in \{0, 1\}} s_c(x, q)$$
where $W \in \mathbb{R}^{2 \times 194}, b \in \mathbb{R}^2$ (`390` learnable parameters in Mode A; `486` total in Mode B).

Upon receiving the binary target $y \in \{0, 1\}$ at query tick $t_q$, define one-hot target $t_y = [1 - y, y]^\top$ and discrete prediction $p_{\hat{y}} = [1 - \hat{y}, \hat{y}]^\top$. The local error is $e = t_y - p_{\hat{y}} \in \{-1, 0, +1\}^2$, and the bounded local associative update is:
$$\Delta W = \operatorname{clip}\!\left(\eta \, e \, \psi(\hat{m}, g)^\top, \, -\delta_{\max}, \, +\delta_{\max}\right), \quad \Delta b = \operatorname{clip}\!\left(\eta \, e, \, -\delta_{\max}, \, +\delta_{\max}\right)$$
$$W \leftarrow \operatorname{clip}(W + \Delta W, -w_{\max}, +w_{\max}), \quad b \leftarrow \operatorname{clip}(b + \Delta b, -w_{\max}, +w_{\max})$$
Because $g_{1-q}(t_q) = 0$ and standardization occurs *prior* to gating, $g_{1-q}(t_q)\hat{m}(t_q - 1) \equiv \mathbf{0}$, so **the inactive dendritic compartment receives an identically zero update ($\Delta W^{(1-q)} \equiv \mathbf{0}, \Delta b^{(1-q)} \equiv \mathbf{0}$)**.

---

## 3. Locked Hyperparameters

| Hyperparameter | Symbol | Locked Value | Notes |
|---|---|---|---|
| Readout learning rate | $\eta$ | `0.08` | Locked from theoretical step-size bound |
| Per-step update clip | $\delta_{\max}$ | `0.025` | Matches R5 per-step clip |
| Synaptic weight clip | $w_{\max}$ | `3.0` | Matches R5 weight bound |
| Dead-coordinate std threshold | $\epsilon_{\text{dead}}$ | `1e-4` | Replaces R5's `0.05` std floor |
| Cumulative trace decay | $\gamma_c$ | `1.0` | Exact spike-count trace |
| Adaptation trace decay | $\gamma_a$ | `0.95` | Native `SpikingCore.adaptation` decay |
| Somatic leak (Mode B transient) | $\lambda$ | `0.90` | Native `SpikingCore` leak |
| Mode B Oja learning rate | $\eta_{\text{oja}}$ | `0.02` | Unsupervised Oja gate calibration |
| Mode B Oja calibration sweeps | $E_{\text{oja}}$ | `10` | Unsupervised passes over calibration $\hat{\delta}_q$ |
| Calibration observation count | $N_{\text{obs}}$ | `1024` | Unlabeled pass over training stream |
| Online training passes | $E_{\text{train}}$ | `1` | Single online pass over `1,024` examples |

---

## 4. Pre-Registered Seed Bank and Data Generators

### 4.1 Seed Bank (Never Mixed)
- **Historical R5B Audit Seeds (Stage 0 only)**:
  - Pilot: `97001, 97007, 97013`
  - Confirmation: `98001, 98007, 98013, 98017, 98019, 98029, 98031, 98037`
- **Round 6C Development Seeds (Stage 2 & Stage 3 Dev Screen — max 3 seeds)**:
  - `120011, 120017, 120023`
- **Round 6C Confirmation Seeds (Stage 3 Confirmation — 8 seeds, run once after dev gates pass)**:
  - `121011, 121017, 121023, 121029, 121031, 121037, 121041, 121043`
- **Deterministic Stream Offsets**:
  - Training stream: `seed + 200000` (`N_train = 1024`)
  - Direct test stream: `seed + 300000` (`N_test = 512`)
  - Counterfactual paired stream: `seed + 400000` (`N_pairs = 256`, yielding `512` paired queries)
  - Auxiliary generalization stream: `seed + 500000`

### 4.2 Benchmark Conditions (Stage 3)
1. **`clean`**: Write 1 at tick `2`, Write 2 at tick `6`, query at tick `10`, background noise $\sigma = 0.01$.
2. **`train_shift_test_shift`**: Write 1 in `{1, 2, 3}`, Write 2 in `{5, 6, 7}` in both train and test.
3. **`ood_clean_to_shift`**: Trained on `clean` (`(2, 6)`), evaluated zero-shot on shifted write ticks (`{1, 2, 3}` and `{5, 6, 7}`).
4. **`reversed_writes`**: Both `(A @ t1, B @ t2)` and `(B @ t1, A @ t2)` write orders in train and test.
5. **`distractors`**: Random pulses on distractor channels `6, 7` (`p = 0.35`, amplitude `0.75`) at non-write ticks `< 10` during test (`paired_distractors` uses separate calls for `x_a` and `x_b` matching R5B).
6. **`paired_same_sample_distractors`**: Identical pre-query distractor realization shared between counterfactual siblings `x_a` and `x_b`.
7. **`noise_change`**: Trained at $\sigma = 0.01$, evaluated at $3\times$ background noise $\sigma = 0.03$.
8. **`query_tick_variation`**: Evaluated across query ticks $t_q \in \{10, 11, 12\}$ (`T = 13`).
9. **Protected Tasks**:
   - `simultaneous_xor_set_v1` (`sim_xor`)
   - `temporal_xor_set_v1` (`xor_separated`)
   - `temporal_order_set_v1` (`order_task`)
10. **Auxiliary Generalization Task (`three_write_variable_window_and_overwrite_v1`)**:
    - **Part 1 (`irrelevant_third_write_windowed_v1`)**: Write A in `{1, 2, 3}`, Write B in `{5, 6, 7}`, plus a third structured write at tick `8` on distractor channels `(6, 7)` carrying a random value $v_C \in \{0, 1\}$ with amplitude `0.85`, queried at tick `10`. Tests whether `CKCER` ignores a structured third key/value write while handling variable write windows.
    - **Part 2 (`key_overwrite_query_v1`)**: Write 1 at tick `2`, Write 2 at tick `5`, and an overwrite pulse at tick `8` that updates either Key A or Key B with a fresh binary value $v_{\text{new}} \in \{0, 1\}$; the query at tick `10` requests the latest value of Key A or Key B.

---

## 5. Stage 2 Fair Baseline Suite (All 10 Mandatory Controls)

Every baseline runs on the exact same frozen `SpikingCore(seed)` and identical data streams:
1. **`first_key_shortcut`**: Deterministic rule returning the first written value $v_{\text{first}}$.
2. **`original_r5_head`**: Unmodified R5 bilinear readout (`518` learnable parameters, `24,576` B random projection, post-gating z-score with `0.05` floor, evaluated on post-query state at $t_q$).
3. **`raw_serving_readout`**: Single-layer online readout matching `SpikingCore.learn` (`lr = 0.025`, clip `0.025`, raw un-normalized features).
4. **`offline_ridge_diagnostic`**: Closed-form Ridge regression ($\lambda_{\text{ridge}} = 10^{-2}$) on the exact context-keyed features — explicitly labeled **DIAGNOSTIC UPPER BOUND ONLY** (not an online local learner).
5. **`matched_capacity_control`**: Un-gated linear readout expanded via a random projection to `194` features (`390` learnable parameters, matching `CKCER-ModeA` capacity exactly) trained with the same local perceptron rule (`lr = 0.08`, pre-gating standardization `dead = 1e-4`).
6. **`query_only_control`**: `CKCER` architecture where the neural state $\hat{m}(t_q - 1)$ is zeroed out, leaving only the query cue $g(t_q)$.
7. **`query_free_state_control`**: `CKCER` architecture where both query gates are fixed to `1` ($g_0 = 1, g_1 = 0$), removing query identity while preserving full neural state $\hat{m}(t_q - 1)$.
8. **`random_label_control`**: `CKCER-ModeA` trained on randomly shuffled training labels $y_{\text{perm}}$.
9. **`fixed_nonlearning_candidate`**: `CKCER-ModeA` after unsupervised calibration (`observe`) with zero weight updates ($W = 0, b = 0$).
10. **`cue_shuffling_intervention`**:
    - `cue_shuf_train`: Query cues randomly permuted across examples during training, evaluated on true cues.
    - `cue_shuf_test`: Trained normally, query cues randomly permuted at test time.

Plus a **Single-Factor Ablation Ladder** from `original_r5_head` to `CKCER-ModeA`:
- `ablation_1_pre_query_trace_only`: R5 architecture (`518` params, post-gating `0.05` std floor) fed pre-query `[c, a]` instead of post-query `[z, v, a, f]`.
- `ablation_2_pre_gating_norm_only`: R5 architecture fed pre-query `[c, a]` with pre-gating `dead = 1e-4` normalization (still keeping un-gated trunk + random projection $P$).
- `ablation_3_full_ckcer_mode_a`: Removal of un-gated shared trunk and random projection $P$ (= `CKCER-ModeA`, `390` params).

---

## 6. Metrics, Hardest-Subset Analysis, and Statistical Protocol

### 6.1 Primary & Hardest-Subset Metrics (Section IX.A)
For every condition, we report:
- `direct`: Overall accuracy on held-out test set (`N = 512`).
- `both_all`: Fraction of counterfactual sibling pairs (`N = 256` pairs) where both Query A and Query B are answered correctly.
- `both_A`: Both-answers-correct rate on **Case A ($v_A \neq v_B$)** (the hardest subset).
- `both_B`: Both-answers-correct rate on **Case B ($v_A = v_B$)**.
- `acc_qA_diff`: Query A accuracy when $v_A \neq v_B$.
- `acc_qB_diff`: Query B accuracy when $v_A \neq v_B$.
- `margin_qA_diff`: Mean signed true-class margin $(2v_A - 1)(s_1(x,0) - s_0(x,0))$ when $v_A \neq v_B$.
- `margin_qB_diff`: Mean signed true-class margin $(2v_B - 1)(s_1(x,1) - s_0(x,1))$ when $v_A \neq v_B$.
- `flip_rate_A`: Prediction flip consistency $\Pr(\hat{y}(x, 0) \neq \hat{y}(x, 1) \mid v_A \neq v_B)$.
- `wrong_first_key_qB_diff`: Rate at which Query B outputs the first-key value $v_{\text{first}}$ when $v_A \neq v_B$.

### 6.2 Statistical Protocol
Across the 8 confirmation seeds (`121011..121043`), we compute:
- `mean`, `sd` (sample SD, `ddof=1`), `median`, `iqr` (`q75 - q25`).
- Paired per-seed differences $\Delta_s = M_{\text{candidate}}(s) - M_{\text{R5}}(s)$ and $\Delta_s^{\text{FK}} = M_{\text{candidate}}(s) - M_{\text{FK}}(s)$.
- `10,000`-resample paired bootstrap 95% confidence interval $[\text{CI}_{\text{low}}, \text{CI}_{\text{high}}]$ seeded at `2101`.
- Paired seed win count (`wins / 8`).

---

## 7. Pre-Registered GO / STOP Decision Gates

All 10 gates must pass on the 3 development seeds (`120011, 120017, 120023`) before running the 8 confirmation seeds (`121011..121043`), and all 10 gates must pass on confirmation to declare **GO**:

| Gate | Criterion | Threshold |
|---|---|---|
| **G1** | Direct query accuracy (`clean`) | $> \text{first\_key\_shortcut} + 0.10$ AND $> \text{R5}$ with $\text{CI}_{95\%,\text{low}}(\Delta_{\text{R5}}) > 0$ |
| **G2** | Hard counterfactual Case A (`both_A`, $v_A \neq v_B$, `clean`) | $\ge 0.75$ AND $> 0.6676$ with $\text{CI}_{95\%,\text{low}}(\Delta_{\text{R5}}) > 0$ |
| **G3** | Same-value counterfactual Case B (`both_B`, $v_A = v_B$, `clean`) | $\ge 0.90$ |
| **G4** | Timing shift (`train_shift_test_shift` & `ood_clean_to_shift`) | `direct` $\ge 0.80$ AND `both_A` $\ge 0.65$ |
| **G5** | Reversed write order (`reversed_writes`) | `direct` $\ge 0.80$ AND `both_A` $\ge 0.65$ |
| **G6** | Distractors (`distractors` & `paired_same_sample_distractors`) | `both_A` $\ge 0.60$ (vs R5B `0.4536`) |
| **G7** | Noise change (`noise_change`, $\sigma: 0.01 \to 0.03$) | `both_A` $\ge 0.60$ (vs R5B `0.4512`) |
| **G8** | Protected tasks (`sim_xor`, `xor_separated`, `order_task`) | No drop $> 0.01$ below R5B (`0.9885`, `0.8408`, `0.9961`) |
| **G9** | Integrity & falsification controls | `random_label` $\in [0.40, 0.60]$; `cue_shuf_test` `both_A` $\le 0.25$; `fixed_nonlearning` `both_A` $= 0.00$ |
| **G10** | Resource ceiling | Process-tree committed bytes $\le 2,000,000,000$; single CPU thread |
