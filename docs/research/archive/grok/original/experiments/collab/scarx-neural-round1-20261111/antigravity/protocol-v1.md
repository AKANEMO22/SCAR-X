# Protocol v1 — Dendritic Coincidence-Gated ALIF (`DC-ALIF`) with Counterfactual Readout Usefulness for Temporal Binding and Continual Retention

- **Campaign**: `scarx-neural-round1-20261111`
- **Agent ID**: `antigravity`
- **Environment**: `Antigravity` (Windows 10.0.26200, CPU-first, Python 3.11.9)
- **Model**: `Gemini 4 Argon (High)`
- **Toolchain Availability**: `ponytail` plugin checked via `search_marketplace` and not installed; recorded honestly per campaign guidelines.
- **Branch**: `research/antigravity-gemini/scarx-neural-round1-20261111`
  - *Worktree & Branch Collision Note*: On this shared host, the `deepseek` participant (`deepseek-ai/deepseek-v4.1-flash-1017` in `DeepSeek Harness (DSH) Web`) was invoked with the template's `AGENT_ID = antigravity` unchanged and claimed `local/worktrees/antigravity-scarx-neural-round1-20261111` (`research/antigravity/scarx-neural-round1-20261111`). To obey the strict rule against modifying or overwriting another participant's branch or running process, we left `research/antigravity/scarx-neural-round1-20261111` completely untouched for the `deepseek` participant and created an isolated worktree `C:\Users\hachimi\Downloads\model agent\scar-x-antigravity-round1` on branch `research/antigravity-gemini/scarx-neural-round1-20261111` from the exact `BASE_SHA`.
- **Base Commit (`BASE_SHA`)**: `1735ad66ea9e7b416f5df15df99d9c13cea46b4e`
- **Status**: `LOCKED_BEFORE_FORMAL_RUN`

---

## 1. Authorized Scope & Non-Negotiable Constraints

This protocol governs an isolated research prototype in `src/scarx/evaluation/` under the permissions granted by the `scarx-neural-round1-20261111` campaign prompt:
- **Serving Isolation**: No changes to `src/scarx/neural/core.py`, `src/scarx/neural/unified.py`, or any serving Hermes weights, databases, or runtime paths.
- **Architecture**: SCAR-X remains **one** Spiking Recurrent Neural Network of `48` ALIF neurons with sparse recurrent connectivity (`fan_in = 8`, `plastic_rows = 12`). Fast Processing, Deep Thinking, and Autonomous Idle Learning remain three operating modes of the same core.
- **Learning Signal (Proposal 30 Approved Exception)**: The candidate uses the approved one-step readout-error feedback $W_{\text{out}}^\top \text{error}$ combined with own-neuron forward eligibility traces. We explicitly disclose that $W_{\text{out}}^\top \text{error}$ is a one-step non-local readout broadcast and is **never** extended to hidden-to-hidden error transport ($W_{\text{rec}}^\top$) or reverse-time backpropagation.
- **BPTT Control Boundary**: `DCALIFBPTTControl` (`control_only=True`) is used strictly as an isolated upper-bound learnability control on the same forward architecture and is rejected by all candidate and serving loaders.
- **Resource Envelope**: CPU-first (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`); process-tree peak RSS and committed memory must remain $\le 2,000,000,000$ bytes across training, checkpoint save/load, and rollback, with zero SSD/pagefile weight offloading.

---

## 2. Independent Root-Cause Diagnosis

Before examining any peer outcomes, we audited `src/scarx/neural/core.py`, `src/scarx/evaluation/readout_eligibility.py`, `src/scarx/evaluation/binding_audit21.py`, `src/scarx/evaluation/tasks21.py`, `src/scarx/evaluation/readout_usefulness.py`, and the historical evidence in Reports 13–44 (`artifacts/direct-eligibility-latest.json` and `artifacts/readout-usefulness-pilot-latest.json`).

### 2.1 Why P1 `strict_binding` Failed Across Both Eligibility AND BPTT in Report 35
In Report 35 (`35_DIRECT_READOUT_ELIGIBILITY_RESULTS.md`), `ReadoutEligibility` failed on `strict_binding` with `correct = 63.41%`, `frozen = 62.24%`, and even `bptt_control = 63.54%`. Three compounding structural bottlenecks in the 1-compartment ALIF core explain why both forward eligibility and full BPTT hit a hard wall near $62.5\%$:

1. **Theorem 1 ($5/8 = 62.5\%$ Linear-Subthreshold Phantom-Coincidence Ceiling)**:
   In `strict_binding` (`binding_audit21.py`), each episode presents two complementary key-value pairs $(k_1, v_1)$ at tick $p_1 \in \{1, 2\}$ and $(k_2, v_2) = (1-k_1, 1-v_1)$ at tick $p_2 \in \{4, 5, 6\}$, followed by query $q \in \{0, 1\}$ at tick $t = 10$ with target $y = v_q = v_0 \oplus q$.
   In `SpikingCore`, subthreshold voltage $v_j(t) = \lambda v_j(t-1) + I_j(t)$ ($\lambda = 0.88$) integrates inputs **linearly** across ticks. Suppose a neuron assigns positive weight $w$ to key channel $k=q$ and value channel $v=1$. At $t=10$, the decayed subthreshold residual from $p_1$ and $p_2$ takes three possible values depending on $(v_q, \text{pair order})$:
   - $(q, 1)$ arrives late at $p_2$ ($y=1$, prob $1/4$): residual $= 2\lambda^{10-p_2}w$.
   - $(q, 1)$ arrives early at $p_1$ ($y=1$, prob $1/4$): residual $= 2\lambda^{10-p_1}w$.
   - $(q, 0)$ at $p_1$ and $(1-q, 1)$ at $p_2$, or vice versa ($y=0$, prob $1/2$): residual $= (\lambda^{10-p_1} + \lambda^{10-p_2})w$.
   Because $\lambda \in (0, 1)$ and $p_1 < p_2$, we have the strict inequality:
   $$2\lambda^{10-p_1} < \lambda^{10-p_1} + \lambda^{10-p_2} < 2\lambda^{10-p_2}$$
   That is, a **mismatched** episode ($y=0$) where $v=1$ appeared at $p_2$ with the *other* key $1-q$ produces a strictly larger linear residual at $t=10$ than a **matched** episode ($y=1$) where $(q, 1)$ appeared early at $p_1$! Any threshold on a single linear trace separates at most one of the two $y=1$ orderings from $y=0$, giving a theoretical Bayes accuracy of $(1/4 + 3/8) = 5/8 = 62.5\%$—matching Report 35 (`62.24%`–`63.54%`) to within sampling noise.

2. **Theorem 2 (Severed Recurrent Synapses Across Silent Delays)**:
   In `SpikingCore.step`, `W_rec` multiplies only the instantaneous 1-tick binary spike `z(t-1)`. During the silent delay ticks $t \in \{7, 8, 9\}$ prior to the query at $t = 10$ (and during $t=3$ between $p_1 \le 2$ and $p_2 \ge 4$), `z(t-1) = 0` unless an input-independent tonic loop fires, severing `W_rec` across inter-event gaps.

3. **Theorem 3 (One-Time Pad Pre-Query Gradient Cancellation & Somatic Reset Erasure)**:
   In `strict_binding`, the label $y = v_0 \oplus q$ has zero mutual information with the pre-query sequence $X_{<10}$ until $q \sim \text{Bernoulli}(0.5)$ arrives at $t=10$: $\mathbb{E}[y - 0.5 \mid X_{<10}] = 0$. In a 1-compartment ALIF neuron, somatic reset (`-theta * z_e`) erases early subthreshold sensitivity $e_{v,j}$, and because the pre-query current $I_j(10)$ enters additively with $v_j(9)$, the pre-query sensitivity $e_{v,j}(9)$ is not multiplied by the query sign $(2q-1)$. Consequently, the expected product $(y - 0.5)\,e_{v,j}(9)$ cancels to zero across $q \in \{0, 1\}$.

### 2.2 Why P2 `context_task` Retained $90.28\%$ Instead of $\ge 92\%$ in Report 42
In Report 42 (`42_PROPOSAL32_READOUT_USEFULNESS_RESULTS.md`), `ReadoutUsefulnessLearner` (`usefulness_full`) eliminated `old_task` forgetting (`0.00%`) and passed `overall_mean = 0.0159` and `overall_worst = 0.0417`, but `context_task` retained $90.28\%$ (`mean_forgetting = 0.0319 > 0.0300`, `worst = 0.0417 > 0.0300`). In `paired_retention.py`, `context_task` is a sequential context-binding task where a context cue arrives in ticks $0..1$ and the target payload arrives in ticks $3..5$. When `DC-ALIF` dendritic plateau traces preserve early context through the delay and counterfactual readout usefulness $\Omega_{k,j}$ consolidates both readout and dendritic-somatic hidden synapses, context representations resist drift across all 20 sequential contexts.

---

## 3. Candidate Mechanism: Dendritic Coincidence-Gated ALIF (`DC-ALIF`)

We augment each of the `48` ALIF neurons with a slow dendritic compartment $d_j(t) \in [-1, 1]$ driven by zero-diagonal cooperative synaptic coincidence and coupled to the soma via a bilinear dendritic-somatic coincidence gate, plus a slow recurrent presynaptic trace.

### 3.1 Exact Forward Equations
Let $N = 48$, $D = 64$, $K = \text{fan\_in} = 8$, and $M = \text{plastic\_rows} = 12$.
At tick $t = 0, \dots, T-1$:

1. **Slow-Conductance Recurrent Presynaptic Vector**:
   $$r_i(t-1) = (1 - \mu_{\text{rec}})\,z_i(t-1) + \mu_{\text{rec}}\,\frac{f_i(t-1)}{1 - \kappa}, \qquad \mu_{\text{rec}} = 0.5$$
   For each neuron $j$, its active presynaptic vector is:
   $$\text{pre}_j(t) = \left[\,x_{\text{active}}(t),\; r_{\text{rec\_idx}[j]}(t-1)\,\right] \in \mathbb{R}^{M + K}$$
   and its combined synaptic row is $w_j = [W_{\text{in}}[j, \text{active}],\, W_{\text{rec}}[j, :]] \in \mathbb{R}^{M + K}$.

2. **Linear Synaptic Drive & Zero-Diagonal Cooperative Synaptic Coincidence**:
   $$I_j(t) = b_j + \sum_{c=1}^{M+K} w_{j,c}\,\text{pre}_{j,c}(t)$$
   $$q_j(t) = \frac{1}{2}\left(\left(\sum_{c=1}^{M+K} w_{j,c}\,\text{pre}_{j,c}(t)\right)^2 - \sum_{c=1}^{M+K} w_{j,c}^2\,\text{pre}_{j,c}(t)^2\right) = \sum_{1 \le c < c' \le M+K} w_{j,c}\,w_{j,c'}\,\text{pre}_{j,c}(t)\,\text{pre}_{j,c'}(t)$$
   - *Why this solves Theorem 1*: Because self-squares $w_{j,c}^2 \text{pre}_{j,c}(t)^2$ are subtracted out, a single active input channel produces $q_j(t) = 0$. When a key-value pair arrives simultaneously at tick $p \in \{p_1, p_2\}$ and neuron $j$ holds contrastive weights $(+w_k, -w_k)$ on keys $(0, 1)$ and $(-w_v, +w_v)$ on values $(0, 1)$, **both** complementary pairs $(0, 1)$ and $(1, 0)$ in a $v_0 = 1$ episode yield the **exact same positive coincidence product** $+w_k w_v x_k x_v > 0$, whereas **both** pairs $(0, 0)$ and $(1, 1)$ in a $v_0 = 0$ episode yield $-w_k w_v x_k x_v < 0$, completely invariant to pair arrival order $(p_1, p_2)$!

3. **Slow Dendritic Plateau Trace (No Somatic Spike Reset)**:
   $$s_{d,j}(t) = \tanh\!\left(\alpha_d I_j(t) + \gamma_q q_j(t)\right)$$
   $$d_j(t) = \kappa_d\,d_j(t-1) + (1 - \kappa_d)\,s_{d,j}(t), \qquad \kappa_d = 0.96$$
   Crucially, $d_j(t)$ is located in the distal dendritic compartment and is **not** reset when the soma emits a spike $z_j(t)$.

4. **Bilinear Dendritic-Somatic Coincidence Gate & Somatic ALIF Update**:
   $$g_{\text{gate},j}(t) = \tanh\!\left(\beta_g I_j(t)\right), \qquad C_j(t) = g_{\text{gate},j}(t)\cdot d_j(t-1)$$
   $$u_j(t) = \lambda v_j(t-1) + \alpha_s I_j(t) + \beta_d C_j(t)$$
   $$\theta_j(t) = \theta_0 + \beta a_j(t-1)$$
   $$z_j(t) = \mathbb{I}\!\left[u_j(t) \ge \theta_j(t)\right], \qquad \psi_j(t) = \gamma \max\!\left(0,\; 1 - \frac{|u_j(t) - \theta_j(t)|}{\theta_0 \sigma}\right)$$
   $$v_j(t) = \text{clip}\!\left(u_j(t) - \theta_j(t)\,z_j(t),\; -2.0,\; 2.0\right)$$
   $$a_j(t) = \rho a_j(t-1) + z_j(t)$$
   $$f_j(t) = \kappa f_j(t-1) + (1 - \kappa)\,z_j(t)$$

5. **Readout Feature Vector**:
   $$\phi(T) = \left[\,f(T),\; \bar{z},\; \text{clip}(v(T)/2, -1, 1),\; a(T)/3,\; d(T)\,\right] \in \mathbb{R}^{5N} = \mathbb{R}^{240}$$

### 3.2 Own-Neuron Forward Eligibility Equations (No Hidden$\to$Hidden Transport)
For each neuron $j$ and each active synapse $c \in \{1, \dots, M+K\}$, define the instantaneous current sensitivity:
$$\frac{\partial I_j(t)}{\partial w_{j,c}} = \text{pre}_{j,c}(t), \qquad \frac{\partial q_j(t)}{\partial w_{j,c}} = \text{pre}_{j,c}(t)\,\left((I_j(t) - b_j) - w_{j,c}\,\text{pre}_{j,c}(t)\right)$$
Then the exact own-neuron forward eligibility recursion (ignoring $\partial \text{pre}_j(t)/\partial w_j$ across neurons, which is strictly excluded by Proposal 30) is:
$$s'_{d,j,c}(t) = \left(1 - s_{d,j}(t)^2\right)\left(\alpha_d \frac{\partial I_j(t)}{\partial w_{j,c}} + \gamma_q \frac{\partial q_j(t)}{\partial w_{j,c}}\right)$$
$$C'_{j,c}(t) = \beta_g\left(1 - g_{\text{gate},j}(t)^2\right) d_j(t-1)\,\frac{\partial I_j(t)}{\partial w_{j,c}} + g_{\text{gate},j}(t)\,e_{d,j,c}(t-1)$$
$$e_{d,j,c}(t) = \kappa_d\,e_{d,j,c}(t-1) + (1 - \kappa_d)\,s'_{d,j,c}(t)$$
$$e_{u,j,c}(t) = \lambda\,e_{v,j,c}(t-1) + \alpha_s \frac{\partial I_j(t)}{\partial w_{j,c}} + \beta_d\,C'_{j,c}(t)$$
$$e_{z,j,c}(t) = \psi_j(t)\,\left(e_{u,j,c}(t) - \beta\,e_{a,j,c}(t-1)\right)$$
$$e_{v,j,c}(t) = \left(1 - \theta_j(t)\,\psi_j(t)\right) e_{u,j,c}(t) + \left(\theta_j(t)\,\psi_j(t)\,\beta - z_j(t)\,\beta\right) e_{a,j,c}(t-1)$$
$$e_{a,j,c}(t) = \rho\,e_{a,j,c}(t-1) + e_{z,j,c}(t)$$
$$e_{f,j,c}(t) = \kappa\,e_{f,j,c}(t-1) + (1 - \kappa)\,e_{z,j,c}(t)$$
- *Why this solves Theorem 3*: In $C'_{j,c}(10)$, the stored dendritic coincidence sensitivity $e_{d,j,c}(9)$ is **multiplied by the somatic query gate** $g_{\text{gate},j}(10)$. When $g_{\text{gate},j}(10)$ has opposite signs for $q=0$ and $q=1$, the product $(y - \hat{p})\,g_{\text{gate},j}(10)\,e_{d,j,c}(9) \propto (2v_0 - 1)(2q - 1)^2$ has the **same non-canceling sign** on both $q=0$ and $q=1$ episodes!

At episode end $T$, the feature-aligned eligibility trace for neuron $j$ and readout projection $W_{\text{out}} = [W_f, W_{\bar{z}}, W_v, W_a, W_d] \in \mathbb{R}^{C_{\text{out}} \times 5N}$ uses the approved one-step readout broadcast:
$$\delta^{(k)}_j = W_{\text{out}}[k, k_N + j]^\top \text{error}, \quad k \in \{f, \bar{z}, v, a, d\}$$
$$\Delta w_{j,c} = -\eta_{\text{hid}}\left(\delta^{(f)}_j e_{f,j,c}(T) + \delta^{(\bar{z})}_j \bar{e}_{z,j,c}(T) + \frac{\delta^{(v)}_j}{2} e_{v,j,c}(T) + \frac{\delta^{(a)}_j}{3} e_{a,j,c}(T) + \delta^{(d)}_j e_{d,j,c}(T)\right)$$

### 3.3 Additional State, RAM, and Compute Footprint
- **Core State Overhead**: `d` is a single `float32[48]` array (`192 bytes`).
- **Eligibility State Overhead**: `e_d` has shape `(12, 48)` (`float32`, `2,304 bytes`) for active input rows and `(48, 8)` (`float32`, `1,536 bytes`) for recurrent fan-in, totaling **`4,032 bytes` (`3.94 KB`)** per learner instance.
- **Compute Overhead**: Elementwise vector operations of size $48 \times 20 = 960$ floats per tick ($< 0.05\text{ ms}$ per episode on CPU). Total process-tree RSS remains $< 150\text{ MB} \ll 2,000,000,000\text{ bytes}$.

### 3.4 What Will Falsify This Hypothesis
1. **Ablation Falsification**: If removing the bilinear dendritic-somatic gate (`no_gate`, $\beta_d = 0$ or $g_{\text{gate}} \equiv 1$ with $\gamma_q = 0$) or replacing `DC-ALIF` with standard 1-compartment `baseline_elig` (`ReadoutEligibility`) achieves equal or higher accuracy on `strict_binding`, the dendritic coincidence mechanism is falsified.
2. **Signal-Specificity Falsification**: If `correct` fails to beat `frozen`, `zero`, and `shuffled` by $\ge 5.0$ percentage points with matched `95%` bootstrap CI $> 0$ on `strict_binding` and `delayed_cue`, the own-neuron forward eligibility signal is falsified.
3. **Continual-Retention Falsification**: If P2 20-context retention with `DC-ALIF + Counterfactual Readout Usefulness` fails any of `old_task >= 0.92`, `context_task >= 0.92`, `overall_mean <= 0.03`, or `overall_worst <= 0.05`, the P2 retention claim is falsified.

---

## 4. Preregistered Experimental Design, Datasets, Splits, and Gates

### 4.1 P1 Benchmark Suite (Temporal Credit Assignment & Non-Linear Binding)
- **Tasks & Generators**:
  1. `delayed_cue` (`src/scarx/evaluation/tasks21.py::generate_delayed_cue`): `seq_len = 12`, cue at $t=0$, distractor burst in $t \in [2, 8]$, context query at $t=10$, response window $t \in [10, 12)$.
  2. `strict_binding` (`src/scarx/evaluation/binding_audit21.py::generate_strict_binding`): `seq_len = 12`, complementary pairs $(k_1, v_1)$ at $p_1 \in \{1, 2\}$ and $(1-k_1, 1-v_1)$ at $p_2 \in \{4, 5, 6\}$, distractors in $t \in [2, 8]$, query $q$ at $t=10$, label $y = v_q = v_0 \oplus q$.
  3. `order_sensitive_vision` (`src/scarx/evaluation/binding_audit21.py::generate_order_sensitive_vision`): `seq_len = 10`, spatial-temporal permutation parity over two ordered visual tokens (`left_right_move`, `top_bottom_swap`, `color_swap`, `diagonal_flip`) with query at $t=8$.
- **Splits & Anti-Leakage Verification**:
  - Per seed $s$:
    - `train`: `n = 640` episodes (`seed = s + 10_000`)
    - `dev`: `n = 256` episodes (`seed = s + 20_000`)
    - `test` (unseen evaluation): `n = 256` episodes (`seed = s + 30_000`)
    - `probe_train`: `n = 256` episodes (`seed = s + 40_000`) for frozen-hidden ridge probe refit
    - `probe_test`: `n = 256` episodes (`seed = s + 50_000`) for unseen representation usefulness evaluation
  - Dataset SHA256 hashes over `(x, y)` tensors are computed and logged in the manifest before training. Zero target $y$ is passed to `step()`, `features()`, or `predict()`.
- **7 Evaluated Conditions (Identical Initial Weights & Episode Ordering per Seed)**:
  1. `correct`: `DC-ALIF` with exact own-neuron forward eligibility + one-step readout broadcast $W_{\text{out}}^\top \text{error}$.
  2. `no_gate`: Mechanism ablation where the bilinear coincidence gate and quadratic synaptic coincidence are disabled ($\gamma_q = 0$, $g_{\text{gate},j}(t) \equiv 1$), reducing the dendritic compartment to a linear passive low-pass filter.
  3. `baseline_elig`: Standard 1-compartment `ReadoutEligibility` (`correct` mode from Proposal 30) on the exact same episodes and budget.
  4. `shuffled`: `DC-ALIF` where the active eligibility tensor is randomly permuted across neurons and active rows via `signal_rng` before each update.
  5. `zero`: `DC-ALIF` with `hidden_lr` multiplied by `0.0` inside the update loop while consuming the exact same `signal_rng` calls as `shuffled` and updating only the linear readout.
  6. `frozen`: `DC-ALIF` where hidden weights are never updated (only readout updates). Must match `zero` bit-for-bit (`max_abs_diff == 0.0`).
  7. `bptt_control`: Isolated `DCALIFBPTTControl` (`control_only=True`) trained with exact reverse-time gradients through the `DC-ALIF` recurrence for `3` epochs (`1,920` updates).
- **Training Budget & Seeds**:
  - `epochs = 6` (`3,840` online updates per condition per task; `bptt_control` uses `3` epochs = `1,920` updates).
  - **Development Screen**: `3` fresh independent model seeds `[101, 102, 103]`.
  - **Confirmation Run (Triggered ONLY if Development Screen passes all locked gates)**: `8` fresh independent model seeds `[201, 202, 203, 204, 205, 206, 207, 208]`.
- **Locked P1 Acceptance Gates**:
  - On **both** `delayed_cue` and `strict_binding` (primary binding benchmarks), and tracked on `order_sensitive_vision`:
    - `mean(correct) - mean(frozen) >= 0.05` (`+5.0` percentage points)
    - `mean(correct) - mean(zero) >= 0.05`
    - `mean(correct) - mean(shuffled) >= 0.05`
    - `mean(correct) - mean(baseline_elig) >= 0.05` on `strict_binding`
    - `mean(correct) - mean(no_gate) >= 0.05` on `strict_binding`
    - Paired bootstrap `95%` confidence interval lower bound (`10,000` resamples, `seed = 20261111`) for `correct - frozen`, `correct - zero`, `correct - shuffled`, and `correct - no_gate` strictly `> 0.0`.
    - **Refit-Head / Representation Usefulness Gate**: On unseen `probe_test` episodes (`n = 256`) with a freshly refitted ridge readout (`alpha = 1.0`) trained on `probe_train` (`n = 256`) over frozen hidden representations $\phi(T)$, `mean(probe_correct) - mean(probe_frozen) >= 0.05` with `95%` CI lower bound `> 0.0`.

### 4.2 P2 Benchmark Suite (20-Context Continual Retention Stream)
- **Stream & Generator**:
  - Standard 20-context stream from `src/scarx/evaluation/paired_retention.py::build_default_tasks` (`old_task`, `new_task`, `context_task`, `burst_task`, cycled across `20` contexts; `train_per_context = 140`, `eval_per_task = 72`, `replay_Per_step = 4`).
- **Conditions & Seeds**:
  - `3` fresh independent model seeds `[20260401, 20260402, 20260403]` across `5` conditions:
    1. `dc_alif_usefulness`: `DC-ALIF` core + Counterfactual Readout Usefulness consolidation ($\Omega_{k,j}$ protecting readout, active input/recurrent hidden synapses, and dendritic-somatic projections).
    2. `baseline_usefulness_full`: Standard 1-compartment `ReadoutUsefulnessLearner` (`usefulness_full` from Proposal 32).
    3. `dc_alif_unprotected`: `DC-ALIF` core with usefulness consolidation disabled ($\lambda_{\text{out}} = \lambda_{\text{hid}} = 0$).
    4. `dc_alif_shuffled_omega`: `DC-ALIF` core where $\Omega_{k,j}$ is randomly permuted across features/neurons at consolidation boundaries.
    5. `dc_alif_frozen_hidden`: `DC-ALIF` core with hidden plasticity disabled ($\eta_{\text{hid}} = 0$) and usefulness consolidation active.
- **Locked P2 Acceptance Gates**:
  - `final_accuracy >= 0.92` on **both** `old_task` and `context_task` (and `overall_final >= 0.85`).
  - `overall_mean_forgetting <= 0.03` (`3.0` percentage points) and `overall_worst_forgetting <= 0.05` (`5.0` percentage points).
  - `mean_forgetting <= 0.03` on both `old_task` and `context_task`.

---

## 5. Mandatory P0 Technical Verification Gate (Pre-Experiment)

No formal P1 or P2 claim is valid unless `scripts/validate_dc_alif_p0.py` and `pytest tests/test_dc_alif_p0.py` pass 100% of the following checks:
1. **Anchored-Surrogate Finite-Difference Audit (`64` Synapses)**:
   On a fixed presynaptic tape (`T = 10`) with anchored spike indicator $z_j(t)$ and surrogate derivative $\psi_j(t)$, the analytical forward recursion for $(e_{d}, e_{v}, e_{a}, e_{f})$ across `32` active input synapses and `32` recurrent synapses must match central finite differences ($\epsilon = 10^{-5}$) within `atol = 1e-5, rtol = 1e-3`.
2. **Zero Target Leakage Audit**:
   Running `features(x)` and `predict(x)` before vs. after passing deliberate adversarial labels $y \in \{0, 1\}$ to an external copy must produce bit-identical spikes, voltages, dendritic traces $d(t)$, and predictions (`max_abs_diff == 0.0`).
3. **Matched RNG & Zero/Frozen Equivalence**:
   `zero` and `frozen` must produce bit-identical weights, predictions, and `signal_rng` states (`max_abs_diff == 0.0`), while `shuffled` advances `signal_rng` identically to `zero` and `correct`.
4. **Mid-Segment Checkpoint Save/Load & Cold-Process Resume**:
   Splitting a training run at step $K$, serializing via `dumps()`, restoring in a fresh Python subprocess via `loads()`, and running steps $K..2K$ must match an uninterrupted run of $2K$ steps bit-for-bit across `W_in`, `W_rec`, `b`, `W_out`, `b_out`, `v`, `a`, `f`, `z`, `d`, `e_v`, `e_a`, `e_f`, `e_d`, `W_in_anchor`, `W_rec_anchor`, `W_in_omega`, `W_rec_omega`, and all RNG states (`max_abs_diff == 0.0`).
5. **Fault-Injection Rollback Integrity**:
   Injecting a runtime exception mid-update or mid-checkpoint-load must roll back all model weights, neuron states (including $d$), eligibility traces (including $e_d$), usefulness states, and RNG states to the exact pre-call snapshot (`max_abs_diff == 0.0`).
6. **Cross-Loader Isolation & Checksum Tamper Rejection**:
   - `UnifiedCore.loads`, `VisualWords.loads`, and `ReadoutEligibility.loads` must reject `DCALIFEligibility` and `DCALIFBPTTControl` payloads with `ValueError`.
   - `DCALIFEligibility.loads` must reject `DCALIFBPTTControl` payloads (`control_only=True`) and any payload with a 1-byte mutated SHA256 checksum.

---

## 6. Hyperparameter Selection & Stop Rules

- All hyperparameters are locked prior to the formal run:
  - `DCALIFCore`: $\lambda = 0.88$, $\rho = 0.94$, $\kappa = 0.90$, $\kappa_d = 0.96$, $\mu_{\text{rec}} = 0.50$, $\theta_0 = 0.32$, $\beta = 0.16$, $\gamma = 0.35$, $\sigma = 0.65$, $\alpha_s = 0.35$, $\alpha_d = 0.25$, $\gamma_q = 2.20$, $\beta_g = 2.00$, $\beta_d = 1.15$.
  - `DCALIFEligibility`: `hidden_lr = 0.028`, `readout_lr = 0.030`, `weight_decay = 1e-4`, `grad_clip = 1.0`.
- At most **2** documented protocol amendments are permitted if and only if diagnosed on `dev` seeds `[101, 102, 103]`. Confirmation seeds `[201..208]` are run **at most once** after the `dev` screen passes all gates.
