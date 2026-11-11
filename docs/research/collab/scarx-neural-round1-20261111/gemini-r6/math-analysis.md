# Mathematical Analysis of Local Associative Query Retrieval in Recurrent ALIF Networks (`gemini-r6`)

- **Campaign**: `scarx-neural-round1-20261111` (SCAR-X Round 6C)
- **Branch**: `research/gemini/r6-local-associative-learning`
- **Mechanism**: **Context-Keyed Competitive Eligibility Readout (`CKCER`)**

---

## 1. Exact Counterfactual Margin Geometry

Consider an episode $x$ in which two independent binary values $v_A, v_B \in \{0, 1\}$ are written to keys $A, B \in \{0, 1\}$ at ticks $t_1 < t_2 < t_q$, and a query $q \in \{0, 1\}$ arrives at tick $t_q$. The target output is:

$$y(v_A, v_B, q) = (1 - q)v_A + q v_B = v_A + q(v_B - v_A) \in \{0, 1\}$$

For any 2-class readout producing logits $s(x, q) = [s_0(x, q), s_1(x, q)]^\top \in \mathbb{R}^2$, define the scalar class-1 margin:

$$z(x, q) \triangleq s_1(x, q) - s_0(x, q)$$

So that $\hat{y}(x, q) = \mathbb{I}[z(x, q) > 0]$ (with ties broken toward class $0$). On any counterfactual sibling pair $(x^{(0)}, x^{(1)})$ sharing identical pre-query history $x_{0:t_q-1}$ and differing only in the query $q \in \{0, 1\}$ at tick $t_q$, decompose the pair of margins $(z(x, 0), z(x, 1))$ into orthogonal **common-mode** and **differential-mode** coordinates:

$$z_{\text{com}}(x) \triangleq \frac{z(x, 0) + z(x, 1)}{2}, \qquad z_{\text{diff}}(x) \triangleq \frac{z(x, 0) - z(x, 1)}{2}$$

so that:

$$z(x, 0) = z_{\text{com}}(x) + z_{\text{diff}}(x), \qquad z(x, 1) = z_{\text{com}}(x) - z_{\text{diff}}(x)$$

### Proposition 1.1 (Exact Necessary and Sufficient Conditions for Counterfactual Retrieval)
1. **Case A ($v_A \neq v_B$, where $v_B = 1 - v_A$ and $v_A - v_B \in \{-1, +1\}$)**:
   Both counterfactual queries are answered correctly ($\hat{y}(x, 0) = v_A$ and $\hat{y}(x, 1) = v_B$) if and only if:
   $$(v_A - v_B)\, z_{\text{diff}}(x) > |z_{\text{com}}(x)|$$
2. **Case B ($v_A = v_B = v \in \{0, 1\}$, where $2v - 1 \in \{-1, +1\}$)**:
   Both counterfactual queries are answered correctly ($\hat{y}(x, 0) = \hat{y}(x, 1) = v$) if and only if:
   $$(2v - 1)\, z_{\text{com}}(x) > |z_{\text{diff}}(x)|$$

### Corollary 1.2 (Why R5's Un-Gated Shared Trunk Fails on Case A)
In the R5 bilinear readout:

$$\phi_{\text{R5}}(s, q) = \begin{bmatrix} \tilde{s} \\ \tilde{c}(q) \\ \tilde{g}_0(s, q) \\ \tilde{g}_1(s, q) \end{bmatrix} \in \mathbb{R}^{192 + 2 + 32 + 32 = 258}$$

let $w = W_{1,:} - W_{0,:} = [w_s^\top, w_c^\top, w_{g_0}^\top, w_{g_1}^\top]^\top \in \mathbb{R}^{258}$ denote the effective margin weight vector. Suppose first that the pre-query state were used so $\tilde{s}$ is identical for $q=0$ and $q=1$. Then:

$$z_{\text{com}}^{\text{R5}}(x) = \underbrace{w_s^\top \tilde{s}}_{\text{192-dim un-gated trunk}} + \frac{1}{2}(w_{c,0} + w_{c,1}) + \frac{1}{2}\left(w_{g_0}^\top (\tilde{g}_0|_{q=0} + \tilde{g}_0|_{q=1}) + w_{g_1}^\top (\tilde{g}_1|_{q=0} + \tilde{g}_1|_{q=1})\right)$$

$$z_{\text{diff}}^{\text{R5}}(x) = \frac{1}{2}(w_{c,0} - w_{c,1}) + \frac{1}{2}\left(w_{g_0}^\top (\tilde{g}_0|_{q=0} - \tilde{g}_0|_{q=1}) - w_{g_1}^\top (\tilde{g}_1|_{q=1} - \tilde{g}_1|_{q=0})\right)$$

Notice two structural pathologies:
1. **Zero Contribution of $w_s \in \mathbb{R}^{192}$ to $z_{\text{diff}}(x)$**:
   The `192`-dimensional un-gated trunk $w_s^\top \tilde{s}$ (representing $74.4\%$ of R5's weights) cancels out of $z_{\text{diff}}^{\text{R5}}(x)$ completely! Yet during single-pass online training, $w_s$ updates on $100\%$ of training examples, accumulating the marginal correlation $\mathbb{E}[(2y - 1)\tilde{s}] = \frac{1}{2}\mathbb{E}[(v_A + v_B - 1)\tilde{s}]$. Whenever $v_A \neq v_B$ ($v_A + v_B - 1 = 0$), $w_s^\top \tilde{s}$ acts purely as zero-mean, high-variance common-mode interference in $z_{\text{com}}^{\text{R5}}(x)$ with variance $\operatorname{Var}(w_s^\top \tilde{s}) = w_s^\top \Sigma_{\tilde{s}\mid v_A \neq v_B} w_s$. By Proposition 1.1, every sample where $|w_s^\top \tilde{s}| \ge (v_A - v_B) z_{\text{diff}}^{\text{R5}}(x)$ forces both queries to output the same answer ($\hat{y}(x, 0) = \hat{y}(x, 1)$), causing a Case-A failure.
2. **Post-Gating Z-Score Cross-Talk ($\|\tilde{g}_{1-q}|_{c_{1-q}=0}\|_2 = 4.3856$)**:
   Because R5 computes calibration statistics $\mu_{g_k} = \mathbb{E}[(P^\top s) c_k(q)] = \frac{1}{2} P^\top \mathbb{E}[s] \neq \mathbf{0}$ and $\sigma_{g_k} = \max(\operatorname{std}((P^\top s)c_k(q)), 0.05)$ *after* multiplying by $c_k(q) \in \{0, 1\}$, setting $c_k(q) = 0$ on the gated-off branch yields a **non-zero constant vector**:
   $$\tilde{g}_k\big|_{c_k(q)=0} = \frac{\mathbf{0} - \mu_{g_k}}{\sigma_{g_k}} = -\frac{\mu_{g_k}}{\sigma_{g_k}} \neq \mathbf{0}$$
   On the historical R5B seeds (`stage0-audit.json`), $\|\tilde{g}_0|_{q=1}\|_2 = 4.3856$. Consequently, on every Query B ($q = 1$) training example, the local delta rule applies a dense non-zero update $\Delta w_{g_0} = \operatorname{clip}(-\eta e(t_q) \mu_{g_0}/\sigma_{g_0}, -\delta_{\max}, +\delta_{\max})$ to Query A's weight compartment $w_{g_0}$!

---

## 2. How `CKCER` Eliminates Both Interference Terms

In `CKCER`, the pre-query eligibility state $m(t_q - 1) = [c(t_q - 1); \, a(t_q - 1)] \in \mathbb{R}^{96}$ is standardized *prior* to multiplicative gating:

$$\hat{m}_r(t_q - 1) = \mathbb{I}[\sigma_{m,r} > \epsilon_{\text{dead}}] \cdot \operatorname{clip}\!\left(\frac{m_r(t_q - 1) - \mu_{m,r}}{\sigma_{m,r}}, -5, +5\right), \qquad \epsilon_{\text{dead}} = 10^{-4}$$

and the feature vector is purely context-keyed with **no un-gated trunk** and **no random projection matrix $P$**:

$$\psi(\hat{m}, g) = \begin{bmatrix} g_0(t_q)\, \hat{m}(t_q - 1) \\ g_1(t_q)\, \hat{m}(t_q - 1) \\ g_0(t_q) \\ g_1(t_q) \end{bmatrix} \in \mathbb{R}^{194}$$

### Proposition 2.1 (Orthogonal Dendritic Compartments and Zero Cross-Talk)
Let $w^{(0)} = W_{1, 0:96} - W_{0, 0:96} \in \mathbb{R}^{96}$ and $w^{(1)} = W_{1, 96:192} - W_{0, 96:192} \in \mathbb{R}^{96}$ be the effective margin weights of the two dendritic branches, with scalar biases $\beta_0, \beta_1 \in \mathbb{R}$. Then:
1. **Exact Branch Isolation During Inference**:
   $$z(x, 0) = {w^{(0)}}^\top \hat{m}(t_q - 1) + \beta_0, \qquad z(x, 1) = {w^{(1)}}^\top \hat{m}(t_q - 1) + \beta_1$$
   Hence:
   $$z_{\text{com}}(x) = \frac{1}{2}(w^{(0)} + w^{(1)})^\top \hat{m}(t_q - 1) + \frac{\beta_0 + \beta_1}{2}, \qquad z_{\text{diff}}(x) = \frac{1}{2}(w^{(0)} - w^{(1)})^\top \hat{m}(t_q - 1) + \frac{\beta_0 - \beta_1}{2}$$
2. **Exact Zero Cross-Compartment Plasticity**:
   When $g(t_q) = [1, 0]^\top$ ($q = 0$), $\psi_{96:192} = 0 \cdot \hat{m}(t_q - 1) \equiv \mathbf{0}$ and $\psi_{193} = 0$, so:
   $$\Delta W_{:, 96:192}\big|_{q=0} \equiv \mathbf{0}, \qquad \Delta W_{:, 193}\big|_{q=0} \equiv \mathbf{0}$$
   Symmetrically, when $g(t_q) = [0, 1]^\top$ ($q = 1$), $\Delta W_{:, 0:96}|_{q=1} \equiv \mathbf{0}$ and $\Delta W_{:, 192}|_{q=1} \equiv \mathbf{0}$.
3. **Asymptotic Margin Alignment Under Independent $(v_A, v_B)$**:
   Because $v_A \perp v_B$, on $q = 0$ examples the target is $y = v_A$ (independent of $v_B$), so $w^{(0)}$ converges to the linear separator of $v_A$ on $\hat{m}(t_q - 1)$ with $\mathbb{E}[w^{(0)\top}\hat{m} \mid v_A, v_B] = \mu_A (2v_A - 1)$ ($\mu_A > 0$). Symmetrically, on $q = 1$ examples $w^{(1)}$ converges to the linear separator of $v_B$ with $\mathbb{E}[w^{(1)\top}\hat{m} \mid v_A, v_B] = \mu_B (2v_B - 1)$ ($\mu_B > 0$). Therefore:
   - On **Case A ($v_B = 1 - v_A$, so $2v_B - 1 = -(2v_A - 1) = -(v_A - v_B)$)**:
     $$\mathbb{E}[z_{\text{diff}}(x) \mid v_A \neq v_B] = \frac{\mu_A + \mu_B}{2}(v_A - v_B), \qquad \mathbb{E}[z_{\text{com}}(x) \mid v_A \neq v_B] = \frac{\mu_A - \mu_B}{2}(v_A - v_B) \approx 0$$
     Thus $(v_A - v_B)\mathbb{E}[z_{\text{diff}}(x)] = \frac{1}{2}(\mu_A + \mu_B) \gg |\mathbb{E}[z_{\text{com}}(x)]| \approx 0$.
   - On **Case B ($v_A = v_B = v$)**:
     $$\mathbb{E}[z_{\text{com}}(x) \mid v_A = v_B = v] = \frac{\mu_A + \mu_B}{2}(2v - 1), \qquad \mathbb{E}[z_{\text{diff}}(x) \mid v_A = v_B = v] = \frac{\mu_A - \mu_B}{2}(2v - 1) \approx 0$$
     Thus $(2v - 1)\mathbb{E}[z_{\text{com}}(x)] = \frac{1}{2}(\mu_A + \mu_B) \gg |\mathbb{E}[z_{\text{diff}}(x)]| \approx 0$.

---

## 3. Dual-Timescale Eligibility Moment Analysis ($c_j(t)$ and $a_j(t)$)

Let $z_j(0), z_j(1), \dots, z_j(t_q - 1) \in \{0, 1\}$ denote the pre-query spike train of ALIF neuron $j \in \{1, \dots, 48\}$. `CKCER` forms $m_j(t_q - 1) = [c_j(t_q - 1), \, a_j(t_q - 1)]^\top$ from:

$$c_j(t_q - 1) = \sum_{s=0}^{t_q - 1} z_j(s), \qquad a_j(t_q - 1) = \sum_{s=0}^{t_q - 1} \gamma_a^{(t_q - 1) - s} z_j(s) \quad (\gamma_a = 0.95)$$

Expanding $\gamma_a^{\tau} = (1 - \epsilon_a)^{\tau} = 1 - \epsilon_a \tau + \mathcal{O}(\epsilon_a^2 \tau^2)$ for $\epsilon_a = 1 - \gamma_a = 0.05$ and $\tau = (t_q - 1) - s \in \{0, \dots, 9\}$ gives:

$$\begin{bmatrix} c_j(t_q - 1) \\ \dfrac{c_j(t_q - 1) - a_j(t_q - 1)}{\epsilon_a} \end{bmatrix} = \begin{bmatrix} \displaystyle\sum_{s=0}^{t_q - 1} z_j(s) \\[10pt] \displaystyle\sum_{s=0}^{t_q - 1} (t_q - 1 - s)\, z_j(s) + \mathcal{O}(\epsilon_a) \end{bmatrix} = \begin{bmatrix} M_j^{(0)} \\ M_j^{(1)} + \mathcal{O}(\epsilon_a) \end{bmatrix}$$

where $M_j^{(0)}$ is the **0-th temporal moment** (exact spike count, invariant to write-tick shifts and write-order permutations) and $M_j^{(1)}$ is the **1-st temporal moment** (spike-time center of mass, which distinguishes early vs. late writes on `order_task` and weights tick-8 overwrites more heavily than tick-2/tick-5 initial writes on `aux_key_overwrite`).

Simultaneously, because both $c_j(t_q - 1)$ and $a_j(t_q - 1)$ depend only on hard thresholded spikes $z_j(s) = \mathbb{I}[v_j(s) \ge \theta + \beta a_j(s - 1)]$ for $s \le t_q - 1$:
1. Subthreshold Gaussian background fluctuations $\xi(s) \sim \mathcal{N}(0, \sigma^2)$ that fail to cross threshold $\theta = 0.5$ are completely rejected (whereas R5's $v(t_q)$ injects un-thresholded noise directly into $P^\top s$).
2. The query pulse at tick $t_q$ cannot perturb $m(t_q - 1)$, eliminating R5's post-query somatic overwrite.

---

## 4. Spectral Convergence of Mode B Unsupervised Oja Gating

In `CKCER-ModeB`, the readout is completely blind to the input query channels `(4, 5)`. Instead, at query tick $t_q$ it observes the 96-dimensional somatic transient:

$$\delta_q(t_q) = \begin{bmatrix} z(t_q) \\ v(t_q) - \lambda v(t_q - 1) \end{bmatrix} \in \mathbb{R}^{96}, \qquad \lambda = 0.65$$

From the ALIF membrane update at tick $t_q$:

$$v(t_q) - \lambda v(t_q - 1) = W_{\text{in}} x(t_q) + \sum_{\text{fan-in}} W_{\text{rec}} z(t_q - 1) - z(t_q)\left(\theta + \beta a(t_q - 1)\right)$$

Since $x(t_q) = \text{PULSE} \cdot e_{4 + q} + \xi(t_q)$, the query pulse drives a deterministic separation vector $\Delta_{\text{in}} = \text{PULSE} \cdot (W_{\text{in}, :, 5} - W_{\text{in}, :, 4}) \in \mathbb{R}^{48}$ in the current $v(t_q) - \lambda v(t_q - 1)$ and in the triggered spikes $z(t_q)$. After coordinate-wise standardization to $\hat{\delta}_q(t_q)$ (`mean = 0`, `std = 1` on active coordinates), the covariance matrix decomposes as:

$$\Sigma_{\hat{\delta}} = \mathbb{E}\!\left[\hat{\delta}_q \hat{\delta}_q^\top\right] = \frac{1}{4}\Delta\hat{\mu}_q \Delta\hat{\mu}_q^\top + \Sigma_{\text{within}}$$

where $\Delta\hat{\mu}_q = \mathbb{E}[\hat{\delta}_q \mid q=1] - \mathbb{E}[\hat{\delta}_q \mid q=0]$ is coherent across all query-responsive coordinates ($\|\Delta\hat{\mu}_q\|_2^2 \approx 75.4$ on the 48-ALIF core), whereas $\Sigma_{\text{within}}$ is weakly correlated recurrent/background variance ($\lambda_{\max}(\Sigma_{\text{within}}) \approx 3.1$). By Oja's theorem (Oja, 1982), the local Hebbian update:

$$u_q = w_q^\top \hat{\delta}_q, \qquad \Delta w_q = \eta_{\text{oja}}\, u_q \left(\hat{\delta}_q - u_q w_q\right)$$

converges almost surely to $\pm v_1(\Sigma_{\hat{\delta}})$, which has cosine similarity $> 0.98$ with $\Delta\hat{\mu}_q / \|\Delta\hat{\mu}_q\|_2$. Because $\mathbb{E}[\hat{\delta}_q] = \mathbf{0}$, the two query clusters are symmetrically centered at $\mp \frac{1}{2} w_q^\top \Delta\hat{\mu}_q$ around the origin `0.0`. Thresholding $u_q = w_q^\top \hat{\delta}_q(t_q)$ at `0.0` (rather than the sample median) is therefore invariant to finite-sample fluctuations in the empirical ratio of $q=0$ to $q=1$ examples, achieving **`1.0000` (100.0%) query-gate clustering purity across all 3 dev seeds and all 8 confirmation seeds**.

---

## 5. Quantitative Single-Factor Ablation Decomposition (Dev & Confirmation)

The table below isolates the exact causal contribution of each mathematical modification from `original_r5_head` (`step 0`) to `CKCER-ModeA` (`step 3`) on both the 3 pre-registered development seeds (`120011..120023`) and the 8 confirmation seeds (`121011..121043`):

| Ablation Step | State Input | Standardization | Trunk & Projection | Params | Dev `direct` | Dev `both_A` | Conf `direct` | Conf `both_A` | Conf `wrong_fk_qB` |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| **Step 0 (`original_r5_head`)** | Post-query `[z,v,a,f]` (`192`) | Post-gating z-score (`0.05` floor) | Un-gated trunk + random $P$ (`32`) | `518` | `0.9323` | `0.7218` | `0.9226` | `0.7520` | `0.1498` |
| **Step 1 (`pre_query_trace_only`)** | Pre-query `[c,a]` (`96`) | Post-gating z-score (`0.05` floor) | Un-gated trunk + random $P$ (`32`) | `326` | `0.9753` | `0.8865` | `0.9785` | `0.9218` | `0.0542` |
| **Step 2 (`pre_gating_norm_plus_trunk`)** | Pre-query `[c,a]` (`96`) | Pre-gating (`dead=1e-4`, zero cross-talk) | Un-gated trunk + random $P$ (`32`) | `326` | `0.9701` | `0.8987` | `0.9722` | `0.9131` | `0.0485` |
| **Step 3 (`CKCER-ModeA`)** | Pre-query `[c,a]` (`96`) | Pre-gating (`dead=1e-4`, zero cross-talk) | **No un-gated trunk, no random $P$** | **`390`** | **`0.9987`** | **`1.0000`** | **`0.9985`** | **`0.9960`** | **`0.0030`** |
| **`CKCER-ModeB` (Neural Query)** | Pre-query `[c,a]` (`96`) | Pre-gating (`dead=1e-4`, zero cross-talk) | **No trunk, no $P$, local Oja gate** | **`486`** | **`0.9987`** | **`1.0000`** | **`0.9985`** | **`0.9960`** | **`0.0030`** |

### Key Takeaways from the Ablation Ladder
1. **Replacing post-query `[z,v,a,f]` with the pre-query dual eligibility trace `[c,a]` (Step 0 $\to$ Step 1)** raises Confirmation Case-A both-correct from `0.7520` to `0.9218` (`+16.98pp`) and cuts wrong first-key responses on Query B from `14.98%` to `5.42%`, confirming that post-query somatic shock and fast-decay forgetting (`feature_decay = 0.25`) account for roughlytwo-thirds of R5's clean Case-A errors.
2. **Eliminating the un-gated shared trunk and random Gaussian bottleneck $P$ alongside pre-gating standardization (Step 1/2 $\to$ Step 3)** closes the remaining `7.42pp–8.29pp` Case-A error gap (`0.9218 / 0.9131` $\to$ **`0.9960`** on Confirmation, and `0.8987` $\to$ **`1.0000`** on Dev), reducing wrong first-key responses on Query B to **`0.30%`** (`0.0030`) while using **24.7% fewer learnable parameters than R5** (`390` vs `518`).
