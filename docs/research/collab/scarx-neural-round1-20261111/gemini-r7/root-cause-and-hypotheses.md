# SCAR-X Round 7 — Root-Cause Analysis & Grounded Hidden-Learning Hypotheses

- **Agent / Role**: `antigravity` (Principal Neural Learning Researcher, Computational Neuroscientist & Algorithm Architect)
- **Model**: `Gemini 3.1 Pro (High)`
- **Baseline Commit**: `495dab2f4bac0180e4d1cd28674391f055ccf09d` (`main`)
- **Branch**: `research/gemini/r7-hidden-representation-learning`
- **Core Architecture**: Unified 1-compartment recurrent ALIF spiking core (`SpikingCore`, $H=48$, sparse recurrence $K=8$, hard binary spikes $z_j(t) \in \{0,1\}$, CPU-first, $\le 2\text{ GB}$ RAM)

---

## I. Structured Root-Cause Analysis of Prior Hidden-Learning Stops

Prior rounds established that two-factor STDP, scalar-reward three-factor eligibility (`PlasticCore` in Phase 2.1), dendritic compartmental changes (`DC-ALIF`), multi-timescale eligibility banks (`MTB`), and own-neuron readout eligibility (`ReadoutEligibility` in Report 35) failed to produce reliable causal hidden representation gains across relational binding and multi-stage continual streams. Below is a rigorous six-part diagnosis grounded in ALIF membrane physics, local sensitivity equations, and empirical measurements on audit seed `129001`.

### 1. Temporal Credit Assignment: The Pre-Query Eligibility Cancellation & Self-Reset Paradox

In `SpikingCore`, neuron $j \in \{1,\dots,48\}$ evolves at tick $t \in \{0,\dots,T-1\}$ according to:
$$
u_j(t) = \lambda v_j(t-1) + \sum_{k=1}^{D} w^{\text{in}}_{jk} x_k(t) + \sum_{r=1}^{K} w^{\text{rec}}_{jr} z_{\text{idx}(j,r)}(t-1), \qquad \theta_j(t) = \theta_0 + \alpha_a a_j(t-1)
$$
$$
z_j(t) = \Theta\big(u_j(t) - \theta_j(t)\big), \qquad v_j(t) = u_j(t) - z_j(t)\theta_j(t), \qquad a_j(t) = \gamma_a a_j(t-1) + z_j(t)
$$
$$
f_j(t) = \gamma_f f_j(t-1) + (1 - \gamma_f) z_j(t)
$$
with $\lambda = 0.85$, $\theta_0 = 0.65$, $\gamma_a = 0.95$, $\alpha_a = 0.08$, and $\gamma_f = 0.92$.

For any synapse $(j,k)$ with presynaptic input $\text{pre}_{jk}(t) \in \{x_k(t), z_{\text{idx}(j,r)}(t-1)\}$, the exact own-neuron forward sensitivities propagate via:
$$
u_{e,jk}(t) = \lambda e_{v,jk}(t-1) + \text{pre}_{jk}(t), \qquad z_{e,jk}(t) = \psi_j(t)\big(u_{e,jk}(t) - \alpha_a e_{a,jk}(t-1)\big)
$$
$$
e_{v,jk}(t) = u_{e,jk}(t) - \theta_j(t) z_{e,jk}(t) - \alpha_a z_j(t) e_{a,jk}(t-1), \qquad e_{a,jk}(t) = \gamma_a e_{a,jk}(t-1) + z_{e,jk}(t)
$$
where $\psi_j(t) = \frac{1}{\tau_s}\sigma\left(\frac{u_j(t)-\theta_j(t)}{\tau_s}\right)\left(1 - \sigma\left(\frac{u_j(t)-\theta_j(t)}{\tau_s}\right)\right)$ ($\tau_s = 0.4$) is the local perisomatic voltage-threshold sensitivity.

**Why Report 35 (`ReadoutEligibility`) failed on `strict_binding` (~62–63%):**
1. **Subthreshold trace $u_{e,jk}(t)$ already bridges the delay**: An early write event at tick $t_1 \in \{1,2\}$ enters $e_{v,jk}(t_1)$ and persists across the delay inside $u_{e,jk}(t_q) = \lambda^{t_q - t_1}\text{pre}_{jk}(t_1) + \dots$ when neuron $j$ remains subthreshold during the delay. At the query tick $t_q = 10$, the instantaneous spike sensitivity $z_{e,jk}(t_q) = \psi_j(t_q)\big(u_{e,jk}(t_q) - \alpha_a e_{a,jk}(t_q-1)\big)$ **directly multiplies the carried write trace $u_{e,jk}(t_q)$ by the post-query somatic slope $\psi_j(t_q)$**.
2. **Pre-query eligibility cancellation when $\gamma_e = 0.92$**: Report 35 accumulated feature eligibility with the slow decay $\gamma_e = \gamma_f = 0.92$:
   $$
   e_{f,jk}(T) = (1 - \gamma_e)\sum_{t=0}^{T} \gamma_e^{T-t} z_{e,jk}(t)
   $$
   When $\gamma_e = 0.92$, early terms $z_{e,jk}(t_1)$ ($t_1 < t_q$) retain weight $0.92^{11 - t_1} \in [0.40, 0.66]$ at $T=11$. However, in any conditional relational task where the label $y = a \oplus q$ depends on the interaction between an early write binding $a \in \{0,1\}$ (at $t_1, t_2$) and a late query $q \in \{0,1\}$ (at $t_q = 10$), the pre-query sensitivity $z_{e,jk}(t_1)$ is statistically independent of $q$. Since $\mathbb{E}[y - \hat{p} \mid a] = 0$, the expected update contributed by pre-query spike sensitivities vanishes:
   $$
   \mathbb{E}_{q}\big[s_j(T)\, z_{e,jk}(t_1) \;\big|\; a\big] = \underbrace{\mathbb{E}_{q}[s_j(T) \mid a]}_{\approx 0} \cdot z_{e,jk}(t_1) = 0
   $$
   while its sample variance $\text{Var}_q[s_j(T) z_{e,jk}(t_1) \mid a] \propto \|z_{e,jk}(t_1)\|^2$ injects severe zero-mean gradient noise into every update.
3. **Hard-reset erasure of subthreshold memory**: Reinforcing pre-query spikes $z_j(t_1) = 1$ triggers the hard voltage reset $v_j(t_1) \leftarrow u_j(t_1) - \theta_j(t_1)$ and subtracts $-\theta_j(t_1) z_{e,jk}(t_1)$ from $e_{v,jk}(t_1)$, **erasing the subthreshold carrier $v_j$ before the query arrives at $t_q = 10$**.
4. **Empirical verification on audit seed `129001` (`strict_binding`)**:
   - At $T=11$, the post-query component $e^{\text{post}}_{f,jk} = \sum_{t=10}^{11} (1-\gamma_f)\gamma_f^{11-t} z_{e,jk}(t)$ has cosine similarity only `0.4870` with total $e_{f,jk}(11)$, because the pre-query component $e^{\text{pre}}_{f,jk}$ ($t < 10$) has **2.22x larger norm** (`0.0956` vs `0.0431`) and cancels true post-query relational credit.
   - Replacing slow $\gamma_e = 0.92$ with a fast event-windowed eligibility filter $\gamma_e = 0.25$ (while keeping membrane leak $\lambda = 0.85$ in $e_v$ and adaptation decay $\gamma_a = 0.95$ in $e_a$) suppresses pre-query spike-reset noise while preserving the subthreshold carrier $u_{e,jk}(t_q)$ inside $z_{e,jk}(t_q)$.

### 2. Recurrent Credit Assignment Across Sparse Fan-In ($K=8$)

- In `SpikingCore`, each neuron $j$ receives recurrent spikes from $K=8$ presynaptic partners $\text{idx}(j,r)$ via $w^{\text{rec}}_{jr}$.
- When a presynaptic partner fires at $t_1$, its spike enters $u_{e,jr}^{\text{rec}}(t_1+1) = z_{\text{idx}(j,r)}(t_1)$ and decays via $\lambda = 0.85$ in $e_{v,jr}^{\text{rec}}(t)$, providing a multi-tick reverberatory bridge between write ticks ($t_1, t_2$) and query tick ($t_q$).
- Under scalar-reward rules (`PlasticCore`), all 48 neurons receive the same scalar advantage $A(T)$, so recurrent synapses $w^{\text{rec}}_{jr}$ cannot know whether neuron $j$ pushed the readout toward or away from the target class.
- Under own-neuron vector readout credit $s_j(T) = \sum_{c} w^{\text{out}}_{cj}(y_c - p_c) / \sigma_{f,j}$, each neuron $j$ receives its own signed target alignment while omitting hidden-to-hidden Jacobian transport ($\partial z_m / \partial z_j$). Empirical ablation on audit seed `129001` confirms that own-neuron vector credit $s_j(T)$ combined with $e_{v,jr}^{\text{rec}}$ increases `strict_binding` ridge probe accuracy from `0.5820` (frozen) to `0.9238` (`+34.18pp`).

### 3. Representation Geometry: Linear Decodability vs. Relational Non-Separability

Empirical dissection of frozen `SpikingCore(H=48)` on audit seed `129001` (`strict_binding`, $N=512$) reveals:
- **Pre-query storage is intact**: At tick $t=9$ (immediately before query $q$ arrives at $t=10$), the stored key-value assignment $a \in \{0,1\}$ is linearly decodable from `f(9)` with **99.80% accuracy** and from `v(9)` with **100.00% accuracy**.
- **Post-query cue is intact**: At tick $t=10$, the query identity $q \in \{0,1\}$ is linearly decodable from `z(10)` and `f(11)` with **100.00% accuracy**.
- **Why frozen `f(11)` fails (`probe = 0.5820–0.6152`)**: Because $f_j(11) = (1-\gamma_f)\sum_{t=0}^{11} \gamma_f^{11-t} z_j(t)$ is a **linear temporal superposition** of pre-query spikes $z_j(t_1, t_2)$ (which depend only on $a$) and post-query spikes $z_j(10, 11)$. In a randomly initialized frozen core, the post-query membrane potential $u_j(10) = \lambda v_j(9) + w^{\text{in}}_j x(10) + \dots$ crosses threshold primarily due to the strong query pulse $w^{\text{in}}_j x(10)$ rather than the subthreshold coincidence of $(a, q)$. Consequently, $z_j(10)$ is dominated by marginal terms $\alpha_j a + \beta_j q$ rather than the multiplicative conjunction $a \wedge q$, and the Parity/XOR target $y = a \oplus q$ has zero correlation with both $a$ and $q$.
- **Quantitative Geometry Shift (Audit Seed `129001`)**:
  - On `strict_binding`: Initial/frozen `f(11)` has Participation Ratio $\text{PR} = 4.36$, top-5 singular values `[0.1726, 0.0665, 0.0554, 0.0529, 0.0482]`, and Fisher Discriminant Ratio $\text{FDR} = 0.2537$. After local `EW-HE` hidden learning, effective rank is preserved ($\text{PR} = 4.27$, top-5 singular values `[0.2481, 0.1342, 0.0919, 0.0916, 0.0791]`) while class-conditional Fisher separation increases **13.66x** to $\text{FDR} = 3.4664$.
  - On `xor`: Initial/frozen `f(11)` has $\text{PR} = 5.05$ and $\text{FDR} = 0.8322$. After local `EW-HE` hidden learning, $\text{FDR}$ increases **13.02x** to $\text{FDR} = 10.8393$ (top singular value expands from `0.1378` to `0.5015`).

### 4. Plasticity Stability: Frozen Calibration Saturation & Timescale Mismatch

Report 35's `ReadoutEligibility` suffered from two additional homeostatic defects:
1. **Frozen calibration saturation (`|raw| < 5` gate shutdown)**: `ReadoutEligibility.calibrate()` computed feature mean $\mu_0$ and variance $\sigma_0^2$ over 128 initial unlabeled sequences and **froze them forever**. As hidden weights $w^{\text{in}}, w^{\text{rec}}$ adapted during training, neuron firing rates shifted so that $|(f_j - \mu_{0,j})/\sigma_{0,j}| \ge 5$, triggering the hard saturation gate `(|raw| < 5)` and **permanently zeroing out the learning signal $s_j$** for the most responsive neurons.
2. **80:1 Readout-to-Hidden Timescale Mismatch**: `ReadoutEligibility` updated `w_out` with learning rate `0.08` (`clip = 0.025`) while restricting hidden updates to `0.001` (`clip = 0.002`). The readout rapidly overfit to the noisy initial features $f^{(0)}$, and when hidden weights slowly rotated $f$, the impatient readout fought the hidden transition.

### 5. Capacity and Inductive Bias of $H=48$ Sparse ALIF

With $H=48$ neurons, $D=8$ inputs, and $K=8$ recurrent fan-in, the core has only $48 \times (8 + 8) = 768$ hidden synapses. Random initial weights partition the 48 neurons poorly: a few low-effective-threshold neurons fire on every tick (saturating adaptation $a_j$), while others remain silent. Local homeostatic synaptic energy normalization equalizes the effective step size across high-gain and low-gain synapses, recruiting all 48 neurons to form balanced coincidence detectors for $(a, q) \in \{0,1\}^2$.

### 6. Readout Interference & Continual Retention Bottleneck

In multi-stage continual streams (`P2`), a shared readout $w^{\text{out}}$ and shared hidden matrices $w^{\text{in}}, w^{\text{rec}}$ experience catastrophic interference when new stages apply dense updates across all synapses:
- **Dense synaptic drift**: Even when an input channel $k$ or hidden unit $j$ is inactive in the current stage ($e_{f,jk} \approx 0$ or $f_j \approx 0$), unmasked momentum traces or global anchor pulls perturb its weights.
- **Unselective consolidation**: Pulling all weights uniformly toward an early stage's anchor $w^{\text{stable}}$ either prevents new skill acquisition (if anchor pull is strong) or fails to protect earlier skills (if anchor pull is weak). Consolidation must be **activity-gated**: synapses active in the current episode are free to adapt toward the current error signal, whereas synapses **inactive** in the current episode ($|e_{f,jk}| \le \tau_e$) are anchored back to their consolidated values $w^{\text{stable}}_{jk}$ weighted by their historical synaptic importance $\Omega_{jk}$, complemented by bounded raw-sequence replay (`ReplayBuffer`).

---

## II. Three Grounded Hidden-Learning Hypotheses (Section V & XIII)

All three hypotheses operate strictly on the unchanged `SpikingCore` 1-compartment ALIF forward dynamics ($H=48$, $K=8$, hard spikes) using 100% pure NumPy local variables and zero non-local graph/history storage.

### Hypothesis H1 — Event-Windowed Homeostatic Eligibility (`EW-HE`) [Selected Primary Mechanism]

1. **Causal Mechanism**:
   Separates the timescale of **subthreshold memory transport** ($\lambda = 0.85$ in $e_v$, $\gamma_a = 0.95$ in $e_a$) from the timescale of **postsynaptic spike-event eligibility accumulation** ($\gamma_e = 0.25$ in $e_f$), tracks online homeostatic feature statistics $(\mu_f, \sigma_f^2)$, and updates hidden synapses via a purely local exponential synaptic trace $M_{jk}$ normalized by local homeostatic synaptic energy $E_{jk}$.
2. **Exact Equations**:
   - **Forward ALIF step (unchanged `SpikingCore`)**:
     $$
     u_j(t) = \lambda v_j(t-1) + \sum_{k=1}^D w^{\text{in}}_{jk} x_k(t) + \sum_{r=1}^K w^{\text{rec}}_{jr} z_{\text{idx}(j,r)}(t-1), \quad \theta_j(t) = \theta_0 + \alpha_a a_j(t-1)
     $$
     $$
     z_j(t) = \Theta(u_j(t) - \theta_j(t)), \quad v_j(t) = u_j(t) - z_j(t)\theta_j(t), \quad a_j(t) = \gamma_a a_j(t-1) + z_j(t)
     $$
   - **Own-neuron sensitivity with Event-Windowed Eligibility ($\gamma_e = 0.25$)**:
     $$
     u_{e,jk}(t) = \lambda e_{v,jk}(t-1) + \text{pre}_{jk}(t), \qquad z_{e,jk}(t) = \psi_j(t)\big(u_{e,jk}(t) - \alpha_a e_{a,jk}(t-1)\big)
     $$
     $$
     e_{v,jk}(t) = u_{e,jk}(t) - \theta_j(t) z_{e,jk}(t) - \alpha_a z_j(t) e_{a,jk}(t-1), \qquad e_{a,jk}(t) = \gamma_a e_{a,jk}(t-1) + z_{e,jk}(t)
     $$
     $$
     e_{f,jk}(t) = \gamma_e e_{f,jk}(t-1) + (1 - \gamma_e) z_{e,jk}(t), \qquad \gamma_e = 0.25
     $$
   - **Online Homeostatic Feature Standardization**:
     $$
     \mu_{f,j}(m) = (1 - \beta_f)\mu_{f,j}(m-1) + \beta_f f_j(T), \quad \sigma_{f,j}^2(m) = (1 - \beta_f)\sigma_{f,j}^2(m-1) + \beta_f (f_j(T) - \mu_{f,j}(m))^2
     $$
     $$
     \tilde{f}_j = \frac{f_j(T) - \mu_{f,j}(m)}{\max(\sqrt{\sigma_{f,j}^2(m)}, 0.05)}, \qquad \hat{f}_j = \text{clip}(\tilde{f}_j, -5, 5), \qquad \beta_f = 0.01
     $$
   - **Local Readout & Pre-Update Own-Neuron Credit Signal**:
     $$
     p_c = \frac{\exp(\sum_j w^{\text{out}}_{cj}\hat{f}_j + b_c)}{\sum_{c'} \exp(\sum_j w^{\text{out}}_{c'j}\hat{f}_j + b_{c'})}, \qquad \delta_c = \mathbb{I}(y = c) - p_c
     $$
     $$
     s_j(m) = \frac{\sum_{c} w^{\text{out}}_{cj}(m-1)\, \delta_c}{\max(\sqrt{\sigma_{f,j}^2(m)}, 0.05)} \cdot \mathbb{I}(|\tilde{f}_j| < 5), \qquad g_{jk}(m) = s_j(m)\, e_{f,jk}(T)
     $$
     *(Note: $s_j(m)$ is evaluated using the pre-update readout weights $w^{\text{out}}(m-1)$ prior to applying $\Delta w^{\text{out}}(m) = \eta_{\text{out}} \delta \hat{f}^\top$, avoiding the label-independent self-alignment bias $+\eta_{\text{out}}\|\delta\|^2 \hat{f}_j$.)*
   - **Pure Local Homeostatic Synaptic Trace Update**:
     $$
     M_{jk}(m) = \beta_M M_{jk}(m-1) + (1 - \beta_M) g_{jk}(m), \qquad E_{jk}(m) = \beta_E E_{jk}(m-1) + (1 - \beta_E) g_{jk}^2(m)
     $$
     $$
     \Delta w_{jk}(m) = \text{clip}\!\left(\eta_{\text{hid}} \frac{M_{jk}(m)}{\sqrt{E_{jk}(m)} + \epsilon_E},\; -\Delta_{\max},\; \Delta_{\max}\right)
     $$
     with $\beta_M = 0.90$, $\beta_E = 0.99$, $\epsilon_E = 10^{-4}$ ($E_{jk}(0) = 10^{-4}$), $\eta_{\text{hid}} = 0.008$, $\Delta_{\max} = 0.01$, and $\eta_{\text{out}} = 0.04$.
3. **Why it differs from STOPPED methods**:
   Unlike DC-ALIF (which altered core neuron compartments) and MTB (which duplicated multi-timescale eligibility banks), `EW-HE` uses a single $(H, D+K)$ eligibility state, preserves `SpikingCore` forward dynamics 100%, resolves the pre-query cancellation paradox via $\gamma_e = 0.25$, and replaces frozen calibration with online homeostatic feature and synaptic energy regulation.
4. **Why frozen hidden fails where H1 succeeds**:
   Frozen `f(11)` superimposes pre-query and post-query spikes linearly without $(a, q)$ coincidence selectivity; `EW-HE` drives hidden weights $w^{\text{in}}, w^{\text{rec}}$ to align subthreshold write traces $u_{e,jk}(t_q)$ with post-query somatic threshold crossings $\psi_j(t_q)$, creating nonlinear conjunction detectors in `f(11)`.
5. **Explicit Falsifier**:
   Falsified if on the DEV screen (`seeds = 140011, 140017, 140023`), `EW-HE` fails to beat `frozen_hidden`, `r35_baseline`, and `shuffled_signal` by at least `+15pp` on `strict_binding` and `context_xor_xnor_switch`, or if `causal_revert` (reverting $w^{\text{in}}, w^{\text{rec}} \leftarrow w^{\text{in}}_0, w^{\text{rec}}_0$) fails to cause a large accuracy drop ($\ge 15\text{pp}$).
6. **Memory and Compute Cost**:
   Three $(48, 16)$ per-episode arrays (`e_v, e_a, e_f`) + two $(48, 16)$ synaptic state arrays (`M, E`) = $5 \times 48 \times 16 \times 4 = 15,360\text{ bytes}$ ($\approx 15\text{ KB}$, $< 0.001\%$ of the $2\text{ GB}$ limit), $O(T \cdot H \cdot (D+K))$ operations per episode.

### Hypothesis H2 — Coincidence-Gated Multiscale Homeostatic Eligibility (`CG-MHE`)

1. **Causal Mechanism**:
   Augments `EW-HE` with a local somatic-temporal coincidence factor $\chi_j(t)$ that explicitly amplifies $z_{e,jk}(t)$ when carried subthreshold/recurrent state $u_{\text{carry},j}(t) = \lambda v_j(t-1) + i_{\text{rec},j}(t)$ and instantaneous sensory drive $i_{\text{in},j}(t) = (w^{\text{in}} x(t))_j$ are simultaneously non-zero, paired with heterogeneous eligibility decays $\gamma_{e,j} \in [0.15, 0.55]$ across $j \in \{1,\dots,48\}$.
2. **Exact Equations**:
   $$
   m_{\text{carry},j}(t) = |u_{\text{carry},j}(t)| + \theta_0 \alpha_a \bar{e}_{a,j}(t-1), \qquad m_{\text{inst},j}(t) = |i_{\text{in},j}(t)|
   $$
   $$
   \chi_j(t) = \frac{2\, m_{\text{carry},j}(t)\, m_{\text{inst},j}(t)}{m_{\text{carry},j}^2(t) + m_{\text{inst},j}^2(t) + 0.05} \in [0, 1]
   $$
   $$
   z^{\text{CG}}_{e,jk}(t) = \big(0.25 + 0.75\,\chi_j(t)\big)\, z_{e,jk}(t), \qquad e_{f,jk}(t) = \gamma_{e,j} e_{f,jk}(t-1) + (1 - \gamma_{e,j}) z^{\text{CG}}_{e,jk}(t)
   $$
3. **Explicit Falsifier**:
   Falsified as an addition over H1 if `CG-MHE` does not improve mean accuracy across the 5 P1 tasks by at least `+1.5pp` over simpler `EW-HE` (Occam's razor pruning rule: any module that does not earn its complexity is rejected).

### Hypothesis H3 — Context-Anchored Metaplastic Synaptic Consolidation (`CA-MSC`) [Integrated with H1 for P2 Continual Learning]

1. **Causal Mechanism**:
   Extends `EW-HE` to continual multi-stage / multi-skill streams by gating synaptic updates with an **episode activity mask** $A_{jk}(m) = \mathbb{I}(|e_{f,jk}(T)| > \tau_e)$ so that inactive synapses do not drift under momentum, while applying local importance-weighted elastic consolidation $\Omega_{jk} (w^{\text{stable}}_{jk} - w_{jk})$ selectively to inactive synapses alongside bounded raw-sequence replay (`ReplayBuffer`).
2. **Exact Equations**:
   $$
   A_{jk}(m) = \mathbb{I}\big(|e_{f,jk}(T)| > 10^{-4}\big), \qquad \hat{s}_{jk}(m) = \frac{A_{jk}(m)\, M_{jk}(m)}{\sqrt{E_{jk}(m)} + \epsilon_E}
   $$
   $$
   \Delta w_{jk}(m) = \text{clip}\!\left(\frac{\eta_{\text{hid}}}{1 + \lambda_\Omega \Omega_{jk}} \hat{s}_{jk}(m) + \kappa_{\text{hid}} \Omega_{jk} (1 - A_{jk}(m))\big(w^{\text{stable}}_{jk} - w_{jk}\big),\; -\Delta_{\max},\; \Delta_{\max}\right)
   $$
   $$
   \Omega_{jk}(m) = \min\!\big(\Omega_{\max},\; \gamma_\Omega \Omega_{jk}(m-1) + \alpha_\Omega |\hat{s}_{jk}(m)|\big)
   $$
3. **Explicit Falsifier**:
   Falsified if on the P2 continual streams, `EW-HE + CA-MSC` fails to achieve `acquisition >= 0.65`, `final_accuracy >= 0.65`, `mean_forgetting <= 0.03`, and `worst_forgetting <= 0.05`, or fails to reduce forgetting by at least `10pp` relative to unconsolidated online learning.

---

## III. Hypothesis Ranking & Experimental Selection

| Rank | Hypothesis | Role in Round 7 | Rationale |
| :---: | :--- | :--- | :--- |
| **1** | **H1 (`EW-HE`) + H3 (`CA-MSC`)** | **Primary Candidate** | Minimal, mathematically exact resolution of the pre-query eligibility cancellation and calibration-freeze bottlenecks in P1, unified with activity-gated local synaptic consolidation + bounded replay in P2. |
| **2** | **H2 (`CG-MHE`)** | **Secondary Ablation** | Tested side-by-side against H1 on P1 to determine whether explicit somatic-temporal coincidence gating $\chi_j(t)$ adds value beyond H1's natural threshold sensitivity $\psi_j(t) u_{e,jk}(t)$. |
