# SCAR-X Round 7 Pre-Registered Experimental Protocol (`protocol-v1`)

- **Campaign**: `scarx-neural-round1-20261111` (Round 7 — Autonomous Hidden Representation, Local Plasticity & Continual Intelligence)
- **Agent / Role**: `antigravity`
- **Model**: `Gemini 3.1 Pro (High)`
- **Baseline Reference**: `495dab2f4bac0180e4d1cd28674391f055ccf09d` (`main`)
- **Research Branch**: `research/gemini/r7-hidden-representation-learning`
- **Core Substrate**: `SpikingCore` (`H = 48`, `fan_in = 8`, `neuron = "alif"`, `leak = 0.85`, `threshold = 0.65`, `adaptation_decay = 0.95`, `adaptation_strength = 0.08`, `feature_decay = 0.92`, hard binary spikes, CPU-first, RAM $\le 2,000,000,000$ bytes)

---

## 1. Scope, Permissions & Hard Invariants

1. **Unchanged Spiking Core**: `src/scarx/neural/core.py` and serving checkpoints are untouched. All forward spike dynamics (`v`, `a`, `z`, `f`) execute identically to `SpikingCore.step(x, track=False)`.
2. **Strictly Local Synaptic Plasticity**:
   - Hidden plasticity (`w_in`, `w_rec`) uses only own-neuron forward sensitivities (`e_v`, `e_a`, `e_f`), local readout error projection (`s_j`), and local homeostatic synaptic trace energy (`M`, `E`).
   - Zero reverse-time graph, zero hidden-to-hidden error transport, zero external dependencies beyond NumPy.
3. **Seed Separation Discipline**:
   - **Historical Audit Seed**: `129001` (used strictly for Stage-0 root-cause dissection).
   - **Pre-Registered DEV Seeds**: `140011, 140017, 140023` ($N_{\text{seeds}} = 3$).
   - **Confirmation Seeds**: Locked and untouched until explicit user review and approval of the DEV screen (`DEV_PROMISING` / `REQUEST_CONFIRMATION`).

---

## 2. P0 Numerical & Architectural Verification Gate (`tests/test_r7_hidden_plasticity.py`)

Before any DEV seed is executed, the candidate implementation (`src/scarx/evaluation/r7_hidden_plasticity.py`) must pass five mandatory P0 checks:
1. **Exact Forward Parity**: `EventWindowedHomeostaticEligibility` produces bit-identical `voltage`, `adaptation`, `spikes`, and `features` to `SpikingCore` across random multi-tick sequences.
2. **Anchored Finite-Difference Local Sensitivity Verification**: At silent-spike subthreshold regimes, analytical own-neuron sensitivities $e_{v,jk}(t)$, $e_{a,jk}(t)$, and $z_{e,jk}(t)$ match central finite differences of single-neuron voltage and soft spike probability (`cosine >= 0.9999`, `max_abs_err < 1e-4`).
3. **Control Identity (`zero_signal == frozen_hidden`)**: Running online training with `condition="zero_signal"` produces bit-identical hidden weights (`w_in`, `w_rec`) and bit-identical test predictions to `condition="frozen_hidden"`.
4. **Bounded State & Memory Budget**: Total array footprint of learner + core + replay buffer remains strictly bounded (`< 250 KB` per instance, process tree RSS `< 2,000,000,000` bytes) with zero history tape growth over sequence count.
5. **Exact Checkpoint Serialization & Continuation**: Saving (`dumps`/`save`) and reloading (`loads`/`load`) mid-stream reproduces bit-identical weights, traces, homeostatic statistics, and predictions across subsequent online learning steps.

---

## 3. P1 Hidden Representation Learning & Zero-Shot OOD Transfer Benchmark

### 3.1 Task Suite (5 Tasks, Each with In-Distribution Test & Zero-Shot OOD Timing/Noise Shift)

All P1 tasks use 12-tick sequences ($T=12$, $D=8$, binary classification $C=2$, `calibration = 128`, `train = 1024`, `probe_fit = 256`, `id_test = 512`, `ood_test = 512`):
1. **`strict_binding`** (Relational Key-Value-Query Counterfactual Binding):
   - Write events at $p_1 \in \{1,2\}$, $p_2 \in \{4,5,6\}$ (OOD shift: $p_1 = 0, p_2 = 7$, lengthening the retention delay). Opposite binary values $(a, 1-a)$ bound to keys $(0, 1)$ in randomized order; query $q \in \{0,1\}$ at tick $10$; target $y = a \oplus q$.
2. **`context_xor_xnor_switch`** (3-Way Relational Rule Switching):
   - Write events at $(p_1, p_2)$ (OOD shift: $p_1 = 0, p_2 = 7$), context cue $c \in \{0,1\}$ on channels $4..5$ at tick $10$, and rule-inversion cue $m \in \{0,1\}$ on channel $6$ at tick $10$; target $y = a \oplus c \oplus m$. All 1st- and 2nd-order marginals have zero correlation with $y$.
3. **`distractor_binding`** (Relational Binding Under Mid-Delay Distractor Pulses & Elevated Noise):
   - Same relational structure as `strict_binding`, but random high-amplitude distractor pulses ($\pm 1.2$) hit channels $6$ and $7$ during delay ticks $\{3, 7, 8\}$ and background Gaussian noise increases from $\sigma = 0.10$ (ID) to $\sigma = 0.14$ (OOD).
4. **`xor`** (Delayed Temporal XOR):
   - Pulses on channels $0..1$ at $p_1 \in \{1,2\}$ and channels $2..3$ at $p_2 \in \{4,5,6\}$ (OOD shift: $p_1 = 0, p_2 = 8$).
5. **`order`** (Temporal Event Order Discrimination):
   - Identical marginal symbol counts with label determined strictly by temporal ordering between $p_1$ and $p_2$ (OOD shift: $p_1 = 0, p_2 = 8$).

### 3.2 Mandatory 8-Condition Control Suite (Section VII)

For every seed and every P1 task, we evaluate:
1. **`initial_hidden`**: Untrained initial hidden weights $(w^{\text{in}}_0, w^{\text{rec}}_0)$, evaluated via `common_refit` ridge probe.
2. **`frozen_hidden`**: Hidden weights frozen at $(w^{\text{in}}_0, w^{\text{rec}}_0)$ while the normalized online readout learns across all 1024 episodes.
3. **`r35_baseline`**: Slow eligibility decay ($\gamma_e = 0.92$) control demonstrating pre-query eligibility cancellation.
4. **`zero_signal`**: Full eligibility tracking active, but hidden credit signal $s_j \equiv 0$ (verifies zero unintended drift).
5. **`shuffled_signal`**: Hidden credit vector $s_j(m)$ randomly permuted across the 48 neurons on every episode (verifies that neuron-specific credit alignment, not random weight perturbation, drives representation gain).
6. **`learned_ew_he` (H1 Primary Candidate)**: Event-Windowed Homeostatic Eligibility ($\gamma_e = 0.25$, online feature standardization, local homeostatic synaptic energy normalization).
7. **`learned_cg_mhe` (H2 Secondary Ablation)**: Coincidence-Gated Multiscale Homeostatic Eligibility ($\chi_j(t)$ somatic-temporal coincidence factor + heterogeneous $\gamma_{e,j} \in [0.15, 0.55]$).
8. **`causal_revert` (Causal Intervention on Learned Hidden Weights)**: Takes the fully trained `learned_ew_he` model, keeps its learned readout $(w^{\text{out}}, b)$ and homeostatic statistics $(\mu_f, \sigma_f)$ intact, and reverts $(w^{\text{in}}, w^{\text{rec}}) \leftarrow (w^{\text{in}}_0, w^{\text{rec}}_0)$ to measure `causal_drop = acc(learned_ew_he) - acc(causal_revert)`.

Every condition reports:
- `id_direct_acc` and `ood_direct_acc` (online learned readout),
- `id_probe_acc` and `ood_probe_acc` (`common_refit` ridge probe fitted on `probe_fit` and evaluated on `id_test` / `ood_test`),
- Representation geometry: Participation Ratio (`PR`) and Fisher Discriminant Ratio (`FDR`).

---

## 4. P2 Continual Multi-Skill & Multi-Context Retention Benchmark

We evaluate continual retention across sequential stages in a single unified `SpikingCore(H=48)`:
1. **Sequential Multi-Skill Continual Stream (`order -> xor -> strict_binding`)**:
   - Three distinct temporal/relational skills learned sequentially in a single `SpikingCore(H=48)` without resetting weights between skills, comparing:
     - `unconsolidated_no_replay` (online `EW-HE` without consolidation or replay),
     - `ew_he_ca_msc` (`EW-HE` + Activity-Gated Local Synaptic Consolidation `CA-MSC` + bounded `ReplayBuffer`).
2. **Metrics & Gate Criteria**:
   - `acquisition_accuracy >= 0.65`,
   - `final_accuracy >= 0.65`,
   - `mean_forgetting <= 0.03`,
   - `worst_forgetting <= 0.05`.
