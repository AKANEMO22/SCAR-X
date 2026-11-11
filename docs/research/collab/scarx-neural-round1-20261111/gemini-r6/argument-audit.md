# SCAR-X Round 6C — Argument Audit & Hostile Reviewer Defense (`argument-audit.md`)

**Agent / Role:** `antigravity` — Principal Computational Neuroscientist, Mathematical Learning Theorist, Experimental Engineer  
**Environment:** `Antigravity` (Windows 11, CPU-first, Python 3.11.9)  
**Model:** `Gemini 4 Argon (High)`  
**Branch:** `research/gemini/r6-local-associative-learning`  
**Baseline Reference:** `main` @ `495dab2f4bac0180e4d1cd28674391f055ccf09d` (incorporating Grok R5A @ `b0480c9c` and DeepSeek R5B @ `d6c609df`)  
**Pre-Registration Commit:** `4b62eddb6a26d1786557121bd05b4f69205ef2bc`

---

## 1. Purpose of This Audit

Every scientific claim in `report.md` and `math-analysis.md` is audited below against the standard objections of a hostile computational neuroscience / machine learning reviewer. For each claim, we state:
1. **Claim**: The exact mathematical or empirical statement made in the Round 6C report.
2. **Hostile Critique / Alternative Explanation**: How a skeptical reviewer could try to explain away the result (e.g., parameter count, training optimization, hidden shortcut, hard-coded cue, timing dependence, or uncontrolled model selection).
3. **Mathematical Proof & Experimental Control Evidence**: The exact theorem in `math-analysis.md` and the exact controlled comparison in `artifacts/gemini-r6/stage0-audit.json`, `dev-screen.json`, or `confirmation.json` that refutes the alternative explanation.
4. **Residual Risk & Scope Boundary**: What the result does *not* prove and where future work is required.

---

## 2. Claim-by-Claim Audit Matrix

### Claim C1: R5B's counterfactual failure (`both_A = 0.6676` in R5B; `0.7520` on 8 R6 seeds) is caused by cross-branch SGD interference and exponentially attenuated early-write projections (`lambda = 0.86^5 = 0.4704`), not by missing information in the 48-ALIF recurrent core or by lack of total parameter count.

- **Hostile Critique / Alternative Explanation:**
  - *Critique 1a:* Maybe the 48-neuron ALIF core simply loses the first key's value $v_A$ by $t_q = 10$, or $v_A$ and $v_B$ are nonlinearly entangled in $m_{\text{pre}}$.
  - *Critique 1b:* Maybe R5's online SGD just needed more epochs, a tuned learning rate, or batch ridge regression (`Ridge-Bilinear`) on $[m_{\text{pre}}, c m_{\text{pre}}]$ to reach 99% `both_A`.
  - *Critique 1c:* Maybe CKCER only beats R5 because CKCER has more parameters.
- **Mathematical Proof & Experimental Control Evidence:**
  - *Refuting 1a (Linear Decodability):* Stage 0 Audit (`stage0-audit.json`) proves that two independent linear ridge probes trained on $m_{\text{pre}}$ decode $v_A$ at **$0.9963 \pm 0.0040$** and $v_B$ at **$0.9990 \pm 0.0020$** (joint **$0.9953 \pm 0.0052$**). Furthermore, the exact analytical 2-weight subtraction of those two probes (`split_probe`: $w_0 = w_A$, $w_1 = w_B - w_A$) inside R5's own $[m_{\text{pre}}, c m_{\text{pre}}]$ space achieves **`direct = 0.9963`, `both_A = 0.9912`** (`stage0-audit.json`). Thus, the 48-ALIF core preserves both facts with $>99.5\%$ linear fidelity, and R5's hypothesis class *contains* a 99.1% solution.
  - *Refuting 1b (Optimization / Ridge Control):* Theorem 1 (`math-analysis.md`) proves that R5's shared trunk $w_0$ receives conflicting gradient updates on every $q=1$ trial ($g = [x, x]$), yielding a $2\times 2$ block coupling matrix $M = \begin{pmatrix} 1 & 1/2 \\ 1/2 & 1/2 \end{pmatrix}$ with condition number $\kappa(M) = 3 + 2\sqrt{2} \approx 5.828$, compounded by $\lambda^2 = 0.86^{10} \approx 0.2213$ attenuation of $v_A$ in $m_{\text{pre}}$, giving $\kappa(\Sigma_{\text{R5}}) > 26$. In Stage 3 Confirmation (`confirmation.json`), closed-form L2 batch regression on $[m_{\text{pre}}, c m_{\text{pre}}]$ (`Ridge-Bilinear`) achieves only `direct = 0.9165 ± 0.0229`, `both_A = 0.6744 ± 0.0928` (`clean`), and collapses to `both_A = 0.4316` on `distractors` and `0.5065` on `noise_change`. Thus, replacing online SGD with closed-form batch Ridge on R5's feature map does *not* fix the problem.
  - *Refuting 1c (Parameter-Matched Control):* Both R5 and CKCER Mode A have **exactly $2 \times 98 = 196$ learnable readout weights** (`ParamMatched-R5-196` vs `CKCER-ModeA`). On `clean` confirmation (`confirmation.json`), `ParamMatched-R5-196` gets `both_A = 0.7520 ± 0.1570` while `CKCER-ModeA` (196 weights) gets **`both_A = 0.9960 ± 0.0114`** ($\Delta = +24.40\text{pp}$, `8/8` seed wins). Adding a second 98-weight linear layer to R5 (`ParamMatched-R5-392`, 392 weights) still only reaches `both_A = 0.8180 ± 0.1190`.
- **Residual Risk & Scope Boundary:**
  - The 48-ALIF core is held frozen during R6 readout learning to isolate the readout and eligibility mechanisms cleanly and protect `sim_xor`, `xor_separated`, and `order_task`. Co-adapting recurrent weights with CKCER eligibility traces is left for future investigation.

---

### Claim C2: Both orthogonal competitive routing ($g \odot (1-g) = 0$) AND the dual-timescale eligibility trace ($[e_{\text{slow}}, e_{\text{fast}}]$) are necessary for robust associative retrieval across clean and perturbed regimes; neither component alone suffices across all stress conditions.

- **Hostile Critique / Alternative Explanation:**
  - *Critique 2a:* Maybe orthogonal routing `(1-g, g)` on $m_{\text{pre}}$ alone (`Ablation-1-OrthogonalOnly`) is already enough, and the dual-timescale eligibility trace $[e_{\text{slow}}, e_{\text{fast}}]$ is decorative complexity.
  - *Critique 2b:* Conversely, maybe the $[e_{\text{slow}}, e_{\text{fast}}]$ trace with R5's uncentered `[1, g]` routing (`Ablation-2-DualTraceSharedTrunk`) is doing all the work, and orthogonal routing `(1-g, g)` is unnecessary.
- **Mathematical Proof & Experimental Control Evidence:**
  - *Refuting 2a (`Ablation-1-OrthogonalOnly` vs `CKCER`):* In the Stage 3 Confirmation single-factor ablation ladder (`confirmation.json`, 8 seeds), `Ablation-1-OrthogonalOnly` (which applies orthogonal routing `[(1-g)m_pre, g m_pre]` directly to $m_{\text{pre}}$ without $e_{\text{slow}}$) improves `clean` `both_A` from `0.7520` (`R5`) to `0.9571 ± 0.0258` by eliminating cross-branch gradient coupling ($\kappa(M): 5.828 \to 1.0$). However, because $m_{\text{pre}}$ still suffers $\lambda = 0.86^5 = 0.4704$ exponential decay of the first write, `Ablation-1-OrthogonalOnly` degrades under every temporal/interference perturbation:
    - `distractors` `both_A`: `0.8168 ± 0.0566` (`Ablation-1`) vs **`0.9733 ± 0.0176`** (`CKCER`, $+15.65\text{pp}$, `8/8` wins)
    - `paired_same_sample_distractors` `both_A`: `0.7796 ± 0.0537` (`Ablation-1`) vs **`0.9658 ± 0.0226`** (`CKCER`, $+18.62\text{pp}$, `8/8` wins)
    - `noise_change` `both_A`: `0.7707 ± 0.0430` (`Ablation-1`) vs **`0.9368 ± 0.0185`** (`CKCER`, $+16.61\text{pp}$, `8/8` wins)
  - *Refuting 2b (`Ablation-2-DualTraceSharedTrunk` vs `CKCER`):* `Ablation-2-DualTraceSharedTrunk` (which feeds $[e_{\text{slow}}, e_{\text{fast}}]$ into R5's uncentered `[1, g]` trunk) still suffers shared-trunk gradient interference on $q=1$ updates, reaching only `both_A = 0.8732 ± 0.0985` on `clean` (vs **`0.9960 ± 0.0114`** for `CKCER`, $+12.28\text{pp}$), `0.6928` on `distractors` (vs **`0.9733`**, $+28.05\text{pp}$), and `0.6591` on `noise_change` (vs **`0.9368`**, $+27.77\text{pp}$).
- **Residual Risk & Scope Boundary:**
  - The dual-timescale eligibility trace uses two pre-registered passive decay constants ($\gamma_{\text{slow}} = 0.96, \gamma_{\text{fast}} = 0.70$) for $K=2$ writes. Scaling to $K \gg 2$ sequential writes over long horizons will require a logarithmic bank of decay rates $\gamma_1 < \dots < \gamma_M$ or content-addressable synaptic tags.

---

### Claim C3: CKCER does not rely on an externally engineered query cue; Mode B discovers the binary query context autonomously from the raw pre- vs post-query neural transition $\Delta_t = z_{\text{post}} - m_{\text{pre}}$ via local unsupervised Oja PCA (`query_cue_external_dependency = False`).

- **Hostile Critique / Alternative Explanation:**
  - *Critique 3:* R5's `query_cue` (`float(np.mean(x_query[:3]) > np.mean(x_query[3:6]))`) is a hand-crafted scalar extracted directly from the input vector $x_{\text{query}}$ outside the neural core. If CKCER only worked in Mode A, a critic could argue that the hard routing step $g \in \{0,1\}$ is still supplied by hand from the input channels.
- **Mathematical Proof & Experimental Control Evidence:**
  - Theorem 4 (`math-analysis.md`) proves that during the query tick $t_q$, the two query keys drive orthogonal input currents into the 48 ALIF neurons, producing a dominant rank-1 separation along the first principal eigenvector $u_1$ of $\operatorname{Cov}(\hat{\Delta}_t)$, where $\Delta_t = z_{\text{post}} - m_{\text{pre}} \in \mathbb{R}^{48}$ depends *strictly* on the recurrent neural core's pre- and post-query states (`capture_r6` in `src/scarx/evaluation/r6_associative.py`).
  - Mode B (`CKCER-ModeB`) learns $w_q \in \mathbb{R}^{48}$ purely via local unsupervised Oja updates ($\Delta w_q = \eta_q y_q (\hat{\Delta}_t - y_q w_q)$) and winner-take-all sign selection $g_t = \mathbb{I}(w_q^\top \hat{\Delta}_t > 0)$, with **zero access to `query_cue` or raw input channels `x_query`**.
  - Unit test `test_hand_computed_oja_update_and_cue_blindness_mode_b` in `tests/test_r6_associative.py` explicitly Corrupts/Flips `query_cue` to random values (`1.0 - cue` or `999.0`) during Mode B inference and verifies that Mode B predictions are **bit-for-bit identical (`np.array_equal`)**.
  - Across **all 3 dev seeds (`120011..120023`) and all 8 confirmation seeds (`121011..121043`)**, Mode B achieves **100.00% unsupervised gate purity (`1.0000 ± 0.0000`)** and matches Mode A to within $0.0000$ on `clean` (`direct = 0.9985`, `both_A = 0.9960`), `ood_clean_to_shift` (`both_A = 0.9959`), `reversed_writes` (`both_A = 0.9941`), `distractors` (`both_A = 0.9733`), and within $0.09\text{pp}$ on `noise_change` (`0.9359` vs `0.9368`).
- **Residual Risk & Scope Boundary:**
  - Because unsupervised PCA is sign-invariant ($\pm u_1$ are both valid unit eigenvectors), which branch (`0` or `1`) receives $q=0$ vs $q=1$ is arbitrary up to permutation; however, because both readout branches $(W^{(0)}, W^{(1)})$ are symmetrically initialized and trained from the same reward/target signal $y^*$, branch permutation is completely transparent to output predictions. Extending Mode B from binary ($K=2$) to $K > 2$ keys requires $K$-unit competitive Oja/Sanger subspaces rather than a single thresholded eigenvector.

---

### Claim C4: CKCER's performance is invariant to write-tick shifts (`ood_clean_to_shift`, `train_shift_test_shift`), reversed write orders (`reversed_writes`), and query-tick delays (`query_tick_variation`), refuting any fixed-tick memorization hypothesis.

- **Hostile Critique / Alternative Explanation:**
  - *Critique 4:* Maybe $[e_{\text{slow}}, e_{\text{fast}}]$ only works when Key A is written at tick $t_1 = 1$, Key B is written at tick $t_2 = 5$, and the query arrives at tick $t_q = 10$ in fixed $A \to B$ order.
- **Mathematical Proof & Experimental Control Evidence:**
  - Theorem 2 (`math-analysis.md`) proves that for *any* distinct write ticks $0 \le t_1 < t_2 < t_q$, the temporal Vandermonde matrix $V(t_1, t_2) = \begin{pmatrix} \gamma_{\text{slow}}^{t_q - 1 - t_1} & \gamma_{\text{slow}}^{t_q - 1 - t_2} \\ \gamma_{\text{fast}}^{t_q - 1 - t_1} & \gamma_{\text{fast}}^{t_q - 1 - t_2} \end{pmatrix}$ has strictly positive determinant $\det V(t_1, t_2) = \gamma_{\text{slow}}^{d_2}\gamma_{\text{fast}}^{d_2}\left(\gamma_{\text{slow}}^{d_1 - d_2} - \gamma_{\text{fast}}^{d_1 - d_2}\right) > 0$ because $\gamma_{\text{slow}} = 0.96 > \gamma_{\text{fast}} = 0.70$ and $d_1 > d_2$.
  - Furthermore, because Key A and Key B project onto distinct input channels of the 48-ALIF recurrent core, `[e_slow, e_fast]` preserves both spatial (key-channel) and temporal (order/horizon) coordinates simultaneously.
  - Empirical confirmation (`confirmation.json`, 8 seeds):
    - **Zero-shot timing shift (`ood_clean_to_shift`, trained on $(t_1=1, t_2=5)$, tested zero-shot on $(t_1=2, t_2=6)$):** `CKCER` achieves **`direct = 0.9990 ± 0.0014`, `both_A = 0.9959 ± 0.0082`** (vs `R5` `both_A = 0.7580 ± 0.1450`, $\Delta = +23.80\text{pp}$, `8/8` wins).
    - **Shifted training (`train_shift_test_shift`):** `CKCER` achieves **`direct = 0.9995 ± 0.0010`, `both_A = 0.9960 ± 0.0080`** (vs `R5` `both_A = 0.7745 ± 0.1407`, $\Delta = +22.15\text{pp}$, `8/8` wins).
    - **Randomly interleaved write order (`reversed_writes`, $A \to B$ and $B \to A$ mixed):** `CKCER` achieves **`direct = 0.9985 ± 0.0021`, `both_A = 0.9941 ± 0.0119`** (vs `R5` `both_A = 0.7459 ± 0.1510`, $\Delta = +24.81\text{pp}$, `8/8` wins).
    - **Query-tick variation (`query_tick_variation`, $t_q = 11$):** `CKCER` achieves **`direct = 0.9985 ± 0.0028`, `both_A = 0.9960 ± 0.0114`** (vs `R5` `both_A = 0.6991 ± 0.1494`, $\Delta = +29.69\text{pp}$, `8/8` wins).
- **Residual Risk & Scope Boundary:**
  - Zero-shot transfer was tested for a 1-tick shift ($(1,5) \to (2,6)$) where the relative inter-write interval $t_2 - t_1 = 4$ is preserved and overall attenuation shifts by a modest scalar factor ($\gamma_{\text{slow}}^1 = 0.96$). Extreme zero-shot dilation (e.g., training on $T=12$ and testing zero-shot on $T=100$ without scale normalization) would attenuate traces beyond fixed thresholds unless layer-normalized or event-gated.

---

### Claim C5: Eligibility windowing and competitive routing protect CKCER against mid-delay distractors (`distractors`, `paired_same_sample_distractors`), noise doubling (`noise_change`), and structured third-write distractors (`aux_structured_third_write_windowed`).

- **Hostile Critique / Alternative Explanation:**
  - *Critique 5:* Passive exponential traces integrate all pre-query activity, so injecting distractor pulses on channels `6:12` at ticks `3, 7, 8` or doubling membrane noise ($\sigma: 0.08 \to 0.16$) should corrupt $e_{\text{slow}}$ and $e_{\text{fast}}$ just as badly as R5's $m_{\text{pre}}$.
- **Mathematical Proof & Experimental Control Evidence:**
  - In `R5`, cross-branch gradient interference ($\kappa(M) \approx 5.83$) means that distractor-induced variance on $q=1$ trials directly perturbs the shared trunk $w_0$, causing catastrophic collapse on `distractors` (`both_A = 0.4845 ± 0.1059`) and `paired_same_sample_distractors` (`both_A = 0.4342 ± 0.1088`), where R5's `wrong_first_key_qB_diff` spikes to `0.2962` and `0.3204`.
  - In `CKCER`, orthogonal branch isolation ($g \odot (1-g) = 0$) prevents $q=1$ updates from corrupting $W^{(0)}$, while the slow trace $e_{\text{slow}}$ ($\gamma_{\text{slow}} = 0.96$) boosts the signal-to-noise ratio of the early write at $t_1 = 1$ by $(0.96^8 / 0.86^5)^2 = (0.7214 / 0.4704)^2 \approx 2.35\times$ relative to $m_{\text{pre}}$.
  - Empirical confirmation (`confirmation.json`, 8 seeds):
    - `distractors`: **`direct = 0.9885 ± 0.0061`, `both_A = 0.9733 ± 0.0176`**, `wrong_first_key_qB_diff = 0.0148` (vs `R5` `both_A = 0.4845`, $\Delta = +48.88\text{pp}$, `8/8` wins).
    - `paired_same_sample_distractors` (identical distractor realization across $(q=0, q=1)$ counterfactual pairs): **`direct = 0.9885 ± 0.0079`, `both_A = 0.9658 ± 0.0226`** (vs `R5` `both_A = 0.4342`, $\Delta = +53.17\text{pp}$, `8/8` wins).
    - `noise_change` ($\sigma: 0.08 \to 0.16$): **`direct = 0.9775 ± 0.0056`, `both_A = 0.9368 ± 0.0185`** (vs `R5` `both_A = 0.4906`, $\Delta = +44.63\text{pp}$, `8/8` wins).
    - `aux_structured_third_write_windowed` (adversarial write-like pulse at $t=7$ on channels `6:10`, using event-gated eligibility accumulation `eligibility_windowed=True`): **`direct = 0.9919 ± 0.0057`, `both_A = 0.9778 ± 0.0145`** (vs `R5` `both_A = 0.5396`, $\Delta = +43.82\text{pp}$, `8/8` wins).
- **Residual Risk & Scope Boundary:**
  - On the second auxiliary stress task `aux_key_overwrite` (where the *same* key A is written at $t=1$ with $\bar{v}_A$ and overwritten at $t=4$ with $v_A$, requiring recency-based discounting of the earlier write to the *same* key), `CKCER` achieves `direct = 0.9397 ± 0.0124`, `both_A = 0.8503 ± 0.0442` (still $+32.28\text{pp}$ over `R5`'s `0.5275`, `8/8` wins), because $e_{\text{slow}}$ retains a residual trace of the stale $t=1$ write. True key-specific overwrite erasure requires a write-gated reset $e_{\text{slow}} \leftarrow (1 - \alpha_{\text{write}}) e_{\text{slow}}$ conditioned on key identity.

---

### Claim C6: There was zero uncontrolled model selection, zero post-hoc threshold tuning, and zero regression on protected SCAR-X benchmarks (`sim_xor`, `xor_separated`, `order_task`).

- **Hostile Critique / Alternative Explanation:**
  - *Critique 6a:* Maybe multiple candidate architectures (Approach A, Approach B, Approach C) were secretly run on the confirmation benchmark and the highest scorer was picked post-hoc.
  - *Critique 6b:* Maybe CKCER degrades previously working SCAR-X tasks (`sim_xor`, `xor_separated`, `order_task`), repeating the earlier DC-ALIF failure mode.
- **Mathematical Proof & Experimental Control Evidence:**
  - *Refuting 6a (Strict Pre-Registration Audit Trail):* Commit `4b62eddb6a26d1786557121bd05b4f69205ef2bc` on branch `research/gemini/r6-local-associative-learning` committed `theory.md` (with the mathematical derivation and rejection of Approaches A and B prior to any R6 experiment) and `protocol-v1.md` (locking all hyperparameters $\gamma_{\text{slow}}=0.96, \gamma_{\text{fast}}=0.70, \text{lr}=0.08$, seeds `120011..120023` and `121011..121043`, and acceptance gates G1–G10) **before** `src/scarx/evaluation/r6_associative.py` was written or executed. Approaches A and B were tested *only* as part of the single-factor ablation ladder (`Ablation-1-OrthogonalOnly` and `Ablation-2-DualTraceSharedTrunk`) required by Section VIII of the mission prompt.
  - *Refuting 6b (Protected Task Parity & Core Isolation):* `src/scarx/neural/core.py` and `src/scarx/neural/unified.py` have **zero lines modified** (`git diff main -- src/scarx/neural/` is empty), and DC-ALIF was not touched or revived. On all 8 confirmation seeds (`confirmation.json`), CKCER preserves full parity on all three protected benchmarks:
    - `sim_xor`: `0.9932 ± 0.0110` vs `0.9934 ± 0.0113` ($\Delta = -0.0002$, tolerance $-0.010$)
    - `xor_separated`: `0.6270 ± 0.0493` vs `0.6282 ± 0.0566` ($\Delta = -0.0012$, tolerance $-0.010$)
    - `order_task`: `0.9883 ± 0.0079` vs `0.9880 ± 0.0080` ($\Delta = +0.0002$, tolerance $-0.010$)
- **Residual Risk & Scope Boundary:**
  - Protected tasks were evaluated using the canonical `tasks21` generator where there is no multi-key query switch ($g_t = 0$, routing purely through Branch 0). Multi-task continual interleaving of `sim_xor` and `r6_associative` within the same weight matrix is outside the Round 6C protocol scope.
