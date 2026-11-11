# SCAR-X Round 6C Scientific Report (`gemini-r6`)
## First-Principles Local Associative Learning and Robust Query Retrieval via Context-Keyed Competitive Eligibility Readout (`CKCER`)

- **Campaign**: `scarx-neural-round1-20261111` (SCAR-X Round 6C)
- **Agent / Environment / Model**: `antigravity` / `Antigravity` (Windows 11, CPU-first, Python 3.11.9) / `Gemini 4 Argon (High)`
- **Branch**: `research/gemini/r6-local-associative-learning`
- **Base Commit (`main`)**: `495dab2f4bac0180e4d1cd28674391f055ccf09d`
- **Pre-Registered Protocol & Theory Commit**: `4b62edd` (committed prior to any scored dev or confirmation runs)
- **Historical References Audited**:
  - Grok R5A: `b0480c9cc02a3f35ba994480df2a7d984b6f89f1`
  - DeepSeek R5B: `d6c609df6f90fc8df96fb3de424dd54388f8dc59`
- **Final Verdict**: **GO (`GO_CONFIRMATION` — 10/10 pre-registered gates passed on both 3 Dev seeds and 8 Confirmation seeds)**

---

## Executive Summary

DeepSeek R5B (`d6c609df6f90fc8df96fb3de424dd54388f8dc59`) proved that a frozen 48-neuron ALIF core linearly encodes two independent stored values $v_A, v_B \in \{0, 1\}$ with `0.9967` / `0.9974` ridge decodability before the query arrives, yet the R5 local bilinear readout achieved only **`0.6676`** both-answers-correct on the hard counterfactual Case A ($v_A \neq v_B$) (`0.7520` on the fresh Round 6 confirmation seed bank), collapsing to **`0.4536` / `0.4845`** under distractors and **`0.4512` / `0.4906`** under noise changes.

From first principles, we proved (**Theorem 1 in `theory.md`**) that counterfactual sibling accuracy on Case A ($v_A \neq v_B$) is governed by the exact margin inequality:
$$(v_A - v_B)\, z_{\text{diff}}(x) > |z_{\text{com}}(x)|$$
and diagnosed three exact structural causes of R5's failure:
1. **Un-gated Shared Trunk Dominance (`192` dims vs `32` gated dims)**: $74.4\%$ of R5's parameters sit in an un-gated shared trunk $W_s s$ that updates on $100\%$ of training examples toward the query-blind average $\frac{1}{2}(v_A + v_B)$, contributing **zero** to the query-switching margin $z_{\text{diff}}(x)$ while injecting high-variance common-mode interference into $z_{\text{com}}(x)$ whenever $v_A \neq v_B$.
2. **Post-Gating Z-Score Cross-Talk (`L2 norm = 4.3856`)**: R5 standardizes features *after* multiplying by the binary query gate $c_k(q) \in \{0, 1\}$. Because $\mathbb{E}[(P^\top s)c_k(q)] \neq \mathbf{0}$, the **gated-off** compartment ($c_k(q) = 0$) has a dense non-zero standardized value $-\mu_{g_k}/\sigma_{g_k}$ (`L2 norm = 4.3856` on historical R5B seeds), causing every Query B training update to corrupt Query A's weights and vice versa.
3. **Post-Query Somatic Overwrite & Fast-Decay Forgetting**: Reading post-query `[z, v, a, f]` at tick $t_q$ exposes the state to the query pulse's somatic shock and subthreshold membrane noise $v(t_q)$, while `features` $f(t_q)$ ($\gamma_f = 0.25$) attenuates the tick-2 write by $0.25^8 = 1.52 \times 10^{-5}$.

To solve all three flaws from first principles without reviving DC-ALIF or modifying `SpikingCore`, we derived and pre-registered **Context-Keyed Competitive Eligibility Readout (`CKCER`)**, which combines:
- A bounded **pre-query dual-timescale eligibility trace** $m(t_q - 1) = [c(t_q - 1); \, a(t_q - 1)] \in \mathbb{R}^{96}$ ($\gamma_c = 1.0$ spike count + $\gamma_a = 0.95$ native ALIF adaptation trace, capturing the 0-th and 1-st temporal moments while filtering subthreshold voltage noise and query-pulse shock).
- **Pre-gating variance-safe homeostatic standardization** ($\epsilon_{\text{dead}} = 10^{-4}$) applied *before* multiplicative dendritic gating, reducing inactive-branch cross-talk from `4.3856` to **`0.0000` (exact zero)**.
- **Purely context-keyed dendritic compartments** $\psi(\hat{m}, g) = [g_0 \hat{m}; \, g_1 \hat{m}; \, g_0; \, g_1] \in \mathbb{R}^{194}$ with **zero un-gated shared trunk** and **zero random projection matrices** (**`390` learnable parameters in Mode A — $24.7\%$ fewer than R5's `518` parameters**).
- **`CKCER-ModeB` (100% Neural-Derived Query)**: Replaces the external 2-channel query cue with an **unsupervised local Oja Hebbian gate** ($w_q \in \mathbb{R}^{96}$, `486` total parameters, still `< 518`) that extracts the query context directly from the ALIF somatic transient $\delta_q(t_q) = [z(t_q); \, v(t_q) - \lambda v(t_q - 1)]$ at tick $t_q$ with **`1.0000` (100.0%) clustering purity** across all 11 dev and confirmation seeds, never accessing input cue channels `(4, 5)` at the readout.

Across the **8 fresh Confirmation seeds (`121011..121043`)**, both **`CKCER-ModeA`** and **`CKCER-ModeB`** pass **10/10 pre-registered gates (`8/8` seed wins on every adversarial condition)**:
- **Clean Confirmation**: `direct = 0.9985` (vs R5 `0.9226`), **Case-A (`vA != vB`) both-correct = `0.9960`** (`median = 1.0000, IQR = 0.0000`, vs R5 `0.7520`, $\Delta = +24.40\text{pp} \ [+15.01, +34.92]$, `8/8` wins).
- **OOD Clean $\to$ Shift (`no retrain`)**: `direct = 0.9990`, **Case-A both-correct = `0.9959`** (vs R5 `0.7580`, $\Delta = +23.80\text{pp}$, `8/8` wins).
- **In-Distribution Shifted Learning**: `direct = 0.9995`, **Case-A both-correct = `0.9960`** (vs R5 `0.7745`, $\Delta = +22.15\text{pp}$, `8/8` wins).
- **Reversed Write Order**: `direct = 0.9985`, **Case-A both-correct = `0.9941`** (vs R5 `0.7459`, $\Delta = +24.81\text{pp}$, `8/8` wins).
- **Distractors / Same-Sample Paired Distractors**: **Case-A both-correct = `0.9733` / `0.9658`** (vs R5 `0.4845` / `0.4342`, $\Delta = +48.88\text{pp}$ / $+53.17\text{pp}$, `8/8` wins).
- **Noise Change ($\sigma: 0.08 \to 0.16$)**: **Case-A both-correct = `0.9368` (`Mode A`) / `0.9359` (`Mode B`)** (vs R5 `0.4906`, $\Delta = +44.63\text{pp}$, `8/8` wins).
- **Query Tick Variation ($t_q = 11$)**: `direct = 0.9985`, **Case-A both-correct = `0.9960`** (vs R5 `0.6991`, $\Delta = +29.69\text{pp}$, `8/8` wins).
- **Auxiliary Task 1 (Variable Write Windows + Structured 3rd Write at Tick 8)**: `direct = 0.9919`, **Case-A both-correct = `0.9778`** (vs R5 `0.5396`, $\Delta = +43.82\text{pp}$, `8/8` wins).
- **Auxiliary Task 2 (3-Write Key Overwrite at Tick 8)**: `direct = 0.9397`, **Case-A both-correct = `0.8503`** (vs R5 `0.5275`, $\Delta = +32.28\text{pp}$, `8/8` wins).
- **Protected Tasks (`sim_xor`, `xor_separated`, `order_task`)**: Zero regression vs R5 (`0.9932` vs `0.9934`, `0.6270` vs `0.6282`, `0.9883` vs `0.9880`).

---

## 1. Stage 0 Audit Summary (`artifacts/gemini-r6/stage0-audit.json`)

Before designing or scoring any Round 6 candidate, `scripts/r6_stage0_audit.py` verified 12 invariants on the historical R5B seeds (`97001..97013` and `98001..98037`):

| Check # | Invariant Verified | Measured Value | Status |
|---:|---|---|---|
| 1 | Frozen 48-ALIF pre-query state linearly encodes independent $v_A, v_B$ and neural query $\Delta s$ | $v_A = 0.9967$, $v_B = 0.9974$, mixed $= 0.9961$, $q_{\text{neural}} = 1.0000$ | **PASS** |
| 2 | R5 query-conditioned feature map is strictly label-independent | $\max|\Delta f| = 0.00\text{e+}00$ | **PASS** |
| 3 | Query cue is extracted from actual input channels `(4, 5)` at the dynamically located query tick | `4096/4096` exact match | **PASS** |
| 4 | First-key shortcut achieves $\approx 0.75$ direct accuracy | `0.7522` | **PASS** |
| 5 | First-key shortcut achieves `0.0000` Case-A both-correct and `0.0000` flip rate | `both_A = 0.0000`, `flip_A = 0.0000` | **PASS** |
| 6 | Counterfactual siblings differ only at query tick $t_q$ on channels `4:6` | off-tick diff $= 0.0$, query diff $= 1.5000$ | **PASS** |
| 7 | Original R5 readout updates real finite weights | $\|\Delta W\|_F = 0.6231$, $\|\Delta b\|_2 = 0.0442$ | **PASS** |
| 8 | Hidden weights untouched (`sha256` identical) & `SpikingCore.step` parity is exact | `max_spike_diff = 0.00e+00` | **PASS** |
| 9 | Calibration statistics are frozen during test evaluation (zero test leakage) | `mean_diff = 0.00e+00` | **PASS** |
| 10 | Query tick is dynamically located (`10` clean, `11` shifted) and preserves cue | `qt_clean = 10`, `qt_shifted = 11` | **PASS** |
| 11 | Noise-padded shift and reversed writes change neural state (`zero_pad_diff = 0.0` caught) | `zpad = 0.0000`, `shift = 3.1670`, `rev = 3.3283` | **PASS** |
| 12 | Exact reproduction of R5B confirmation results on `98001..98037` | `direct = 0.8989`, `both_A = 0.6676`, `both_B = 0.9809`, `dist = 0.4536`, `noise = 0.4512` | **PASS** |

Stage 0 also quantified the **unconditional z-score cross-talk norm** on R5's gated-off branch: **`4.3856`** (`r5_gated_off_crosstalk_l2_norm` in `stage0-audit.json`).

---

## 2. Stage 2 Fair Baseline Comparison (Dev & Confirmation)

All 10 mandatory Stage 2 baselines were evaluated alongside `CKCER-ModeA` and `CKCER-ModeB` on the clean independent two-key retrieval benchmark across the 3 Dev seeds (`120011, 120017, 120023`) and 8 Confirmation seeds (`121011, 121017, 121023, 121029, 121031, 121037, 121041, 121043`):

| Model / Control | Params | Dev `direct` | Dev `both_A` | Conf `direct` (mean ± SD) | Conf `both_all` | Conf `both_A` (mean ± SD) | Conf `both_B` | Conf `flip_A` | Conf `wrong_fk_qB` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. `first_key_shortcut` | `0` | `0.7448` | `0.0000` | `0.7515 ± 0.0215` | `0.4956` | `0.0000 ± 0.0000` | `1.0000` | `0.0000` | `1.0000` |
| 2. `original_r5_head` | `518` | `0.9323` | `0.7218` | `0.9226 ± 0.0440` | `0.8770` | `0.7520 ± 0.1570` | `0.9991` | `0.7942` | `0.1498` |
| 3. `raw_serving_readout` (`lr=0.025`) | `388` | `0.7311` | `0.1822` | `0.7563 ± 0.0523` | `0.5342` | `0.1463 ± 0.1299` | `0.9274` | `0.2220` | `0.4648` |
| 4. `offline_ridge_diagnostic` *(Diag. Only)* | `390` | `0.9987` | `1.0000` | `0.9973 ± 0.0033` | `0.9966` | `0.9960 ± 0.0043` | `0.9972` | `0.9960` | `0.0020` |
| 5. `matched_capacity_control` (un-gated) | `390` | `0.7487` | `0.0128` | `0.7236 ± 0.0488` | `0.4697` | `0.0116 ± 0.0147` | `0.9394` | `0.0269` | `0.5067` |
| 6. `query_only_control` | `390` | `0.5052` | `0.0000` | `0.5107 ± 0.0259` | `0.2632` | `0.1977 ± 0.2729` | `0.3155` | `0.3750` | `0.5204` |
| 7. `query_free_state_control` | `390` | `0.7520` | `0.0000` | `0.7559 ± 0.0206` | `0.4922` | `0.0000 ± 0.0000` | `0.9935` | `0.0000` | `0.5008` |
| 8. `random_label_control` | `390` | `0.5163` | `0.3093` | `0.5366 ± 0.1231` | `0.2661` | `0.2787 ± 0.1375` | `0.2554` | `0.4901` | `0.4783` |
| 9. `fixed_nonlearning_candidate` | `390` | `0.5052` | `0.0000` | `0.5007 ± 0.0169` | `0.2529` | `0.0000 ± 0.0000` | `0.5101` | `0.0000` | `0.4969` |
| 10a. `cue_shuf_train` | `390` | `0.8210` | `0.4748` | `0.7185 ± 0.0811` | `0.5796` | `0.1755 ± 0.1355` | `0.9934` | `0.5083` | `0.4220` |
| 10b. `cue_shuf_test` | `390` | `0.7578` | `0.2411` | `0.7612 ± 0.0220` | `0.6182` | `0.2436 ± 0.0340` | `1.0000` | `0.4837` | `0.2422` |
| **`CKCER-ModeA` (Explicit Query Cue)** | **`390`** | **`0.9987`** | **`1.0000`** | **`0.9985 ± 0.0028`** | **`0.9976`** | **`0.9960 ± 0.0114`** | **`0.9991`** | **`0.9960`** | **`0.0030`** |
| **`CKCER-ModeB` (Neural-Derived Query)** | **`486`** | **`0.9987`** | **`1.0000`** | **`0.9985 ± 0.0028`** | **`0.9976`** | **`0.9960 ± 0.0114`** | **`0.9991`** | **`0.9960`** | **`0.0030`** |

**Critical Control Findings**:
- **`CKCER` matches the offline Ridge diagnostic ceiling (`0.9960` vs `0.9960` on Case A, `0.9985` vs `0.9973` on direct accuracy)**: Online local associative learning in `CKCER` closes **100% of the gap** to the closed-form linear Ridge diagnostic on the same representation.
- **`matched_capacity_control` (`390` params, un-gated `m_pre` + random projection + cue) gets `0.0116` on Case A (`0.7236` direct)**: Proves that `CKCER`'s performance does **not** come from parameter count, step size (`lr = 0.08`), or the pre-query trace `[c, a]` alone without multiplicative context gating.

---

## 3. Stage 3 Adversarial Benchmark Suite (8 Confirmation Seeds: `121011..121043`)

### 3.1 Overall Direct & Counterfactual Accuracy Across All 10 Adversarial Conditions

| Condition | R5 `direct` | CKCER-A `direct` | CKCER-B `direct` | R5 `both_A` (mean ± SD) | CKCER-A `both_A` (mean ± SD) | CKCER-B `both_A` (mean ± SD) | Paired $\Delta_{\text{A-R5}}$ `both_A` [95% CI] | Seed Wins |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 1. `clean` | `0.9226` | **`0.9985`** | **`0.9985`** | `0.7520 ± 0.1570` | **`0.9960 ± 0.0114`** | **`0.9960 ± 0.0114`** | **`+24.40pp [+15.01, +34.92]`** | **`8/8`** |
| 2. `ood_clean_to_shift` *(no retrain)* | `0.9209` | **`0.9990`** | **`0.9990`** | `0.7580 ± 0.1459` | **`0.9959 ± 0.0099`** | **`0.9959 ± 0.0099`** | **`+23.80pp [+14.71, +33.14]`** | **`8/8`** |
| 3. `train_shift_test_shift` | `0.9395` | **`0.9995`** | **`0.9995`** | `0.7745 ± 0.1328` | **`0.9960 ± 0.0114`** | **`0.9960 ± 0.0114`** | **`+22.15pp [+13.76, +30.30]`** | **`8/8`** |
| 4. `reversed_writes` | `0.9153` | **`0.9985`** | **`0.9985`** | `0.7459 ± 0.1235` | **`0.9941 ± 0.0099`** | **`0.9941 ± 0.0099`** | **`+24.81pp [+17.19, +32.62]`** | **`8/8`** |
| 5. `distractors` | `0.8267` | **`0.9885`** | **`0.9885`** | `0.4845 ± 0.0680` | **`0.9733 ± 0.0191`** | **`0.9733 ± 0.0191`** | **`+48.88pp [+45.18, +53.35]`** | **`8/8`** |
| 6. `paired_same_sample_distractors` | `0.8267` | **`0.9885`** | **`0.9885`** | `0.4342 ± 0.0970` | **`0.9658 ± 0.0177`** | **`0.9658 ± 0.0177`** | **`+53.17pp [+47.01, +59.10]`** | **`8/8`** |
| 7. `noise_change` ($\sigma: 0.08 \to 0.16$) | `0.8235` | **`0.9775`** | **`0.9778`** | `0.4906 ± 0.1055` | **`0.9368 ± 0.0442`** | **`0.9359 ± 0.0436`** | **`+44.63pp [+39.23, +51.00]`** | **`8/8`** |
| 8. `query_tick_variation` ($t_q = 11$) | `0.9050` | **`0.9985`** | **`0.9985`** | `0.6991 ± 0.1930` | **`0.9960 ± 0.0114`** | **`0.9960 ± 0.0114`** | **`+29.69pp [+18.84, +41.40]`** | **`8/8`** |
| 9. `aux_structured_third_write_windowed` | `0.8342` | **`0.9919`** | **`0.9919`** | `0.5396 ± 0.1072` | **`0.9778 ± 0.0179`** | **`0.9778 ± 0.0179`** | **`+43.82pp [+36.80, +50.07]`** | **`8/8`** |
| 10. `aux_key_overwrite` | `0.8418` | **`0.9397`** | **`0.9395`** | `0.5275 ± 0.0684` | **`0.8503 ± 0.0484`** | **`0.8503 ± 0.0484`** | **`+32.28pp [+28.84, +35.97]`** | **`8/8`** |

### 3.2 Section IX.A Hardest-Subset Breakdown (`Case A: vA != vB`, 8 Confirmation Seeds)

| Condition | Model | `both_A` Mean (Median / IQR) | `acc_qA_diff` | `acc_qB_diff` | `margin_qA_diff` | `margin_qB_diff` | `flip_rate_A` | `wrong_first_key_qB` |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `clean` | `R5` | `0.7520` (`0.7934` / `0.2463`) | `0.8595` | `0.8502` | `+5.505` | `+5.368` | `0.7942` | `0.1498` |
| `clean` | **`CKCER-ModeA`** | **`0.9960` (`1.0000` / `0.0000`)** | **`0.9990`** | **`0.9970`** | **`+7.098`** | **`+7.065`** | **`0.9960`** | **`0.0030`** |
| `clean` | **`CKCER-ModeB`** | **`0.9960` (`1.0000` / `0.0000`)** | **`0.9990`** | **`0.9970`** | **`+7.098`** | **`+7.065`** | **`0.9960`** | **`0.0030`** |
| `ood_clean_to_shift` | `R5` | `0.7580` (`0.8008` / `0.2246`) | `0.8644` | `0.8513` | `+5.457` | `+5.264` | `0.8001` | `0.1487` |
| `ood_clean_to_shift` | **`CKCER-ModeA/B`** | **`0.9959` (`1.0000` / `0.0021`)** | **`0.9980`** | **`0.9980`** | **`+7.055`** | **`+7.010`** | **`0.9959`** | **`0.0020`** |
| `train_shift_test_shift` | `R5` | `0.7745` (`0.7448` / `0.2029`) | `0.8806` | `0.8696` | `+5.644` | `+5.411` | `0.7989` | `0.1304` |
| `train_shift_test_shift` | **`CKCER-ModeA/B`** | **`0.9960` (`1.0000` / `0.0000`)** | **`0.9990`** | **`0.9970`** | **`+7.083`** | **`+7.013`** | **`0.9960`** | **`0.0030`** |
| `reversed_writes` | `R5` | `0.7459` (`0.7257` / `0.1970`) | `0.8540` | `0.8412` | `+5.357` | `+5.206` | `0.7967` | `0.1588` |
| `reversed_writes` | **`CKCER-ModeA/B`** | **`0.9941` (`1.0000` / `0.0099`)** | **`0.9971`** | **`0.9970`** | **`+6.872`** | **`+6.726`** | **`0.9941`** | **`0.0030`** |
| `distractors` | `R5` | `0.4845` (`0.4857` / `0.0659`) | `0.7038` | `0.6564` | `+2.893` | `+2.805` | `0.6089` | `0.3436` |
| `distractors` | **`CKCER-ModeA/B`** | **`0.9733` (`0.9779` / `0.0256`)** | **`0.9890`** | **`0.9843`** | **`+6.486`** | **`+6.290`** | **`0.9733`** | **`0.0157`** |
| `paired_same_sample_dist` | `R5` | `0.4342` (`0.4480` / `0.0508`) | `0.7039` | `0.6276` | `+3.118` | `+2.577` | `0.5368` | `0.3724` |
| `paired_same_sample_dist` | **`CKCER-ModeA/B`** | **`0.9658` (`0.9730` / `0.0187`)** | **`0.9842`** | **`0.9786`** | **`+6.561`** | **`+6.350`** | **`0.9689`** | **`0.0214`** |
| `noise_change` | `R5` | `0.4906` (`0.5209` / `0.1734`) | `0.7093` | `0.6492` | `+3.538` | `+2.397` | `0.6225` | `0.3536` |
| `noise_change` | **`CKCER-ModeA`** | **`0.9368` (`0.9483` / `0.0541`)** | **`0.9686`** | **`0.9682`** | **`+6.663`** | **`+6.565`** | **`0.9368`** | **`0.0347`** |
| `noise_change` | **`CKCER-ModeB`** | **`0.9359` (`0.9446` / `0.0541`)** | **`0.9677`** | **`0.9682`** | **`+6.639`** | **`+6.566`** | **`0.9359`** | **`0.0347`** |
| `query_tick_variation` | `R5` | `0.6991` (`0.7492` / `0.3463`) | `0.8185` | `0.7910` | `+5.274` | `+4.682` | `0.7888` | `0.2090` |
| `query_tick_variation` | **`CKCER-ModeA/B`** | **`0.9960` (`1.0000` / `0.0000`)** | **`0.9990`** | **`0.9970`** | **`+7.098`** | **`+7.065`** | **`0.9960`** | **`0.0030`** |
| `aux_third_write_window` | `R5` | `0.5396` (`0.5031` / `0.1148`) | `0.6756` | `0.6987` | `+3.039` | `+3.077` | `0.7049` | `0.3013` |
| `aux_third_write_window` | **`CKCER-ModeA/B`** | **`0.9778` (`0.9760` / `0.0276`)** | **`0.9916`** | **`0.9862`** | **`+6.444`** | **`+6.289`** | **`0.9778`** | **`0.0138`** |
| `aux_key_overwrite` | `R5` | `0.5275` (`0.5132` / `0.0776`) | `0.7508` | `0.7401` | `+5.199` | `+5.829` | `0.5641` | `0.2599` |
| `aux_key_overwrite` | **`CKCER-ModeA/B`** | **`0.8503` (`0.8591` / `0.0816`)** | **`0.9204`** | **`0.9290`** | **`+6.847`** | **`+6.610`** | **`0.8512`** | **`0.0710`** |

### 3.3 Per-Seed Confirmation Table (`clean` Condition Across All 8 Seeds)

| Seed | First-Key `direct` | R5 `direct` | R5 `both_A` | R5 `both_B` | CKCER-A `direct` | CKCER-A `both_A` | CKCER-A `both_B` | CKCER-B `direct` | CKCER-B `both_A` | CKCER-B Oja Purity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `121011` | `0.7559` | `0.9629` | `0.9078` | `1.0000` | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** |
| `121017` | `0.7617` | `0.8711` | `0.6098` | `1.0000` | **`0.9980`** | **`1.0000`** | **`0.9925`** | **`0.9980`** | **`1.0000`** | **`1.0000`** |
| `121023` | `0.7461` | `0.8789` | `0.4876` | `1.0000` | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** |
| `121029` | `0.7793` | `0.8633` | `0.6371` | `1.0000` | **`0.9922`** | **`0.9677`** | **`1.0000`** | **`0.9922`** | **`0.9677`** | **`1.0000`** |
| `121031` | `0.7539` | `0.9453` | `0.7868` | `1.0000` | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** |
| `121037` | `0.7598` | `0.9707` | `0.9206` | `1.0000` | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** |
| `121041` | `0.7051` | `0.9492` | `0.8000` | `0.9924` | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** |
| `121043` | `0.7500` | `0.9395` | `0.8661` | `1.0000` | **`0.9980`** | **`1.0000`** | **`1.0000`** | **`0.9980`** | **`1.0000`** | **`1.0000`** |
| **Mean** | `0.7515` | `0.9226` | `0.7520` | `0.9991` | **`0.9985`** | **`0.9960`** | **`0.9991`** | **`0.9985`** | **`0.9960`** | **`1.0000`** |

Notice that on **`7` out of `8` confirmation seeds** (and **`3` out of `3` dev seeds**), both `CKCER-ModeA` and `CKCER-ModeB` achieve **`1.0000` (`100.0%`)** on the hardest Case-A counterfactual test (`vA != vB`), and on the single remaining seed (`121029`, where R5 dropped to `0.6371`), both `CKCER-ModeA` and `CKCER-ModeB` achieve **`0.9677`**.

---

## 4. Answers to All 10 Required Section XII Questions

1. **What is the exact mathematical reason R5 failed on the hard counterfactual case?**
   By **Theorem 1** (`theory.md`), Case-A both-correct requires $(v_A - v_B)z_{\text{diff}}(x) > |z_{\text{com}}(x)|$. R5 fails because:
   - Its `192`-dim un-gated shared trunk $W_s s$ ($74.4\%$ of parameters) updates on $100\%$ of training examples toward the query-free sum $v_A + v_B$, contributing **zero** to $z_{\text{diff}}(x)$ while injecting large common-mode variance into $z_{\text{com}}(x)$ whenever $v_A \neq v_B$.
   - Its post-gating z-score standardization gives the **gated-off** branch ($c_k(q) = 0$) a non-zero standardized vector $-\mu_{g_k}/\sigma_{g_k}$ of L2 norm **`4.3856`**, so every Query B example corrupts Query A's weights $W_{g_0}$ and vice versa.
   - Its post-query state `post` at tick $t_q$ suffers from query-pulse somatic shock and fast-decay forgetting (`feature_decay = 0.25`).
2. **Which three theoretical mechanisms were derived, and why were two rejected before testing?**
   In `theory.md`, we derived:
   - **Approach 1 (`AMER` — Additive Modulated Eligibility Readout, Family B)**: Rejected pre-test because additive query conditioning cannot change the sign of state-dependent terms when $q$ flips ($z_{\text{diff}}(x) = \text{const}$), so $(v_A - v_B)z_{\text{diff}}$ is negative on at least $50\%$ of Case-A pairs (`Counterexample 1`).
   - **Approach 2 (`PQG-Full` — Post-Query Full-State Gated Readout, Family A)**: Rejected pre-test because it doubles parameter count to `772` (`> 518`), retains post-query somatic shock and subthreshold voltage noise $v(t_q)$, and suffers from post-gating z-score cross-talk (`Counterexample 2`).
   - **Approach 3 (`CKCER` — Context-Keyed Competitive Eligibility Readout, Family C+B+A synthesis)**: Selected pre-test because it solves all three mathematical flaws with `390` parameters (`Mode A`) / `486` parameters (`Mode B`), both smaller than R5 (`518`).
3. **What is the exact equation of the selected mechanism?**
   - **Pre-query eligibility state**: $m(t_q - 1) = [c(t_q - 1); \, a(t_q - 1)] \in \mathbb{R}^{96}$, where $c_j(t) = c_j(t - 1) + z_j(t)$ and $a_j(t) = 0.95 a_j(t - 1) + z_j(t)$.
   - **Pre-gating standardization**: $\hat{m}_r = \mathbb{I}[\sigma_{m,r} > 10^{-4}] \cdot \operatorname{clip}((m_r - \mu_{m,r})/\sigma_{m,r}, -5, 5)$.
   - **Context-keyed dendritic features**: $\psi(\hat{m}, g) = [g_0(t_q)\hat{m}(t_q - 1); \, g_1(t_q)\hat{m}(t_q - 1); \, g_0(t_q); \, g_1(t_q)] \in \mathbb{R}^{194}$.
   - **Local associative update**: $p = \operatorname{softmax}(W\psi + b)$, $e = t_y - p$, $\Delta W = \operatorname{clip}(\eta e \psi^\top, -0.025, +0.025)$ with $\eta = 0.08$. Because $g_{1-q}(t_q)\hat{m}(t_q - 1) \equiv \mathbf{0}$, $\Delta W^{(1-q)} \equiv \mathbf{0}$ (exact zero cross-talk).
4. **Which signals are strictly local, and which teaching/context signals are broadcast or input-derived?**
   - **Strictly local per neuron/synapse**: Spike counts $c_j(t)$, adaptation traces $a_j(t)$, somatic transients $\delta_{q,j}(t_q) = [z_j(t_q), v_j(t_q) - \lambda v_j(t_q - 1)]$, homeostatic statistics $(\mu_{m,r}, \sigma_{m,r})$, and compartment-gated trace products $g_k(t_q)\hat{m}_r(t_q - 1)$.
   - **Broadcast / context signals**: The 2-bit output teaching signal $y(t_q) \in \{0, 1\}$ (yielding output-local error $e \in [-1, 1]^2$ at the 2 readout soma units) arriving only at query tick $t_q$; plus in `Mode A` the 2-channel input query cue $g(t_q) = [x_{\text{ch4}}(t_q), x_{\text{ch5}}(t_q)]$, or in `Mode B` the 1-bit lateral WTA gate $\hat{g}(t_q)$ computed from the local Oja unit $w_q^\top \hat{\delta}_q(t_q)$.
5. **Does the mechanism require the external query cue, or can it operate from neural state alone?**
   **It operates from neural state alone with zero loss of accuracy.** `CKCER-ModeB` never reads input channels `4` or `5` at the readout; its unsupervised Oja Hebbian gate extracts the query identity from the ALIF somatic transient $\delta_q(t_q)$ with **`1.0000` (100.0%) purity** across all 3 dev seeds and all 8 confirmation seeds, matching `CKCER-ModeA` (`0.9985` direct, `0.9960` Case-A both-correct).
6. **Does it beat the ~75% shortcut and R5 baseline on held-out independent two-key retrieval?**
   **Yes.** On the 8 confirmation seeds, `CKCER` (`0.9985`) beats the first-key shortcut (`0.7515`) by **`+24.71pp [+23.36pp, +26.32pp]`** (`8/8` wins) and beats `original_r5_head` (`0.9226`) by **`+7.60pp [+4.76pp, +10.74pp]`** (`8/8` wins).
7. **Does it pass the hard counterfactual $A \neq B$ gate ($\ge 0.75$)?**
   **Yes, by `+24.60pp` above the gate.** `CKCER-ModeA` and `CKCER-ModeB` achieve **`0.9960`** Case-A both-correct (`median = 1.0000`, `IQR = 0.0000`), beating R5 (`0.7520`) by **`+24.40pp [+15.01pp, +34.92pp]`** (`8/8` wins).
8. **Does it survive timing shifts, reversed write order, distractors, and noise changes?**
   **Yes, across every condition (`8/8` seed wins on all conditions)**:
   - `ood_clean_to_shift`: `direct = 0.9990`, `both_A = 0.9959`
   - `train_shift_test_shift`: `direct = 0.9995`, `both_A = 0.9960`
   - `reversed_writes`: `direct = 0.9985`, `both_A = 0.9941`
   - `distractors` / `paired_same_sample_distractors`: `direct = 0.9885`, `both_A = 0.9733` / `0.9658`
   - `noise_change` ($\sigma = 0.16$): `direct = 0.9775` (`Mode A`) / `0.9778` (`Mode B`), `both_A = 0.9368` / `0.9359`
   - `query_tick_variation` ($t_q = 11$): `direct = 0.9985`, `both_A = 0.9960`
   - `aux_structured_third_write_windowed`: `direct = 0.9919`, `both_A = 0.9778`
   - `aux_key_overwrite`: `direct = 0.9397`, `both_A = 0.8503`
9. **Does it preserve prior XOR/order benchmarks?**
   **Yes.** On the 8 confirmation seeds, `sim_xor` is `0.9932` (vs R5 `0.9934`, $\Delta = -0.0002$), `xor_separated` is `0.6270` (vs R5 `0.6282`, $\Delta = -0.0012$), and `order_task` is `0.9883` (vs R5 `0.9880`, $\Delta = +0.0002$), all well within the `< 0.01` (`1.0pp`) non-regression gate `G8`.
10. **What is the strongest scientific objection to the result, and what control answers it?**
    - **Objection 1**: *"Perhaps `CKCER` succeeds merely because `[c, a]` is a cleaner pre-query feature vector or because `lr = 0.08` with `dead = 1e-4` is a better optimizer, not because of context-keyed dendritic gating with zero un-gated trunk."*
      - **Control Answer**: Both `matched_capacity_control` (`390` params, using the exact same standardized `[c, a]` trace, `lr = 0.08`, and `dead = 1e-4` without multiplicative gating) and `step2_pre_gating_norm_plus_trunk` (`[c, a]` + pre-gating `dead = 1e-4` + R5's un-gated trunk and random projection $P$) were tested on the exact same seeds. `matched_capacity_control` collapses to **`0.0116`** Case-A both-correct (`0.7236` direct), and `step2_pre_gating_norm_plus_trunk` reaches only **`0.9131`** Case-A both-correct with `4.85%` wrong first-key bias. Removing the un-gated shared trunk and random projection $P$ (`Step 3 = CKCER`) jumps Case-A both-correct to **`0.9960`** and cuts wrong first-key errors by **$16\times$** (`0.0485` $\to$ `0.0030`).
    - **Objection 2**: *"Perhaps `Mode A` relies on an externally supplied input query cue that a biological readout would not receive directly."*
      - **Control Answer**: `CKCER-ModeB` removes the external query cue from the readout entirely and learns the dendritic gate unsupervised via Oja's local Hebbian rule from the ALIF somatic transient $\delta_q(t_q)$, achieving **`1.0000` gate purity** and **`0.9960` Case-A both-correct**.

---

## 5. Error Atlas & Honest Limitations

1. **Residual Errors on `aux_key_overwrite` (`both_A = 0.8503` vs `0.9960` on 2-write tasks)**:
   When a key is written at tick `2` ($v_{A0}$) and overwritten at tick `8` ($v_{A,\text{new}} = 1 - v_{A0}$), the cumulative spike count $c(t_q - 1)$ ($\gamma_c = 1.0$) sums spikes from both the old and new writes equally, so `CKCER` must rely on the difference between $c(t_q - 1)$ ($\gamma_c = 1.0$) and $a(t_q - 1)$ ($\gamma_a = 0.95$, where $0.95^7 = 0.698$ vs $0.95^1 = 0.950$) to separate the tick-8 overwrite from the tick-2 initial write. While `CKCER` still beats R5 by **`+32.28pp`** (`0.8503` vs `0.5275`), repeated overwrites of the same key over longer horizons would require a faster-decaying eligibility trace alongside $\gamma_a = 0.95$.
2. **Scope Limitation (No P2 Continual-Learning Claim)**:
   In accordance with Round 6C scope rules, `CKCER` is evaluated strictly in `src/scarx/evaluation/r6_associative.py` on frozen 48-ALIF representations. While dendritic compartment isolation ($g_{1-q}\hat{m} \equiv \mathbf{0}$) eliminates cross-query overwrite within the associative retrieval task, this report makes **no claim** of solving P2 continual multi-task dialogue retention or hidden-weight credit assignment.
