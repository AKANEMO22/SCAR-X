# PROTOCOL v1 — MTB: multi-timescale readout receptive field

**Campaign:** `scarx-neural-round1-20261111`
**Agent:** antigravity
**Environment actually used:** DeepSeek Harness (DSH) Web, Windows 11, session Lead
**Model actually used:** `deepseek-ai/deepseek-v4.1-flash-1017`
*(Antigravity is the agent/environment label. The executing model is the line above;
the environment name is not the model name.)*
**BASE_SHA:** `1735ad66ea9e7b416f5df15df99d9c13cea46b4e`
**Hypothesis commit:** `0010661e76a4505e1d03993a765dad828059dd51` (`hypothesis-v1.md`)
**Branch:** `research/antigravity/scarx-neural-round1-20261111`

This protocol was written and committed **before** any formal run of the
candidate. Amendments are appended as `protocol-v1a.md`, …; results produced under
v1 are never re-interpreted under a later version.

---

## 1. Authority and limits granted by this prompt

**Granted:** create a private branch/worktree; choose a hypothesis; write this
protocol first; implement a private research prototype (`evaluation/`, never
imported by serving code); run pilot + confirmation after the P0 gates; push code
and results; open a Draft PR; read and re-test the other two branches; write
cross-reviews; amend my own branch on evidence.

**NOT granted / forbidden:** merging to `main`; self-approving release; changing
weights, database or the model in use in Hermes; editing other agents' branches;
changing the goal or lowering criteria after seeing results; paid APIs or new
cost; overwriting historical data or results.

The repository carries historical "chưa duyệt" (not-approved) notices: proposal
P2-usefulness *pilot* (report 41/42) and `readout-protection-v1` remain **not
approved** and are **not** touched here. This prompt authorises the *private
research experiments described in this protocol* only — not any other change.
The user-approved proposal-30 signal boundary (neuron-local forward eligibility
plus one-step `W_out.T @ error`) is used **unchanged** and is not widened.

## 2. Invariants this protocol cannot change

- SCAR-X remains **one** Spiking Recurrent Neural Network with LIF/ALIF neurons.
- Synapses learn for real; save/restore preserves learned state.
- CPU-first; whole-runtime target **< 2 000 000 000 bytes** RAM.
- No Transformer, no pretrained LLM, no LLM proxy, no pretrained CNN/ViT/VLM as a
  vision core.
- No whole-network or through-time backward in the candidate.
- No autograd / BPTT / RTRL in the candidate. BPTT exists **only** in
  `src/scarx/evaluation/controls.py` as an isolated control, whose weights are
  never imported into the candidate or the product.
- Fast / Deep / Idle remain three modes of **one** core, not three models.
- More neurons, more ticks or more data do not by themselves prove intelligence.

## 3. Conditions (all share neuron dynamics, ticks, data and hidden rule)

Neuron dynamics, spiking (`p=σ((v−limit)/0.4)`, Bernoulli), recurrence, adaptation,
`Config` scalars and the hidden learning rule are **identical** in every arm. The
conditions differ **only** in the readout's temporal basis and the signal that
drives hidden learning.

| id | readout basis | hidden learning | purpose |
|---|---|---|---|
| `prod` | K=1, `d=0.25` (production `feature_decay`) | full local score×advantage | the real incumbent |
| `mtb` | K=5, `d∈{0.10,0.25,0.50,0.75,0.90}` | full local score×advantage | **candidate** |
| `mtb_dim` | K=5 (same decays) | full local | *(same as mtb; the matched control is `prod+x240`)* |
| `prod_x240` | K=1 `d=0.25` expanded to 240d by a **fixed random projection** | full local | **matched-dimension control**: same parameters/compute, one timescale |
| `prod_frozen` | K=1, `d=0.25` | none (frozen hidden) | no-learning control |
| `mtb_frozen` | K=5 | none (frozen hidden) | isolated readout-only control |
| `prod_readout` | K=1, `d=0.25` | readout-only (hidden frozen) | incumbent readout-only |
| `mtb_readout` | K=5 | readout-only (hidden frozen) | candidate readout-only |
| `mtb_shuffle` | K=5 | full local, **shuffled sign** of the modulation | signal-must-matter control |
| `mtb_one_decay` | K=5 slots, **all `d=0.25`** (same dims, one timescale) | full local | purely-timescale control at identical width |
| `bptt_ctl` | K=5 | isolated BPTT (`evaluation/controls.py`) | task-learnability ceiling only; never imported |

`mtb_one_decay` and `prod_x240` both hold dimension and parameter count equal to
`mtb`; together they separate "more parameters" from "more timescales".

## 4. Data — generators, versions, splits, leakage

- **Source:** repository-authored synthetic generators only.
  `scarx.evaluation.tasks21.representation` (`order`, `xor`) and
  `scarx.evaluation.binding_audit21.strict_binding`. No downloaded, private or
  model-generated data. No Qwen output is used as a label anywhere.
- **Why these tasks:** `order` and `xor` are the two tasks on which the repository
  previously passed the representation gate (report 20) and therefore prove the
  harness can detect a real effect; `strict_binding` is the registered failure case.
- **SHA-256 of the generated datasets** is written to each run's `manifest.json`
  by hashing the concatenated `(x, y)` arrays per split. Regeneration is
  deterministic from the split seed in §5.
- **Splits, disjoint by seed offset** (copied exactly from the repository's own
  convention in `scripts/run_p1_paired_score.py` so results stay comparable):

| role | seed offset | size (dev) | size (confirmation) | used for |
|---|---|---|---|---|
| calibration | `+6 200 000` | 128 | 128 | readout normalisation only, **no labels** used for fitting |
| training | `+6 300 000` | 2048 | 2048 | optimisation |
| probe fit | `+6 400 000` | 128 | 128 | refit-head train |
| **dev / test** | `+6 500 000` | 256 | 512 | scoring |
| OOD | `+6 600 000` | 128 | 128 | shifted-position generalisation |

- **No leakage:** labels enter only the loss/readout target. The target is never
  placed into the neural input vector; a P0 test asserts the input tensor is
  bit-identical when labels are permuted.
- **Verified labels:** all labels are produced by the same code that generates the
  inputs. An independent **label-oracle** check is run in P0: the order-3 analytic
  feature reaches 1,00 on `strict_binding` and ≈0,50 on its label-shuffled control
  (measured, §2.1 of the hypothesis); this proves the label is carried by the
  inputs and that the shuffled control is honest.

## 5. Seeds

- **Development (screen):** `21001, 21007, 21013` — 3 fresh, never used in any
  prior SCAR-X run (verified by grep over `scripts/` and `evaluation/`).
- **Confirmation (opened only if the screen passes):** `22003, 22009, 22013,
  22019, 22021, 22027, 22031, 22037` — 8 fresh seeds, unused, disjoint from
  development and from every seed set in reports 20/21/27/35/36/37.
- Each seed instantiates **one** fresh model. Three checkpoints of one model are
  **not** three independent initialisations and will never be counted as such.
- No best-seed selection: statistics are computed over **all** seeds of the arm.

## 6. Budget, updates, ticks

- H=48, K=8 (fan-in), R=12 (plastic rows) — the registered topology, unchanged.
- 2048 training examples, 1 update opportunity per example, `blocks=4`
  (`plastic_rows=12` covers all 48 rows), 12 ticks/example.
- Equal-**update** comparison is primary; an equal-**wall-time** comparison is
  reported as secondary because `mtb` has a wider readout.
- **Hard limit: 3 hours per formal run.** The screen is budgeted ≈15 min; the
  confirmation ≈60 min. Wall time and peak RAM are recorded per trial; a run that
  exceeds the ceiling is stopped and reported as such, not silently extended.
- Exactly **two** legitimate adjustment rounds are allowed for this hypothesis
  (v1 dev screen, v1a amendment). After that the verdict stands.

## 7. Metrics, gates, statistics

**Primary:** `dev_direct` = argmax accuracy of the real readout on the dev split.

**Secondary:** hidden-representation gain = `ridge_probe(learned hidden) −
ridge_probe(initial hidden)` on the **same** fit/test split, same algorithm,
same budget, head refit from scratch; plus acquisition, final, OOD accuracy,
spike-train identity, weight-delta norms, `retained_after` metrics.

**Gates (all must hold; thresholds are the master-plan values and are NOT lowered):**

| gate | threshold |
|---|---|
| G1 acquisition/final | ≥ 65% dev, and not chance-level |
| G2 representation | CI95% lower bound of hidden gain **≥ 5pp** on ≥ 2 of 3 tasks |
| G3 representation vs readout-only | candidate beats `readout-only` with corrected significance |
| G4 forgetting | mean ≤ 3pp, worst ≤ 5pp (retention arm, §9) |
| G5 attribution (primary) | `mtb` − `prod_x240` CI95% lower bound **> 0** |
| G6 attribution (timescale) | `mtb` > `mtb_one_decay` with CI95% lower bound > 0 |
| G7 signal matters | `mtb` > `mtb_shuffle` with corrected significance |
| G8 RAM | peak process-tree commit < 2 000 000 000 B |

**Statistics:** seed-level mean and 95% bootstrap CI (10 000 resamples,
`numpy.random.default_rng(2101)`, matching `phase21.interval`); paired contrasts
computed per seed; sign-flip exact p; **Holm** correction across the three tasks.
A CI containing 0 is a fail, not "trending". Three paired conditions are **not**
three independent seeds.

**Selection rule:** hyperparameters are chosen on **development** only. `readout_lr`
is screened on dev over `{0.025, 0.05, 0.10}` for every arm with the same rule; the
single winner per arm is locked before confirmation and then frozen. Confirmation
never tunes anything.

## 8. P0 — technical checks required before any capability number

All must pass; a failure means fixing the implementation, **never** reporting a
capability score from it.

1. **Forward/signal parity.** `mtb` forward reproduces `prod` exactly when all
   `d_k = 0.25`; spike trains must be **bit-identical** across all arms for the
   same seed (asserted by hashing). This is the central claim: the condition is
   readout-only.
2. **No target leakage.** Permuting labels leaves the input tensor bit-identical
   (SHA-256 compare) and hidden weights update bit-identically.
3. **Checkpoint completeness.** Save/restore must preserve: `w_in, w_rec, w_out,
   bias, stable_*, indices`, neuron state (`voltage, adaptation, spikes, pre,
   post, features, elig_*`, **all K feature channels**, `voltage_pre`, moments),
   RNG (`rng`, `mod_rng`), `updates`, `reward_mean`, `rows`, `ticks`, topology,
   vocabulary, controller and scheduler state.
4. **Cold-process resume.** A separate fresh Python process loads the checkpoint
   and performs the next learning step; result must equal the in-process
   continuation bit-for-bit (array hashes).
5. **Rollback.** A deliberately failing update must restore the full previous
   state; every array hash returns to the pre-update value.
6. **Loader rejection.** Wrong format tag, wrong array set, wrong shape/dtype,
   corrupt hash, duplicate member, non-finite value, oversized payload must each
   be rejected with an error — never silently migrated.
7. **Peak memory.** Measure RSS **and** commit for learn / save / load / rollback,
   **in-process and across the process tree** (parent + workers). A per-process cap
   is not evidence that the whole job is under 2 GB.
8. **No offload.** No weight migration to SSD/pagefile; all learned state stays in
   RAM (asserted by array-shape accounting and absence of any file-backed array).
9. **Test suite + parity.** `python -m pytest -q` must pass, plus the CPU/native
   parity tests for the touched runtime paths.

## 9. Retention arm

Aquisition and retention are always reported together. The retention stream reuses
the repository's established metric: 3 task phases A→B→C with 2048 examples each,
`retained_after` measured on A; `forgetting = max(acc_A during stream) − acc_A at
end`. Frozen and readout-only arms are included so that "no forgetting" cannot be
achieved by simply not learning. A model that does not learn cannot pass G1 no
matter how small its forgetting.

## 10. Screening and stop rule

- **Screen:** dev only, 3 seeds, 11 conditions × 3 tasks. Pass requires **all** of:
  `mtb` beats `prod_x240` (G5) and `mtb_one_decay` (G6); `mtb_frozen`−`prod_frozen`
  > 0; spike-train identity holds.
- **If the screen fails → STOP this hypothesis.** No retune, no extra epochs, no
  LR sweep to rescue it, no renaming. Verdict `NO_GO` or `PIVOT` with the measured
  reason. Budget is not extended because a result is disappointing.
- **If the screen passes →** freeze hyperparameters, open the 8 fresh confirmation
  seeds once, and evaluate G1–G8.
- Any observed RAM breach, NaN, persistence error, test-set leak, or signal-locality
  violation stops the run immediately and is reported as a failure, not discarded.

## 11. Loop closure

**Never allowed:** using confirmation/dev accuracy to pick an implementation, or
re-running confirmation with a different configuration and reporting the better one.

## 12. Claim scope

Every claim is stated with: task, H=48, synthetic generator, seed count, and the
exact metric. Explicitly **not** claimed by this protocol: normal chat, general
vision, OCR, VQA, reasoning, lifelong skill retention, or that SCAR-X "understands"
anything. A positive result here means only: *on these registered synthetic tasks,
at H=48, the multi-timescale readout receptive field — and not extra parameters —
improves acquisition and the measured hidden-representation gain.*

## 13. Artifacts

```
runs/<unique-run-id>/            result.json, report.md, manifest.json, per-trial raw predictions
artifacts/antigravity-r1/        checkpoints + cold-resume receipts (not committed if large)
experiments/collab/scarx-neural-round1-20261111/antigravity/   this protocol, amendments
docs/research/collab/scarx-neural-round1-20261111/antigravity/ report, lessons-learned
```

No shared `latest.json`. Every run id is unique and prior runs are never rewritten.
No database, secrets, private logs or personal conversation content are committed.
