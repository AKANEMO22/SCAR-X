# MTB report — multi-timescale readout receptive field

**Campaign:** `scarx-neural-round1-20261111`
**Agent:** antigravity
**Environment actually used:** DeepSeek Harness (DSH) Web, Windows 11, session Lead
**Model actually used:** `deepseek-ai/deepseek-v4.1-flash-1017`
*(Antigravity is the agent/environment label. The environment name is not the
model name; the model that executed this work is the line above.)*
**Ponytail:** not available as a plugin in this profile. `docs/STATUS.md:1320`
already records the same unavailability for a prior session. No ponytail run is
claimed or simulated.

| Item | Value |
|---|---|
| BASE_SHA (shared, verified) | `1735ad66ea9e7b416f5df15df99d9c13cea46b4e` |
| Hypothesis commit (before reading peers) | `0010661e76a4505e1d03993a765dad828059dd51` |
| Protocol commit | `b0f365906c3ff0f295d772c9a772e1b649913dbf` |
| Protocol SHA-256 | `a087cbaca583491131f6e03dee780336c4dd60176cbc1407dbc069e286d2d2d0` |
| Candidate commit (this report) | see final handoff signal |
| Screen run id | `20261110T194509Z-collab-mtb-screen-v1-de8c62` |
| Superseded run id | `20261110T192339Z-collab-mtb-screen-v1-a7409e` (crashed; marked `SUPERSEDED.txt`) |
| Self verdict | **PIVOT** — mechanism diagnosis survives; the learning hypothesis does not |

---

## 1. Hypothesis (as locked, before results)

> Giving the **readout** a bank of leaky integrators spanning multiple timescales
> (neuron dynamics, spiking, recurrence and the hidden learning rule held
> bit-identical) will raise acquisition on tasks needing multi-timescale
> integration **and** will increase the measured hidden-representation gain,
> because the single-decay readout masks what the hidden weights learned.

The null: any apparent gain is explained by added readout parameters, extra
compute, or a changed evaluation basis.

Falsifiers registered in advance: MTB fails to beat the dimension-matched control;
hidden-representation gain fails; the shuffled-signal control matches MTB; gates on
acquisition/forgetting fail; or spike trains differ between MTB and the
dimension-matched control.

## 2. What was actually changed

One thing only: the readout's temporal basis.

```
production:  f[t] = 0.25·f[t-1] + 0.75·z[t]                     (K=1, 48 dims)
candidate :  f_k[t] = d_k·f_k[t-1] + (1-d_k)·z[t]               (K=5, 240 dims)
             d_k ∈ {0.10, 0.25, 0.50, 0.75, 0.90}
             logits = W_out · concat(f_1..f_5) + bias            (one shared W_out)
```

Neuron equations (`v`, `limit`, `p=σ((v−limit)/0.4)`, Bernoulli spikes,
adaptation, recurrence) and the hidden learning rule (neuron-local forward
eligibility × scalar advantage) are **unchanged**. Nothing is imported by serving
code; the prototype lives only in `src/scarx/evaluation/multi_timescale.py`.

**Signal boundary:** hidden learning uses only the approved proposal-30
`W_out.T @ error` per-neuron factor plus forward eligibility. No hidden→hidden
error transport, no BPTT, no RTRL, no autograd, no reverse traversal. The readout
uses its own local delta rule. This is disclosed as **not fully biologically
local**, exactly as proposal 30 requires.

## 3. P0 technical gates — 30/30 pass

Run: `python scripts/validate_multi_timescale_p0.py` → `artifacts/antigravity-r1/p0-validation.json`

| Area | Result |
|---|---|
| Forward parity vs `SpikingCore` (hard threshold) | max abs diff 1.19e-07 ≤ 4 ULP (float32 `1−0.92` rounding); **spikes bit-identical** |
| Forward parity vs `PlasticCore` (stochastic, shared RNG stream) | 0.000e+00 |
| Cross-arm spike-train bit-identity (7 arms, identical weights) | **holds** — 1 distinct hash |
| Hidden update independent of readout width (equal reward) | w_in/w_rec bit-equal |
| Hidden update independent of label (equal reward) | w_in/w_rec bit-equal |
| Checkpoint completeness | 19 arrays incl. all 5 decay channels, RNG, counters |
| Cold-process resume | next learning step **bit-exact** |
| Rollback | every array hash restored |
| Loader rejection | 8/8 malformed payloads rejected |
| Peak process-tree commit | **700,256,256 B** < 2,000,000,000 B |
| File-backed arrays / SSD offload | none |

Repository suite: **287 passed, 4 skipped, 1 failed**. The single failure
(`tests/test_openai_backend.py::test_explicit_json_teaching_updates_persistent_model`,
HTTP 507) reproduces **identically on the pristine baseline with my files
removed**, so it pre-exists and is not caused by this work.

## 4. Results

297 registered trials: 11 arms × 3 tasks × 3 dev seeds × 3 `readout_lr`.
Grid integrity asserted: 297/297 cells, 0 missing, 0 duplicates. `readout_lr` was
selected per arm **on development only** and locked to
`artifacts/antigravity-r1/locked-readout-lr.json` before any confirmation.

### 4.1 Acquisition — all gates FAIL

| Contrast (paired by seed, pp) | mean [95% CI] | Gate |
|---|---|---|
| `mtb − prod_x240` (dimension-matched) | **+3.60 [−1.30, +8.77]** | G5 **FAIL** |
| `mtb − mtb_one_decay` (same width, one timescale) | **+4.90 [−0.22, +10.42]** | G6 **FAIL** |
| `mtb − mtb_shuffle` (signal matters) | **+0.65 [−0.95, +2.30]** | G7 **FAIL** |
| `mtb_frozen − prod_frozen` | **+4.51 [−0.09, +10.07]** | **FAIL** |
| `mtb − mtb_frozen` (hidden learning contributes) | **+0.43 [−0.91, +1.87]** | — |
| `mtb − mtb_readout` | +0.43 [−0.91, +1.87] | — |
| `mtb − prod` (incumbent) | +4.25 [−1.65, +11.07] | — |
| `prod − prod_frozen` | +0.69 [−1.17, +2.39] | — |

Every CI contains 0. `screen_pass = False`, `screen_gates_unknown = []` (so this
is a real failure, not an unknown one).

### 4.2 The decisive negative: hidden learning contributes nothing

Direct accuracy, mean of 3 dev seeds, each arm at its own locked lr:

| arm | order | xor | strict_binding |
|---|---:|---:|---:|
| `prod` | 51.43 | 50.65 | 52.47 |
| `mtb` | **68.88** | 48.05 | 50.39 |
| `prod_x240` | 55.21 | 51.69 | 49.61 |
| `mtb_one_decay` | 53.39 | 49.61 | 49.61 |
| `prod_frozen` | 51.43 | 50.00 | 51.04 |
| `mtb_frozen` | **66.54** | 49.61 | 49.87 |
| `mtb_readout` | **66.54** | 49.61 | 49.87 |
| `mtb_shuffle` | **66.15** | 48.83 | 50.39 |
| `mtb_only_fast` | 48.57 | 53.12 | 50.00 |
| `mtb_only_slow` | **65.89** | 49.87 | 50.00 |

The entire `mtb` advantage sits on `order`, and it is present at **the same level
with the hidden frozen** (66.54), with the readout trained alone (66.54), and with
the learning signal sign-shuffled (66.15). `mtb_only_slow` — a *single* decay of
0.90 at 48 dims — already reaches 65.89. Everything the "multi-timescale" arm
achieves is available from one slow trace with no learning at all.

`xor` and `strict_binding` do not move for any arm (48–53%, at chance).

### 4.3 Hidden-representation gain — no arm passes

| arm | order | xor | strict_binding |
|---|---:|---:|---:|
| `prod` | +3.26 | +1.04 | −2.08 |
| `mtb` | +1.82 | +1.43 | −1.30 |
| `prod_x240` | +5.21 | +3.12 | −3.91 |
| `mtb_shuffle` | −1.30 | +1.17 | −5.08 |

Control arms (`*_frozen`, `*_readout`) are exactly 0.00 by construction, which
validates the probe. No gain reaches the registered +5pp lower-CI threshold; the
best (`prod_x240` order +5.21pp mean) is a point estimate with no CI separation
and belongs to a **control** arm, not the candidate.

## 5. Verdict — **PIVOT**

**The learning hypothesis is rejected.** Recorded plainly:

- The candidate does **not** beat the dimension-matched control with a CI
  excluding 0 (G5 FAIL). Parameter count is not the explanation, but neither is
  timescale multiplicity established.
- It does **not** beat the same-width single-timescale control (G6 FAIL).
- **The learning signal does not matter** (G7 FAIL): `mtb_shuffle` is
  statistically indistinguishable from `mtb`.
- **Hidden learning contributes nothing** (+0.43pp [−0.91, +1.87]): the frozen
  hidden is as good as the learned one.
- Nothing generalizes: all of the effect is on `order`; `xor` and `strict_binding`
  stay at chance.

Per the protocol's locked stop rule, I do **not** retune, do not add seeds, do not
open the 8 confirmation seeds, and do not rename the family. The confirmation
phase was never run because the screen did not pass — that is the designed
outcome, not an omission.

### What genuinely survives (and its exact scope)

A **diagnostic**, not a capability gain: on the registered 12-tick `order` task at
H=48, the production single-decay readout (0.25) is a weak linear readout of a
delayed order bit, and a slower/multi-timescale basis reads it far better
**before and without any synaptic learning**. My probe measured this on frozen
weights (single decay 0.59–0.68 vs 5-decay bank 0.94–0.96, with a
dimension-matched control bit-identical to the single decay). This is an
**information-access** statement about the readout, not evidence that SCAR-X
learns better.

## 6. Claim matrix

| Claim | Metric | Source | Scope | Limitation |
|---|---|---|---|---|
| MTB beats the dimension-matched control | +3.60pp [−1.30, +8.77] | screen result | order/xor/binding, H=48, 3 dev seeds | **FAILED** — CI contains 0 |
| MTB beats single-timescale at equal width | +4.90pp [−0.22, +10.42] | same | same | **FAILED** |
| The learning signal matters | +0.65pp [−0.95, +2.30] | same | same | **FAILED** |
| Hidden learning contributes | +0.43pp [−0.91, +1.87] | same | same | **FAILED** — frozen hidden is equivalent |
| Slow readout basis recovers a delayed order bit without learning | frozen probe 0.94–0.96 vs 0.59–0.68 | `probe_design2.py` scratch + P0 | 12-tick order, H=48 | diagnostic only; not a skill, not a serving head |
| Condition is readout-only (identity preserved) | spike hashes equal on identical weights | P0 check 2 | 7 arms | post-training arms legitimately diverge per rule group |
| Technical integrity | 30/30 P0, 287 suite pass, 700 MB peak | `p0-validation.json` | this prototype | pre-existing suite failure unrelated |

## 7. What did not go as planned (disclosed)

1. **First screen run crashed** in `summarise()` on a contrast key I referenced
   but never defined (`mtb_minus_prod_frozen`). 290/297 trials survived; the run
   is marked `SUPERSEDED.txt` and is not cited.
2. **The rerun then wrote a broken summary**: `per_arm` lr lock was empty during
   the screen phase, and `screen_pass` was **vacuously `True`** because all gates
   were `None`. This is precisely the silent-success failure mode the campaign
   forbids. Fixed: unknown gates are now reported as unknown and `screen_pass`
   requires every gate `is True`.
3. **Contrast pairing was keyed on `readout_lr`**, which differs per arm by
   design, silently producing `None` contrasts. Fixed: paired on seed, the actual
   unit of independence.
4. **Spike-hash equality was asserted across arms with different hidden rules**,
   which must diverge after training. Fixed: reported per rule group. The
   readout-only claim rests on P0's identical-weight check, not on this.
5. **P0 initially reported 5 failures** that were my *test* bugs (wrong
   `feature_decay` reference, wrong zip member name, RNG stream not shared, a
   float32 ULP tolerance). Each was diagnosed before changing anything; the model
   code was correct in those cases.

All four aggregation defects were found by my own integrity assertions rather
than by inspection, and `result.json` was regenerated from the **same raw
trials** (`scripts/recompute_mtb_summary.py`) with grid integrity asserted —
no trial numbers were altered.

## 8. Relation to prior work — why this was not a re-tune

Report 22 (per-neuron heterogeneous time constants) screened as an
inference-only view and was STOPPED (`STOP_TIMESCALE_VIEW_SCREEN_V1`). That work
changed **neuron** dynamics (`leak`/`adaptation_decay`/`feature_decay` per neuron).
MTB changes **nothing in the neuron**; only the readout's receptive field. The
difference is not a rename: report 22's arms have different spike trains by
construction, whereas MTB's arms are asserted bit-identical at P0 on shared
weights. Reports 26/27/35/36/37 changed encoder, task adapter, head objective or
learning signal; MTB changes only the readout basis and kept those rules.

Honestly, though: my result lands in the **same place** as report 22's screen —
slower readout bases read the delayed bit better, and no learning advantage is
demonstrated. That convergence should be read as a genuine boundary of this
architecture family at this scale, not as a new opening.

## 9. Cross-review work

- **grok** (`d50987862ff80d2bd8556eecfb6af1b259cc591e`): reviewed, verdict
  **PARTIAL**. I reproduced all six 8-seed arm means and their headline CI
  exactly, and identified their interval as a t-interval rather than the
  repository bootstrap — a method choice that **determines their NO-GO verdict**
  (bootstrap on the same eight numbers gives a positive lower bound). Recorded as
  `grok-003` and flagged for adjudication rather than resolved in either
  direction. See `cross-review-grok.md`.
- **deepseek**: **no branch exists on the remote.** `git ls-remote` at review time
  shows only `main`, `cursor/scarx-release-candidate`,
  `research/grok/scarx-neural-round1-20261111`, and my own branch. There is
  nothing to fetch, so this review **could not be performed**. I did not invent a
  review or a signal for an absent target.

## 10. Open blockers

1. **`deepseek` branch absent** — the second cross-review is not possible. This
   will be declared in the final handoff rather than fabricated.
2. **No `gh` CLI and no token in this environment**, so the Draft PR must be
   opened manually by the user (git push over HTTPS works). I will provide the
   exact compare URL.
3. **Strict binding is not solvable by this architecture at any learning-rule
   quality.** My scratch probe showed the task needs a 3rd-order feature
   (order-1 0.52, order-2 0.54, order-3 **1.00**, label-shuffled 0.45). An
   additive point neuron plus a linear readout is 1st-order, so the 65% gate is
   unreachable there regardless of credit assignment. This is a **task-validity**
   finding that affects how every branch should read its `strict_binding` numbers,
   and it is worth registering as its own future protocol.

## 11. Not claimed

No normal chat, no general vision, no OCR, no VQA, no reasoning, no lifelong skill
retention, no forgetting gate met (retention was not run because the screen
failed). Every number here is a registered synthetic task at H=48 on CPU. This
branch is **not** proposed for integration into Hermes.
