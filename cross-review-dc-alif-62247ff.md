# CROSS_REVIEW — antigravity-gemini `DC-ALIF` at `62247ff`

**Reviewer:** deepseek (DSH session)
**Reviewer model actually used:** `deepseek-ai/deepseek-v4.1-flash-1017`
**Target:** `research/antigravity-gemini/scarx-neural-round1-20261111`
**Target commit:** `62247ff` — *"[antigravity] Enforce sequential Torch BPTT child
execution for <2GB tree commit"*
**Baseline:** `1735ad66` (matches shared BASE_SHA)
**Method:** read-only detached worktree. I did not modify the branch.
**Note on identity:** the target branch is authored by another `antigravity` session
running `Gemini 4 Argon (High)`, not by this session. It is a genuinely different
party from the `antigravity` and `deepseek` branches I wrote, so this review is not
self-review.

---

## What I actually ran

| Step | Result |
|---|---|
| Protocol SHA / content read | done |
| `pytest tests/test_dc_alif_p0.py` | **6 passed in 5.02s** |
| `import run_collab_dc_alif` | imports OK, `main` present |
| Resolved the actual task list used by the runner | `('order','xor','strict_binding')` |
| Resolved the actual dev/confirm seeds used | `DEV=(35113,35119,35141)`, `CONFIRM=(35201…35261)` |
| Checked whether the protocol's named generators exist | **they do not exist** |
| Checked for any completed run | **no run, no artifact** |
| **Did not** run the 54-trial screen or P2 stream | NOT_RUN |

The P0 suite genuinely passes. That is real work and it is honestly reported.

---

## Findings

### `ds-001` — **critical** — the protocol specifies three generators that do not exist, and the runner silently uses three different tasks

Protocol §4.1 preregisters these tasks:

1. `delayed_cue` — `tasks21.py::generate_delayed_cue`
2. `strict_binding` — `binding_audit21.py::generate_strict_binding`
3. `order_sensitive_vision` — `binding_audit21.py::generate_order_sensitive_vision`

**None of those three functions exists in the tree.** The real names are
`tasks21.representation`, `tasks21.context_task` and `binding_audit21.strict_binding`,
and there is no vision generator anywhere. Both `delayed_cue` and
`order_sensitive_vision` — two of the three primary benchmarks — are **absent**.

The runner does not fail: it imports `TASKS` from `direct_eligibility_common`, which
provides `('order', 'xor', 'strict_binding')`. So the experiment that would run is a
**different experiment** from the one preregistered: `order` and `xor` replace
`delayed_cue` and `order_sensitive_vision`.

This matters beyond bookkeeping. **Gates are task-specific.** §4.1 requires
`mean(correct) − mean(baseline_elig) ≥ 0.05` **on `strict_binding`** and
`mean(correct) − mean(no_gate) ≥ 0.05` **on `strict_binding`**, plus the refit-head
gate on unseen `probe_test`. Those are defined against tasks the runner does not run.
And `order`/`xor` are exactly the two tasks my own branch and grok's found to be
solvable by a **frozen first-order readout** — so substituting them for the binding
tasks imports precisely the failure mode the campaign is trying to escape.

**Required action:** either implement the three named generators, or amend the
protocol to the actual task list *and* re-derive every gate against it, **before**
any run. Do not run 54 trials and then relabel.

### `ds-002` — **critical** — the preregistered seeds are not the seeds the runner uses

Protocol §4.1: development seeds `[101, 102, 103]`, confirmation `[201…208]`.
Actual, inherited from `direct_eligibility_common`: `DEV=(35113, 35119, 35141)`,
`CONFIRM=(35201, 35219, 35227, 35233, 35239, 35243, 35257, 35261)`.

The protocol text and the executable configuration disagree. Worse, those inherited
seeds belong to a **previous** experiment family (report 35's direct-eligibility
run), so the preregistration does not demonstrate fresh, unused seeds — the property
§4.1 claims to establish.

**Required action:** state the actual seeds in the protocol, verify they are unused
against `scripts/` and `evaluation/`, and re-commit. A preregistration whose seed
list does not match the code is not a preregistration.

### `ds-003` — **major** — §3.2's derivative is inconsistent with §3.1's definition

§3.1 defines the linear drive **with** bias and the coincidence term **without** it:

```
I_j(t) = b_j + Σ w·pre
q_j(t) = ½[(Σ w·pre)² − Σ (w·pre)²]     ← uses Σ w·pre, not I_j − b_j
```

§3.2 then writes the coincidence derivative as

```
∂q_j/∂w = pre·((I_j − b_j) − w·pre)
```

which is the derivative of `½[(I_j − b_j)² − …]`, not of the `q_j` actually defined
above. When `b_j ≠ 0` the two differ by `b_j·pre` per synapse. The hand-written
eligibility therefore does not analytically match the forward pass it claims to
differentiate.

The code follows §3.1 (`q_t = 0.5*(I_in*I_in - sq_pairs)` using `I_in` with no bias
subtraction), so **the code is self-consistent but the protocol's math is wrong**.
Given that §3.2's own "why this solves Theorem 3" argument rests on that recursion,
the argument needs re-derivation.

**Required action:** either set `b_j = 0` in the coincidence path explicitly, or
correct §3.2's derivative to `pre·((Σ w·pre) − w·pre)`. Then re-verify against the
finite-difference audit.

### `ds-004` — **major** — Theorem 1's `λ = 0.88` does not match the core it claims to analyse

§2.1 states the linear-subthreshold ceiling for `SpikingCore` with `λ = 0.88`. The
repository's `Config.leak` default is **0.85**, and the shipped `UnifiedCore` uses
**0.65** (`unified.py`: `leak=.65`). No configuration in the tree uses `0.88`.

The theorem's conclusion (`5/8 = 62.5%`) is presented as explaining report 35's
observed 62.24–63.54%, and the coincidence is suggestive. But a numerical ceiling
derived at the wrong decay cannot be claimed to *explain* a measurement made at a
different decay — the residual weights `λ^{10−p}` change materially
(0.85⁴ ≈ 0.52 vs 0.88⁴ ≈ 0.60).

**Required action:** recompute Theorem 1 at the decay actually used by the code under
test, and state which `Config` the theorem is about. If the ceiling moves with `λ`,
say so — the Theorem as written claims a specific number.

### `ds-005` — **minor** — `no_gate` ablation is confounded with an unregistered input-coactivation modifier

`dc_alif_eligibility.py:155-158` scales three coefficients by
`(1 − coact)²`, `(1 − coact)` and `(1 − coact)`, where `coact = compute_coact(x)` is
an input-derived term. None of `compute_coact`, `alpha_d_eff`, `alpha_s_eff` or
`beta_g_eff` appears in §3.1's forward equations, which define `α_d`, `α_s`, `β_g` as
constants (`α_d = 0.25`, `α_s = 0.35`, `β_g = 2.00`).

So the implemented neuron is not the neuron the protocol documents, and the `no_gate`
ablation sets `beta_g_eff = 0` **and** removes the `coact` path, so it ablates two
things at once. §3.4 claims `no_gate` isolates the dendritic coincidence mechanism; it
does not.

**Required action:** document `compute_coact` in the forward equations, or remove it.
Then make `no_gate` a single-variable ablation.

### `ds-006` — **minor** — no run exists, so no capability claim is checkable

There is no `runs/*dc-alif*` directory and no `artifacts/*dc-alif*` receipt. The
branch contains a protocol, a prototype, P0 tests and runners — no results.
`pytest tests/test_dc_alif_p0.py` passing is a **technical** result and is correctly
not presented as a capability result. But the protocol's §4.2 P2 retention gates
(`old_task ≥ 0.92`, `context_task ≥ 0.92`, `mean ≤ 0.03`, `worst ≤ 0.05`) and all P1
gates are unmeasured.

**Required action:** none for correctness — but the branch should not be read as
having demonstrated anything about learning. It has not run.

---

## What holds up

1. **P0 is real.** 6/6 tests pass in 5 seconds, covering the finite-difference
   eligibility audit, zero-leakage, matched-RNG `zero`/`frozen` equivalence, cold
   resume, rollback and cross-loader isolation. That is a well-built technical gate.
2. **The identity is respected.** One 48-neuron ALIF population, `fan_in=8`,
   `plastic_rows=12`, no serving-path edits, BPTT confined to a `control_only`
   class that other loaders reject. I confirmed no import of `torch` from the
   candidate module.
3. **The signal-boundary disclosure is explicit and correct.** §1 states
   `W_out.T @ error` is non-local, one-step, and never extended to hidden-to-hidden
   or reverse-time. That is the honest framing the campaign requires.
4. **The `no_gate` and `baseline_elig` ablations are the right idea** — attacking the
   mechanism rather than the tuning. The execution is confounded (`ds-005`), but the
   experimental instinct is sound.
5. **The 2 GB disclosure is honest.** It reports process-tree committed memory and
   notes the BPTT child is now sequential rather than claiming a per-process cap as a
   tree total.

## Theorems: plausible but unverified here

I did **not** independently verify Theorems 1–3. Theorem 1 is the load-bearing one
(it is the entire justification for the DC-ALIF mechanism), and it is stated at the
wrong `λ` (`ds-004`). My own branch's measurement is relevant and should be
cross-checked: on a different binding task I found the deterministic hard-threshold
`core.step` at `decay=0.25` leaves a **linearly readable** residue at 0.91 (label
shuffled 0.57), while the Bernoulli `features_many` path reads 0.48. Which sampler a
ceiling theorem assumes therefore changes its answer. `DC-ALIF` uses hard threshold,
so Theorem 1's linear-ceiling argument may simply not apply to it.

## Verdict: **CHANGES_REQUIRED**

Not `INVALID_EXPERIMENT` — the work is real, P0 passes, and the mechanism is
plausibly the most interesting proposal in this campaign. But as of `62247ff` the
preregistration does not describe the experiment the code would run (`ds-001`,
`ds-002`), and the mathematical derivation that motivates the mechanism is internally
inconsistent (`ds-003`) and stated at the wrong decay (`ds-004`). Running 54 trials
on this configuration would produce numbers whose gates cannot be interpreted.

**Reproduction scope, stated precisely:** I ran the P0 test suite (6/6 pass),
confirmed the runner imports, and resolved the effective task list and seed list from
the code. I did **not** run the screen, the confirmation, the P2 stream, or
independently verify Theorems 1–3. Those remain NOT_VERIFIED.

```json
SCARX_SIGNAL_V1
{
  "campaign": "scarx-neural-round1-20261111",
  "agent": "deepseek",
  "phase": "CROSS_REVIEW",
  "target_agent": "antigravity-gemini",
  "target_commit": "62247ff",
  "verdict": "CHANGES_REQUIRED",
  "reproduction_scope": "Ran pytest tests/test_dc_alif_p0.py (6 passed in 5.02s). Confirmed run_collab_dc_alif imports and exposes main. Resolved the effective TASKS=('order','xor','strict_binding') and DEV/CONFIRM seed tuples from direct_eligibility_common, and confirmed the three generators named in protocol section 4.1 do not exist. No run, no artifact present. Did not execute the 54-trial screen, confirmation, or P2 stream; did not verify Theorems 1-3.",
  "evidence_paths": [
    "experiments/collab/scarx-neural-round1-20261111/antigravity/protocol-v1.md",
    "src/scarx/evaluation/dc_alif_eligibility.py",
    "scripts/run_collab_dc_alif.py",
    "tests/test_dc_alif_p0.py"
  ],
  "findings": [
    {"id": "ds-001", "severity": "critical", "claim": "the experiment matches the preregistered design", "evidence": "protocol names generate_delayed_cue, generate_strict_binding, generate_order_sensitive_vision; none exist. Runner imports TASKS=('order','xor','strict_binding') from direct_eligibility_common, substituting two different tasks. Gates in section 4.1 are written against strict_binding and probe_test.", "required_action": "Implement the named generators or amend the protocol to the real task list and re-derive every gate before any run."},
    {"id": "ds-002", "severity": "critical", "claim": "preregistered fresh seeds are used", "evidence": "protocol says dev [101,102,103] and confirm [201..208]; actual DEV=(35113,35119,35141), CONFIRM=(35201..35261) inherited from the report-35 family.", "required_action": "Record the actual seeds, verify they are unused, re-commit the protocol."},
    {"id": "ds-003", "severity": "major", "claim": "section 3.2 derivative matches section 3.1 forward pass", "evidence": "q_j is defined from sum(w*pre) without bias; the derivative uses (I_j - b_j), which differs by b_j*pre per synapse when b_j != 0. Code follows 3.1, so the implementation is self-consistent but the protocol math is wrong.", "required_action": "Correct the derivative or set b_j=0 in the coincidence path, then re-verify the finite-difference audit."},
    {"id": "ds-004", "severity": "major", "claim": "Theorem 1 explains the report-35 62.5% ceiling", "evidence": "Theorem assumes lambda=0.88; Config.leak default is 0.85 and UnifiedCore uses 0.65. No config in the tree uses 0.88. The ceiling depends on lambda^(10-p).", "required_action": "Recompute at the code's actual decay and state which Config the theorem describes."},
    {"id": "ds-005", "severity": "minor", "claim": "no_gate isolates the dendritic coincidence mechanism", "evidence": "dc_alif_eligibility.py scales alpha_d, alpha_s, beta_g by an input-derived coact term absent from section 3.1. no_gate sets beta_g_eff=0 and removes the coact path simultaneously.", "required_action": "Document compute_coact in the forward equations or remove it; make no_gate single-variable."},
    {"id": "ds-006", "severity": "minor", "claim": "learning results exist", "evidence": "no runs/*dc-alif* and no artifacts/*dc-alif*. Only protocol, prototype, P0 tests and runners are present.", "required_action": "Do not present this branch as having demonstrated learning; P0 pass is technical only."}
  ]
}
```
