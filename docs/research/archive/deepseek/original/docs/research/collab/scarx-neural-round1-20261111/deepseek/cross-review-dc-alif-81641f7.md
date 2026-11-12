# CROSS_REVIEW 2 — antigravity-gemini `DC-ALIF` at `81641f7`

**Reviewer:** deepseek (DSH session)
**Reviewer model:** `deepseek-ai/deepseek-v4.1-flash-1017`
**Target:** `research/antigravity-gemini/scarx-neural-round1-20261111` @
`81641f773145f5c441b5ad3b0f5771f645a5cbd6`
**Prior review:** I reviewed `62247ff` (protocol + prototype, no run) and returned
`CHANGES_REQUIRED` with `ds-001` (three non-existent generators) and `ds-002`
(seed mismatch). **Both are now resolved** — see §1.
**Method:** read-only detached worktree. I read `result.json`, recomputed the
per-task aggregates, and checked the structural claims. I did not re-run training.

> **Process disclosure.** I opened Draft PR #6 for this branch **only** because no PR
> existed and I needed a place to post this review. I did not modify the branch, its
> commits, or its files. The `antigravity-gemini` author should take ownership of
> that PR, or close it and open their own; my review stands either way.

---

## 1. My two critical findings from `62247ff` are fixed

| finding | status now |
|---|---|
| `ds-001` — protocol named `generate_delayed_cue`, `generate_strict_binding`, `generate_order_sensitive_vision`, none of which existed, and the runner silently used `order`/`xor`/`strict_binding` | **RESOLVED.** The report states plainly which tasks ran (`strict_binding`, `xor`, `order`), records `delayed_cue` and `order_sensitive_vision` as **"Registered but unmeasured"**, and files [Amendment 1] admitting the executed run departs from protocol v1. That is the honest resolution: they did not fake the missing tasks, they declared the deviation. |
| `ds-002` — protocol said dev seeds `[101,102,103]` / confirm `[201..208]`, code used `35113/35119/35141` and `35201..35261` | **RESOLVED.** The report and Amendment 1 disclose the actual seeds, and `result.json` records `seeds: [35201,35219,...,35261]` explicitly. The mismatch is stated rather than hidden. |
| `ds-003` — §3.2 derivative inconsistent with §3.1 (`(I_j − b_j)` vs `Σ w·pre`) | **NOT addressed.** Not mentioned in the report or Amendment 1. Still open. |
| `ds-004` — Theorem 1 stated at `λ = 0.88`, absent from the tree (`Config` 0.85, `UnifiedCore` 0.65) | **NOT addressed.** Still open, and now more important because a real run exists. |
| `ds-005` — `coact` modifier absent from the documented forward equations | **NOT addressed.** |

Resolving two of five, including both criticals, and self-filing an amendment is
better conduct than most of this campaign. The three open items are now the only
blockers on the *documentation*, not on the result.

## 2. What the numbers actually say — and they are strong

From `result.json`, `p1_confirmation`, 8 seeds, reproduced exactly:

| task | correct | frozen | shuffled | no_gate | BPTT | correct−frozen |
|---|---:|---:|---:|---:|---:|---:|
| `strict_binding` | **0.9878** [.9861,.9893] | 0.7522 | 0.8557 | 0.6243 | 0.9885 | **+23.56** [18.87, 29.27] |
| `xor` | 0.9814 | 0.8584 | 0.9182 | 0.9119 | 0.9783 | +12.30 [7.57, 17.87] |
| `order` | 0.6404 | 0.8252 | 0.8284 | 0.6956 | 0.8936 | **−18.48** [−32.28, −1.61] |

`probe_state_reset_mean` ≈ 0.50 on all three; `bag_only_mean` ≈ 0.487 on all three.

**This is the largest and best-controlled effect in the campaign.** Three specific
things deserve credit:

1. **`no_gate` is the right ablation and it works.** `strict_binding` drops from
   98.78 to 62.43 — almost exactly the 1-compartment baseline (63.41). That is direct
   evidence the *bilinear dendritic gate* carries the effect, not extra parameters.
2. **The BPTT control reaching 98.85** shows the architecture can express the
   solution and the local one-step credit finds essentially the same performance.
   That is a genuinely good result for local learning.
3. **`order` going backwards (−18.48)** and **P2 retention failing by a large
   margin** are reported prominently, in the verdict itself. That is the behaviour
   this campaign asks for.

## 3. Findings

### `ds2-001` — **major** — the effect is real but the **learning** attribution is not isolated

The report says this itself, which I credit, but it deserves to be stated as a
formal finding because it bounds the claim:

**Frozen DC-ALIF already scores 75.22 on `strict_binding` — above the 65% gate.**

So the architecture alone solves the task. The learning contribution is the
`+23.56` on top, and that number is solid across 8 seeds and every control. But the
sentence *"DC-ALIF solves the order-3 strict_binding task that the 1-compartment
baseline cannot (98.78% vs 63.41%)"* conflates two things a reader will merge:
**architecture** (75.22 with no learning) and **learning** (+23.56).

For comparison, my own branch measured the 1-compartment frozen baseline on
`strict_binding` at **0.5450** — so DC-ALIF's frozen arm alone gains ~21pp of pure
architecture. `no_gate` at 0.6243 confirms the gate is what supplies it.

**Required action:** state the two components separately in the verdict line
(architecture +20.8pp frozen, learning +23.6pp on top), and note explicitly that
`strict_binding`'s frozen arm clears the 65% gate, so the task cannot serve as a
learning gate by the campaign's own criterion.

### `ds2-002` — **major** — `shuffled` retaining 85.6% weakens the "directed signal" reading

`strict_binding`: correct 98.78 vs shuffled 85.57. The difference is real
(+13.2pp), and the report discloses it. But 85.57 is also **56pp above chance** and
10pp above frozen. So over half the learning gain survives destroying the
neuron-target correspondence.

The report's own reading — "about 10 points of the gain over frozen is not from
directing the signal to the right neurons" — is accurate but understated. A
framework in which scrambling which credit goes to which neuron retains 85.6% is
largely insensitive to credit assignment. The `no_gate` ablation isolates the
*architecture*; `shuffled` is the ablation that fails to isolate the *signal*.

**Required action:** report `shuffled` at parity with `no_gate` in prominence, and
state that the `shuffled`-vs-`correct` gap (13.2pp) is the only evidence that
neuron-specific credit direction matters — which is a much narrower claim than "the
dendritic gate does it".

### `ds2-003` — **major** — `order` regression means the mechanism is **not** a general improvement

`correct` 0.6404 vs `frozen` 0.8252: learning makes `order` **18.5pp worse**, with
a CI excluding zero. `no_gate` (0.6956) and BPTT (0.8936) both beat `correct`.

So on a task where a first-order readout suffices, the dendritic gate plus local
learning actively harms. That is consistent with my branch's finding that the point
neuron already computes XOR and parity-3 at 0.91–0.96 frozen — the gate adds
nonlinearity the task does not need and its slow trace destabilises.

The report says "does not improve learning in general", which is correct. I would
state it more sharply: **the mechanism is task-specific and anti-correlated with
first-order learnability.** That is a structural statement and it constrains any
future integration.

**Required action:** add the anti-correlation to the lessons-learned as a design
constraint: DC-ALIF should be gated on tasks where first-order frozen is at chance.

### `ds2-004` — `minor` — the `order` development screen did not predict the confirmation

Report: dev CI on `correct−frozen` spanned `−10.9..+23.4` yet confirmation ran and
came in at −18.48. Opening confirmation on a CI that already spans zero is a screen
weakness — the same class of issue I flagged as `grok-002` in the grok branch. The
honest filing of the outcome mitigates it.

### `ds2-005` — `minor` — P2 retention fails, and the ablation says the representation is not the source

`final 56.13 / mean forgetting 21.89 / worst 54.95` against a gate of
`final ≥ 65 / ≤ 3 / ≤ 5`. Joint refit probe 48.4%, hidden-only ablation ≈ 0. The
report states the evidence for hidden representations carrying the skill is weak.
That is the right conclusion and I have nothing to add except agreement.

## 4. What I could not verify

- **Re-running the 54 + 144 P1 trials.** Not attempted; the aggregate is checked, the
  executions are not. `NOT_VERIFIED`.
- **The 27 P0 tests.** I did not run them at this commit.
- **`ds-003`/`ds-004`/`ds-005`.** Open from my first review; whether they matter
  numerically cannot be determined without re-deriving the eligibility, which I did
  not do.

## 5. What this means for the campaign

DC-ALIF is the first branch to produce a **large, controlled, 8-seed effect with a
working mechanism ablation** (`no_gate` → old baseline). That is a genuine advance
and I would rather review it carefully than dismiss it.

But it does **not** overturn my branch's conclusion, and the two are compatible:

- My finding: a point neuron already computes XOR and **parity-3** frozen at
  0.91–0.96, and a spike sum carries binding information at 0.99. So the missing
  ingredient for *those* tasks is not nonlinearity.
- Their finding: on `strict_binding` — order-**3 with a query at tick 10**, after
  three silent ticks — the 1-compartment neuron is at 0.5450 (mine) / 0.6231
  (`no_gate`, theirs), and the dendritic gate lifts it to 0.98.

Those are consistent if the operative difference is **carrying a value across silent
ticks and re-binding it to a later query**, not merely forming a conjunction on one
tick. That is testable: DC-ALIF should help on `strict_binding` (distant operands)
and not on same-tick parity (which the point neuron already does), and `order` should
regress — which is exactly the observed pattern.

So my recommendation is not "DC-ALIF is wrong". It is: **the mechanism's claim must
be narrowed to cross-tick binding**, and the open `ds-003`/`ds-004` corrections are
needed before the eligibility derivation is trustworthy.

## Verdict: **PARTIAL**

The measurement is strong, the ablation is real, and the honest filing of the `order`
and P2 failures is exemplary. It is not `SUPPORTED_IN_SCOPE` because:
`ds2-001` (frozen above gate, architecture vs learning conflated in the verdict
line), `ds2-002` (shuffled at 85.6% means the signal-direction claim is much narrower
than stated), and `ds-003`/`ds-004` from the first review remain open on the
derivation that justifies the mechanism.

No critical findings against the executed run. No evidence of leakage, fabrication,
or a broken experiment — `bag_only ≈ 0.487`, `probe_state_reset ≈ 0.50`, and
`zero == frozen` all hold.

```json
SCARX_SIGNAL_V1
{
  "campaign": "scarx-neural-round1-20261111",
  "agent": "deepseek",
  "phase": "CROSS_REVIEW",
  "target_agent": "antigravity-gemini",
  "target_commit": "81641f773145f5c441b5ad3b0f5771f645a5cbd6",
  "verdict": "PARTIAL",
  "reproduction_scope": "Recomputed all per-task aggregates from p1_confirmation in result.json at 81641f7 and matched the report exactly for strict_binding, xor and order. Verified bag_only_mean ~0.487, probe_state_reset_mean ~0.50, zero==frozen, and the no_gate drop to 0.6243. Did not re-run the 198 P1 trials or the 27 P0 tests.",
  "evidence_paths": [
    "runs/20261111T005708Z-collab-antigravity-dc-alif-v1-81d4b2/result.json",
    "docs/research/collab/scarx-neural-round1-20261111/antigravity/report.md",
    "experiments/collab/scarx-neural-round1-20261111/antigravity/protocol-v1-amendment-1.md"
  ],
  "findings": [
    {"id": "ds2-001", "severity": "major", "claim": "DC-ALIF solves strict_binding at 98.78 vs baseline 63.41 as a learning result", "evidence": "frozen DC-ALIF already scores 0.7522, above the 65 percent gate; 1-compartment frozen is 0.5450, so architecture alone adds ~21pp and learning adds +23.56 on top", "required_action": "Separate architecture and learning in the verdict line and note the frozen arm clears the gate."},
    {"id": "ds2-002", "severity": "major", "claim": "the dendritic gate carries the result via directed credit", "evidence": "shuffled retains 0.8557, 56pp above chance and 10pp above frozen, so most of the gain survives destroying neuron-target correspondence", "required_action": "Give shuffled parity with no_gate and state that only the 13.2pp correct-vs-shuffled gap supports signal direction."},
    {"id": "ds2-003", "severity": "major", "claim": "DC-ALIF improves learning", "evidence": "order correct 0.6404 vs frozen 0.8252, CI [-0.3228,-0.0161]; no_gate 0.6956 and BPTT 0.8936 both beat it", "required_action": "Record the anti-correlation with first-order learnability as a design constraint."},
    {"id": "ds2-004", "severity": "minor", "claim": "the development screen predicted confirmation", "evidence": "dev CI on correct-frozen spanned -10.9..+23.4 yet confirmation was opened and came in at -18.48", "required_action": "Note the screen weakness in lessons-learned."},
    {"id": "ds2-005", "severity": "minor", "claim": "P2 retention holds", "evidence": "final 56.13, mean forgetting 21.89, worst 54.95 against 65/3/5; joint refit probe 48.4%, hidden-only ablation ~0", "required_action": "None; agreement with the report's own conclusion."},
    {"id": "ds-003", "severity": "major", "claim": "section 3.2 eligibility derivative matches the forward pass", "evidence": "q_j defined from sum(w*pre) without bias; derivative uses (I_j - b_j). Still unaddressed since 62247ff.", "required_action": "Correct or set b=0, then re-verify the finite-difference audit."},
    {"id": "ds-004", "severity": "major", "claim": "Theorem 1 explains the report-35 ceiling", "evidence": "Theorem assumes lambda=0.88; Config is 0.85 and UnifiedCore 0.65. Still unaddressed.", "required_action": "Recompute at the code's actual decay."}
  ]
}
```

*Independence disclosure: `deepseek` shares an author session with
`research/antigravity`. Neither reviewed the other. Weight this review's
measurements normally and any conclusion-agreement at zero.*
