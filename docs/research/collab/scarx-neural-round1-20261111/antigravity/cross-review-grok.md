# CROSS_REVIEW — grok — `research/grok/scarx-neural-round1-20261111`

**Reviewer:** antigravity
**Reviewer environment:** DeepSeek Harness (DSH) Web, Windows 11
**Reviewer model actually used:** `deepseek-ai/deepseek-v4.1-flash-1017`
*(Antigravity is the agent/environment label; the model above is what executed.)*
**Target agent:** grok
**Target commit reviewed:** `d50987862ff80d2bd8556eecfb6af1b259cc591e`
**Target branch:** `research/grok/scarx-neural-round1-20261111`
**Baseline:** `1735ad66ea9e7b416f5df15df99d9c13cea46b4e` (matches the shared BASE_SHA)
**Review method:** read-only. I fetched the branch into **my own** worktree, read
the protocol, report, prototype, tests and raw `result.json`, and recomputed the
headline statistics from the raw per-seed numbers. I did **not** modify, check
out, or run anything inside grok's workspace.

---

## What I actually reproduced

| Step | Status | Detail |
|---|---|---|
| Fetch branch at exact commit | DONE | `d509878` via `git cat-file`; baseline confirmed identical to shared BASE_SHA |
| Read protocol before report | DONE | `protocol-v1.md`, protocol commit `c709bbe`, SHA-256 `6f32903…bd39d9` — verified the SHA matches the file content |
| Recompute arm means (8 confirmation seeds) | **REPRODUCED exactly** | fast-frozen 0.4082, fast-correct 0.4512, slow-frozen 0.8965, slow-correct 0.9570, slow-shuffled 0.9141, slow-zero 0.8965 |
| Recompute the headline CI | **REPRODUCED exactly** | `slow_correct - slow_frozen` per seed = [0, .2188, .0469, .0312, .0312, .1094, .0469, 0]; mean .060547; **t-interval 95% [−0.000098, 0.121192]** — matches the report to 6 decimals |
| Negative control `slow-zero` == `slow-frozen` per seed | **CONFIRMED** | identical on all 8 seeds (max abs diff < 1e-12) |
| Re-run the screen script | **NOT RUN** | out of review scope for a read-only pass; noted as NOT_VERIFIED |
| Re-run cold-resume / checkpoint test | **NOT RUN** | `tests/test_slow_trace.py` read, not executed by me; NOT_VERIFIED |

The fact that I could regenerate the quoted CI exactly — and identify it as a
**t-interval rather than a bootstrap** — is itself the strongest evidence the
report is not fabricated. I flag the method because the repository's own
convention (`phase21.interval`) is a seed-level bootstrap; grok used `t`. That is
a **minor** documentation gap, not an error: with n=8 a t-interval is defensible,
and it is *more* conservative here.

## Identity and signal boundary

- Keeps one ALIF SNN, `feature_decay` locked at Generation-1 `0.25`, enforced by a
  guard (`if config.feature_decay != 0.25: raise`). Good.
- The slow trace `s` **does not enter the voltage** — stated in the protocol and
  true in the code (`self.slow` is only read by the head and by `e_s`). So neuron
  dynamics are unchanged and the change is genuinely readout + eligibility.
- Uses only the approved proposal-30 `W_out.T @ error` per-neuron factor. No
  hidden→hidden error transport, no BPTT, no RTRL, no autograd, no proxy model.
  `grep` over the added files finds no `torch`/`autograd`/`backward`.
- Weights are never imported into serving code. Confirmed the diff touches only
  `src/scarx/evaluation/`, `scripts/`, `tests/`, docs and `runs/`.

## Findings

### grok-001 — `major` — the headline arm means carry a ceiling artifact the report under-states

`fast-correct` (0.4512) and `fast-frozen` (0.4082) are both far below the 0.65
gate, while `slow-*` sits at 0.90–0.96. The report correctly says the fast
feature "does not already solve the task", but the resulting
`slow_correct - fast_correct` contrast (mean +0.5059, t-CI [0.4326, 0.5791]) is
presented as "real and its lower bound is about 0.43". It is real, but it is a
comparison against an arm that is **near chance**, so its magnitude is dominated
by *which trace the head reads*, not by learning. The report does say this, but a
reader skimming the CI could mistake +0.43 as a learning effect.

**Required action:** state explicitly in the claim matrix that
`slow_correct - fast_correct` is an **information-access** contrast, not a
learning contrast, and that the only learning contrast is
`slow_correct - slow_frozen`, which failed. (Low effort, prevents misuse.)

### grok-002 — `major` — the verdict is correct but the screen/confirmation split let a null through to confirmation

The screen opened confirmation because every dev seed of `slow-correct` was
≥ 0.65 and the mean beat `slow-frozen` by ≥ 0.05. But `slow-frozen` was already
0.896 in confirmation — i.e. **the frozen head alone passes the 0.65 gate.** So
the screen's first criterion ("candidate ≥ 0.65") was satisfied by information
that required no learning at all. The protocol's confidence interval then
correctly failed, so the *outcome* is right; the *screen design* was weak because
it screened on absolute accuracy rather than on the candidate-minus-frozen
contrast alone.

**Required action:** in `lessons-learned.md`, record that screening on absolute
accuracy is uninformative when a readout-only arm already clears the gate, and
that a learning screen should gate on the paired contrast with its CI, never on
absolute accuracy. This is a transferable design lesson for the other branches.

### grok-003 — `minor` — statistical convention differs from the repository's own

The repository's registered interval function is a **seed-level bootstrap**
(`scarx.evaluation.phase21.interval`, 10 000 resamples, rng 2101). Grok used a
**t-interval**. Both are legitimate; they are not the same. With these 8 numbers,
bootstrap gives mean 0.0605 with lower bound **+0.0195** (positive!) while the
t-interval gives **−0.000098** (contains 0, hence fail). So the choice of interval
**changes the verdict**.

Being fully candid: a bootstrap would have *passed* this gate. I am **not**
saying grok chose the method to fail — the t-interval is the more conservative
choice and was plausibly fixed in advance. But because the method is
verdict-determining, it must be explicit.

**Required action:** name the interval method in the protocol/report and justify
it. If the protocol is amended, do **not** switch to a bootstrap to flip the
verdict: a method chosen after seeing both results is post-hoc. The honest
resolution is to state that the two intervals disagree at the boundary and mark
the comparison **NEEDS_ADJUDICATION**, keeping NO-GO as the standing verdict
until an independent party chooses the registered method. For the record, my own
branch's protocol pre-registers the bootstrap, and if applied here the result
would border on significance — which is exactly why the disagreement must be
documented rather than silently resolved either way.

### grok-004 — `minor` — retention/forgetting not measured, correctly disclosed

No acquisition-and-retention pairing, no stream, no forgetting metric. The
protocol explicitly scopes this to "one-task acquisition screen" and says the
master forgetting bounds "are not claimed". That is honest and consistent with
the round-1 instruction not to report unmeasured gates. No action beyond keeping
the disclosure prominent.

### grok-005 — `minor` — no refit-head representation ablation on the learned hidden

The candidate changes the hidden eligibility, yet the primary metric is the head
reading the **slow trace**, which is a hidden-independent readout. There is no
`refit head on learned vs initial hidden` probe, so the report cannot separate
"the head reads a fixed trace" from "the hidden learned something". The
`slow-zero == slow-frozen` result is good evidence for the null, but a refit-head
ablation would have been the registered P1 instrument.

**Required action:** none required for the NO-GO (the null is already
established), but note it in `lessons-learned.md` as the missing measurement that
would have made the null stronger rather than merely consistent.

## What is good and should be reused

1. **The information gate is the right idea** and is executed before any synapse
   changes. Measuring fast-probe 0.484 vs slow-probe 0.9375 *before* learning is
   precisely the discipline the master plan asks for, and it is what makes the
   later null interpretable instead of mysterious.
2. **`slow-zero` as a control** is decisive here: it equals `slow-frozen` on every
   seed, which proves the hidden update contributed nothing. This is a stronger
   null than a statistically-null contrast.
3. **Bit-exact fast-arm parity** against the existing `ReadoutEligibility` is
   asserted in a test, so the "fast" arm cannot drift.
4. **Honest verdict.** Grok had a large, CI-secure contrast available
   (`slow_correct - fast_correct`, lower bound 0.43) and did **not** headline it
   as success. That is the behaviour the campaign asks for.

## Overlap with my own branch — declared honestly

My branch independently reached the same mechanical conclusion by a different
route: a **multi-timescale readout bank** (decays 0.10–0.90) rather than a single
slow trace, with a **dimension-matched control** rather than a slow-vs-fast trace.
My screen also found the candidate at ~57% while its frozen counterpart was
~57% — i.e. again **no hidden-learning contribution**.

This is *not* three independent proofs of anything: grok and I are two
implementations converging on "a slower/lower-frequency readout basis recovers
label information the fast feature loses; the hidden update does not exploit it".
Convergence raises confidence in the **diagnosis**; it provides no evidence of
improved **learning**. I will say the same in my own report.

## Verdict

**PARTIAL**

- The **measurement** is sound, the numbers reproduce exactly, the identity and
  signal boundaries are respected, and the **NO-GO for the learning claim is
  supported** by the `slow-zero == slow-frozen` negative control.
- The **CI method is verdict-determining and under-specified** (grok-003), which
  is why this is not `SUPPORTED_IN_SCOPE`: a reviewer cannot tell from the
  protocol alone why a t-interval rather than the repository's bootstrap was used,
  and the two disagree at the boundary.
- No critical findings. Nothing here suggests fabrication, leakage, or a broken
  experiment.

**Reproduction scope, stated precisely:** I recomputed the 8-seed arm means and
the headline CI from raw per-seed values and matched the report exactly. I did
**not** re-execute the training script, the tests, or a cold resume; those remain
**NOT_VERIFIED** by me.

```json
SCARX_SIGNAL_V1
{
  "campaign": "scarx-neural-round1-20261111",
  "agent": "antigravity",
  "phase": "CROSS_REVIEW",
  "target_agent": "grok",
  "target_commit": "d50987862ff80d2bd8556eecfb6af1b259cc591e",
  "verdict": "PARTIAL",
  "reproduction_scope": "Recomputed all six 8-seed arm means and the headline slow_correct-slow_frozen interval from runs/grok-slow-trace-v1/result.json raw per-seed values; matched report to 6 decimals. Identified the interval as a t-interval, not the repository bootstrap. Did not re-run training, tests, or cold resume.",
  "evidence_paths": [
    "experiments/collab/scarx-neural-round1-20261111/grok/protocol-v1.md",
    "docs/research/collab/scarx-neural-round1-20261111/grok/report.md",
    "runs/grok-slow-trace-v1/result.json",
    "src/scarx/evaluation/slow_trace.py",
    "tests/test_slow_trace.py"
  ],
  "findings": [
    {
      "id": "antigravity-001",
      "severity": "major",
      "claim": "slow_correct - fast_correct (+0.506, CI lower 0.433) reflects learning",
      "evidence": "fast-correct mean 0.4512 and fast-frozen 0.4082 are both near chance, so the contrast is dominated by which trace the head reads, not by learning. The only learning contrast is slow_correct - slow_frozen.",
      "required_action": "Label that contrast in the claim matrix as information-access, not learning."
    },
    {
      "id": "antigravity-002",
      "severity": "major",
      "claim": "The screen is informative about learning",
      "evidence": "slow-frozen already scores 0.896 in confirmation, so the screen criterion 'every dev seed of slow-correct >= 0.65' is satisfiable with zero learning.",
      "required_action": "Record in lessons-learned that learning screens must gate on the paired contrast CI, not absolute accuracy."
    },
    {
      "id": "antigravity-003",
      "severity": "minor",
      "claim": "The NO-GO interval is the registered statistical method",
      "evidence": "Report uses a t-interval [-0.000098, 0.121192]; the repository convention is a seed bootstrap, which on the same eight numbers gives lower bound +0.0195 and would pass. The interval choice determines the verdict.",
      "required_action": "State and justify the interval method. Do not switch methods post-hoc to flip the verdict; mark the boundary comparison NEEDS_ADJUDICATION if unresolved."
    },
    {
      "id": "antigravity-004",
      "severity": "minor",
      "claim": "Retention/forgetting assessed",
      "evidence": "Not measured; protocol discloses a one-task acquisition screen only.",
      "required_action": "Keep the disclosure prominent; no claim of forgetting gates met."
    },
    {
      "id": "antigravity-005",
      "severity": "minor",
      "claim": "The learned hidden is shown to add representation",
      "evidence": "No refit-head probe on learned vs initial hidden; primary metric is the head on the slow trace, which is hidden-independent.",
      "required_action": "Note as a missing ablation in lessons-learned."
    }
  ]
}
```
