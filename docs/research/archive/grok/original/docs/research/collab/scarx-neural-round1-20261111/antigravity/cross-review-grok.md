# Cross-review of `grok` â€” slow forward eligibility

Reviewer: antigravity. Environment: Antigravity (Windows). Model: Claude Sonnet 5.5 (High) for this review; the branch itself was authored under Gemini 4 Argon (High).
BASE_SHA: `1735ad66ea9e7b416f5df15df99d9c13cea46b4e`
Reviewed: `research/grok/scarx-neural-round1-20261111` @ `d509878` (read-only; nothing was written inside that worktree, bytecode disabled, `-p no:cacheprovider`).
Reproduction driver: `scripts/review/repro_grok.py` (imports grok's own `grok_slow_trace_screen` helpers, one numerical thread).

## Reproduction

| Check | Grok reported | Reproduced | Match |
|---|---|---|---|
| `pytest tests/test_slow_trace.py` | "4 passed" (report) | **6 passed** (file has 6 tests) | yes, report count is stale |
| Information gate, fast feature probe | 0.484375 | 0.484375 | exact |
| Information gate, slow trace probe | 0.9375 | 0.9375 | exact |
| 71005 confirmation `slow-correct âˆ’ slow-frozen`, mean / 95% lower | 0.02734 / âˆ’0.00289 | 0.02734 / âˆ’0.00289 | exact (all 8 per-seed gains identical) |
| `slow-zero == slow-frozen` on every seed | yes | yes | exact |
| Train/dev vs confirmation example overlap | not stated | 0 / 0 identical sequences | clean |
| Cold-process resume of one learning step (`slow-correct`, seed 71037) | not stated | loss and dump SHA-256 identical in a fresh process | pass |

- Positive finding reproduced: an untrained slow trace carries the order bit (0.9375 vs 0.484).
- Negative control reproduced: the zero-signal arm equals the frozen arm, so the hidden update is not what produces the score.
- Cold resume reproduced.

## Assessment

1. **Verdict NO-GO is correct and the amendment is the right call.** Amendment 1 (confirmation previously scored on development split 71004) was a genuine leakage-class defect, was disclosed, and the historical file was preserved. The corrected CI lower bound âˆ’0.0029 < 0.05 stands.
2. **The headline contrast is against the wrong baseline.** The "slow vs fast-correct +0.457, lower 0.39" number measures a change of readout memory, not of learning. On 71005 `fast-correct` is only 0.494 (vs 0.451 in the superseded run) while `slow-frozen` is already 0.924. The report states this itself; reviewers should not quote the 0.39 bound as a learning result.
3. **Mechanism limit.** The slow trace is first-order in the spikes. It can hold *order* but cannot form a keyÃ—valueÃ—query conjunction. In my branch this separation is measured directly: the same readout-memory idea cannot pass `strict_binding` (a cross-neuron conjunction task) without a within-neuron product path.
4. **Statistical note.** The interval uses t with 8 gains where several are exactly 0 (the slow head saturates). With a 64-example test split, one example is 1.56 pp, so a lower bound within 0.3 pp of 0 is not distinguishable from 0. The protocol-locked gate is nevertheless the gate, and the report applies it as written.
5. **Novelty versus the stopped family.** Report 35 (`STOP_DIRECT_READOUT_ELIGIBILITY_V1`) stopped this readout-eligibility family. The branch argues it is a different mechanism (extra slow trace in the error path, not an LR/epoch retune); I accept that argument, and the NO-GO result means the stop rule is not being circumvented.
6. **Scope.** Only one 12-tick order task, 64-example splits, 2 epochs. Conversation, vision and retention were not measured, and the report says so.

## What I would not accept from this branch

- Any statement that the slow trace "teaches" the hidden layer.
- Counting `slow-correct` 0.957 (superseded run) as held-out.

## Verdict on the claims

| Claim | Reviewer status |
|---|---|
| Fast feature is not already a shortcut at 0.65 | supported (reproduced) |
| Slow trace readable before learning | supported (reproduced) |
| Hidden slow eligibility beats frozen by CI lower â‰¥ 0.05 | not supported (reproduced failure) |
| Hidden update necessary | not supported, zero == frozen reproduced |

No blocking technical defect found after Amendment 1.
