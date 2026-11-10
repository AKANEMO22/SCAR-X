# Grok report — slow forward eligibility, round 1

Environment: Cursor on Windows. Model: Grok 4.7.
Ponytail was not installed, so it was not used.
Baseline: `1735ad66ea9e7b416f5df15df99d9c13cea46b4e`
Protocol commit, made before the formal run: `c709bbe01f97c38c84924a554bb787db801701bd`
Protocol SHA-256: `6f3290337275ec37da21d69d402455961a455e66505fc06caebba45866bd39d9`
Raw result: `runs/grok-slow-trace-v1/result.json`
Command, from a clean checkout of this branch, with `PYTHONPATH=src` and one numerical thread:

```powershell
python -m pytest tests/test_slow_trace.py -q
python scripts/grok_slow_trace_screen.py
```

The second run of the same script, which also stored development predictions, reproduced the same accuracies. It took 13.43 seconds. Peak RSS was 39,309,312 bytes and peak private bytes were recorded in the result file. There was no child process. Both are far under 2 GB. Technical tests: 4 passed.

## Hypothesis

Generation-1 feature decay is 0.25. Proposal 30 multiplies `W_out.T @ error` by the forward sensitivity of that fast feature. This prototype keeps that decay and that one-step error, and instead multiplies the error by the forward sensitivity of an extra spike trace with decay 0.95. The trace does not enter the voltage. Trial 35 is still STOP; this was not an learning-rate or epoch retune of that run.

## Information gate

Untrained ridge probes on the development sequences:

| State | Development probe |
|---|---|
| Fast features, decay 0.25 | 0.484375 |
| Slow trace, decay 0.95 | 0.9375 |

The fast feature does not already solve the task at the 0.65 shortcut threshold. The slow trace does hold the order bit before any synapse update. That is an information fact, not a learned skill.

## Learning result

The screen rule opened the eight confirmation seeds because every development seed of `slow-correct` was at least 0.65 and its mean sat above the frozen slow head and at least 0.05 above the fast `e_f` arm.

Confirmation, eight fresh initializations, development accuracy of the actual head:

| Arm | Mean |
|---|---|
| fast-frozen | 0.408 |
| fast-correct | 0.451 |
| slow-frozen | 0.896 |
| slow-correct | 0.957 |
| slow-shuffled | 0.914 |
| slow-zero | 0.896 |

`slow-zero` matched `slow-frozen` on all eight seeds. The 95% interval of the eight `slow-correct - slow-frozen` gains is `-0.000098` to `0.121`. The pre-registered lower bound was 0.05. It failed.

The large gap versus `fast-correct` is real and its lower bound is about 0.43, but the frozen slow head already scores 0.896. The gap is the head reading a trace that still contains the bit, not evidence that the hidden update learned the order. The shuffled-signal arm stays high, so the directed eligibility is not what produces the score.

## Amendment 1 — confirmation split 71005

Codex review of `14bf4b2` showed that the confirmation block scored development seed 71004. That file is preserved at `runs/grok-slow-trace-v1/result.json` and is not a held-out confirmation. The corrected run is `runs/grok-slow-trace-v1a-confirmation-71005/result.json`. Seeds, budget and the 0.05 bound were not changed. The development screen still opened confirmation. On 71005 the eight-seed gain versus `slow-frozen` has mean 0.0273 and 95% lower bound **-0.00289**. `slow-zero` again matches `slow-frozen` on every seed. The gate still fails.

## Verdict

**NO-GO** for the claim that slow eligibility teaches the hidden synapses a delayed order bit that the fast path and a frozen hidden lack. The corrected confirmation split does not reverse that verdict.

What stands: at this decay, the fast feature is a poor linear readout of the delayed order, and a slow local trace is a strong one, before learning. What does not stand: calling the candidate a better learner. Conversation, vision, and retention were not measured.

## Claim matrix

| Claim | Metric | Source | Scope | Limit |
|---|---|---|---|---|
| Fast features are not already a 0.65 shortcut | ridge probe 0.484 | `result.json` information_gate | this 12-tick order task | not a general encoding theorem |
| Slow trace is linearly readable before learning | ridge probe 0.9375 | same | same | probe is not the serving head and not a skill |
| Hidden slow eligibility beats readout-only by CI lower bound 0.05 | lower bound -0.000098 | eight confirmation seeds | same | failed |
| Hidden update is necessary | zero signal equals frozen | same | same | negative control passed against the learning claim |
