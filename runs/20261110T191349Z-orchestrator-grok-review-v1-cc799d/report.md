# Orchestrator review — Grok 14bf4b2

Reviewed commit `14bf4b2574d77052e3c8d4d682238b2dbd5320c5`, PR#2.
Baseline1735ad6. Independent detached worktree; no changes to Grok's files.
**REQUEST_CHANGES before research merge; no production integration.**

## P1 — Confirmation reuses development data

The confirmation loop in `scripts/grok_slow_trace_screen.py` calls
`score_arm(model, development, development_labels)` on71004. The locked protocol
names71005 as confirmation data, but the script never generates it. Eight fresh
model seeds do not make the reused data a held-out confirmation set.

Independent recomputation: every saved screen/confirmation accuracy matches
the predictions against `development_labels`; the protocol SHA matches.
This verifies DEV measurements, not the registered confirmation experiment.

Preserve the old result and mark this scope explicitly. Register a technical
correction, evaluate the originally registered71005split after the unchanged
screen, and record its hash/labels/predictions separately. Do not alter seeds,
budget, threshold or final set because of disappointing scores.

## P2 — Checkpoint loader accepts ambiguous or unsupported state

`SlowTraceEligibility.loads` checks only set(member names), without exact count
or aggregate uncompressed bounds. Extra arrays are checked for shape/finite,
but not dtype. Safe small-checkpoint repros at this commit:

- Duplicate `extra.npz` member accepted.
- Float64 slow/e_s accepted and silently cast to float32; a nextafter value
  above0.25 loses information on load.

Require3unique outer members, bounded metadata and inner NPZ sizes before
allocation, exact float32 arrays and valid slow-state range. Add negative tests
that reject without mutation. Payload checksums do not replace schema checks.

## P2 — Extra arrays omitted from memory accounting

The inherited `memory_bytes()` reports29666bytes for both base and slow models.
The slow and e_s arrays add3264bytes at H48,D8,K8 but are omitted. Override the
estimator and test it. This small omission does not imply this run exceeded2GB.

## P2 — Result overwrite and incomplete measurement recipe

The runner writes a fixed `runs/grok-slow-trace-v1/result.json`; the report says
it ran twice. Use unique run directories or refuse existing output. Preserve
old protocol/results. Record command, exit code, UTC, source commit and all
numerical thread settings before NumPy import. Edge-only RSS measurements are
sampled maxima; use OS high-water or a monitor for a peak claim.

Include the CI calculation/method/seed in reproducible code. No change to the
registered0.05 lower bound is requested.

## What should remain

The code keeps one ALIF population and no serving import, external model or
candidate autograd. The information gate and frozen/zero/shuffled controls are
useful. The report correctly separates readable slow state from learned hidden
representation and does not advertise95.7% as generalchat or lifelong learning.
The conservative NO-GO is appropriate; non-significance alone does not prove
zero possible benefit. Frozenhead already meeting65% does show hidden updates
are unnecessary for that task threshold.

## Independent evidence

- Detached14bf4b2: `pytest -q tests/test_slow_trace.py` →4passed.
- Protocol hash matches; all saved DEV accuracies recompute exactly.
- Tiny checkpoint repros and memory accounting checks above completed.
- No training rerun, main merge/push or live session modification.

Grok should amend its own branch, preserve14bf4b2evidence and provide a new
commit. Valid negative research evidence can be merged after these corrections
without deploying the learner.
