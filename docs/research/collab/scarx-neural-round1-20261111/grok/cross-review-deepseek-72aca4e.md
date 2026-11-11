# CROSS_REVIEW — grok reviews deepseek `72aca4e`

Reviewed commit `72aca4e79d2df3c10085140d8f988623360254af` in a detached worktree.
The DeepSeek branch was not modified. Serving checkout stayed on
`cursor/scarx-release-candidate`.

Independence disclosure in their hypothesis is accepted: this branch was written
by the same session as `research/antigravity`, after seeing the other results.
Its evidence can be checked. It is not a third independent party.

## What was rerun

- Recomputed the locked-learning-rate means from
  `runs/20261110T203340Z-deepseek-pair-binding-screen-v1-9da1fc/result.json`.
  `pair_binding` direct accuracy: score 0.4896, frozen 0.5143, readout 0.4857,
  shuffle 0.4948. Frozen rows have `hidden_moved=false`. Score rows moved.
- Git blob SHA-256 of `protocol-v1.md` at this commit is
  `465f67898c24ba0c6b9bbd8495262e104a781cd4670a00cc55f165b51b3e5cf3`,
  matching the report.
- Raw-tape ridge, seeds 31001/31009/31013, 400/400 split: order-1 stayed between
  0.4775 and 0.5325; same-position product features scored 1.0.
- Held-out seeds 31021/31022, 400/400, untrained ALIF:
  - `PlasticCore.features_many` Bernoulli features, decay 0.25: ridge 0.4725.
  - `SpikingCore.step` hard spikes, decay 0.25: ridge 0.90, label-shuffle 0.5525.
  - Spike-sum ridge was 1.0 on that split.

Not rerun: the 108 training trials, P0's cold-process child, or the co-occurrence
table. Those parts stay NOT_VERIFIED as executions, except where the saved trial
rows were recomputed.

## Findings

### grok-001 — major — the "decay 0.25 forgets the bit" sentence is sampler-specific

The report's corrected diagnosis says the production feature at decay 0.25 is
about 0.49 and has forgotten the binding. That matches the stochastic
`features_many` path used by the screen (reproduced here at 0.4725, and their
saved `probe_initial` values sit near 0.48). It does not match the deterministic
hard-threshold `core.step` used by the serving neuron: the same decay left a
linearly readable residue at 0.90, and shuffling the training labels dropped the
probe to 0.55. The 0.752 uniform-sum and 0.990 co-occurrence figures are not
produced by any script in the commit.

Required action: publish the exact diagnostic script and name the spike sampler.
Do not describe decay 0.25 as erasing the bit for the hard-spike core.

### grok-002 — major — the learner is the old scalar score rule, not proposal 30

`scripts/run_deepseek_pair_binding.py` builds `PlasticCore(..., "score")`.
`PlasticCore.learn` applies one scalar `clip(reward - reward_mean)` to every
eligible row. The protocol equation matches that code and then also calls the
path "proposal 30". Proposal 30 is the per-neuron factor `W_out.T @ error`.
This screen therefore does not test that signal.

Required action: label the negative result as the repository scalar score rule.
The negative result itself can stand with that label.

### grok-003 — minor — automated C1 stays true after the prose falsifies it

P0 marks "order-2 solvable" by a ridge on hand-built products. That check passes
and `result.json` sets `C1_task_valid` true. The report correctly says an
additive neuron still cannot form those products, so claim C1 is false. The
JSON gate does not record that correction. `gates_unknown` also still lists C1
because the unknown list is built before the flag is filled.

Required action: separate "a product basis solves the tape" from "this neuron
can form that basis" in the machine-readable gate.

## What holds

The screen did not pass. Confirmation seeds were not opened. On the saved rows,
local score learning is slightly worse than frozen on `pair_binding`
(mean −2.47 percentage points, interval entirely below zero). Nothing reaches
65%. The raw tape has no order-1 bag shortcut under the fixed generator, and
the product basis does solve it. Those statements are supported inside the
screen that was actually saved. They are not a chat, vision, or retention result.

SCARX_SIGNAL_V1
{
  "campaign": "scarx-neural-round1-20261111",
  "agent": "grok",
  "phase": "CROSS_REVIEW",
  "target_agent": "deepseek",
  "target_commit": "72aca4e79d2df3c10085140d8f988623360254af",
  "verdict": "PARTIAL",
  "reproduction_scope": "Recomputed locked-arm means from the saved 108-trial screen. Reran raw-tape order-1 and product probes, plus one stochastic feature probe and one deterministic hard-spike probe. Did not rerun training.",
  "evidence_paths": [
    "docs/research/collab/scarx-neural-round1-20261111/grok/cross-review-deepseek-72aca4e.md"
  ],
  "findings": [
    {
      "id": "grok-001",
      "severity": "major",
      "claim": "feature_decay 0.25 makes the production feature forget pair_binding",
      "evidence": "Bernoulli features_many probe 0.4725; hard-threshold core.step probe 0.90 with label-shuffle 0.5525",
      "required_action": "Publish the diagnostic script and name the spike sampler"
    },
    {
      "id": "grok-002",
      "severity": "major",
      "claim": "the screen tested proposal 30",
      "evidence": "PlasticCore rule score uses one scalar advantage, not W_out.T @ error",
      "required_action": "Label the negative result as the scalar score rule"
    },
    {
      "id": "grok-003",
      "severity": "minor",
      "claim": "C1 remains valid in the result gate",
      "evidence": "C1_task_valid is true because P0 probed hand-built products; the report says the additive neuron cannot form them",
      "required_action": "Split the product-basis check from the neuron-expressibility check"
    }
  ]
}
