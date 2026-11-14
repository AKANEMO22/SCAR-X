# History P0 stopped at the unchanged CPU/CUDA parity gate

**P0 did not pass. No DEV or live promotion.** Hermes remains in GPU mode with
the existing RC8 payload, unchanged weights/RNG/replay and five live updates.
The user's execution steering was to retain GPU and use Antigravity; CPU calls
were short numerical controls on disposable research objects, never a serving
mode switch.

[Locked specification](../../experiments/hybrid-update-history-p0-v1.md),
[manifest](../../runs/20261114T173119Z-history-ablation-p0-v1/manifest.json),
[original result](../../runs/20261114T173119Z-history-ablation-p0-v1/result.json),
[independent data rescore](../../runs/20261114T173119Z-history-ablation-p0-v1/independent-data-audit.json).

## What executed

The full intended matrix was five arms ×two seeds,490 actual method calls
including parity, cold next-update, replay and rollback. The persistent global
cap was600,100 per arm/seed and300 seconds. Eight pure numerical/envelope tests
passed before execution with **zero model.learn calls**. The first pure run
caught our Windows lock unlink-before-close mistake; fixed code and both
receipts are retained. The deliberate duplicate-ZIP-member test emits a warning.

The actual matrix stopped after143 charged calls. Only seed260313 ran; frozen
and readout-only completed their numerical/persistence controls. Current-hidden
completed17 steps then failed the loss parity check at step18. Current-only,
matched and the second seed did not execute learning trajectories. No seed or
failed sample was replaced, no tolerance was loosened and the result stays FAIL.

A separately recorded bounded failure diagnostic replayed17 current-hidden
updates on each device, then examined the next eight **forward-only** ticks.
Those34 additional updates share the original budget: **177 total**, maximum87
for one arm/seed, charged-call span106.33s. No attempt was refunded or moved to a
new ledger. The full490-call P0 was not rerun or relabeled complete.

## Observed numerical cause

Before the failing sequence, the largest CPU/GPU state difference was
7.15×10⁻⁷. The first seven forward ticks had identical spikes. At tick8, neuron91
had the following pre-spike margin (voltage minus threshold):

| | CPU | CUDA0 |
|---|---:|---:|
| Margin |0|−1.7881393432617188×10⁻⁷|
| Spike under the existing `margin >= 0` equation |1|0|
| Subsequent target probability |0.8587650061|0.8690466285|
| Loss |0.1522599613|0.1403584975|

The loss gap0.01190146 exceeds the locked3e-5 gate. Raw margins/spikes are saved
for every tick, and the independent auditor recalculates threshold decisions
and loss from those arrays. This supports a **near-boundary FP32 hard-spike
divergence** in this instance, not a claim that either device is intrinsically
more correct or that all training diverges. Earlier parity receipts on other
fixtures stay valid for their own scope; they never guaranteed all trajectories.

The failure occurs on the existing current rule before testing the new history
variants. It therefore does not establish whether dropping momentum improves
representation or retention. Independent CPU/GPU trajectories of a discontinuous
spiking system need careful numerical certification; changing the acceptance
definition after this outcome would require a separately locked amendment.
No epsilon was added to thresholds, no precision changed and no DEV was opened.

## Audit and resources

`scripts/audit_hybrid_history_p0.py` checks hashes, the append-only budget chain,
17 recorded hidden history/energy/delta calculations, norm after clipping, raw
spike differences and recomputed loss. It adds no forward/backward/update calls.
This is independent calculation by the same orchestrator, not an outside peer
approval. Actual CUDA updates were FP32; FP64 was used only for scalar norm/loss
diagnostics. The copied method differs from the pinned base in exactly the
registered three hidden-update statements; an AST regression checks that scope.

P0 process peak RSS757,800,960 bytes, private commit781,299,712 bytes; sampled
physical GPU free memory≥2,769MiB. These concern this research process and point
samples, not whole Hermes/system peaks. No VLM/ASR or web calls. Diagnostic buffers
are temporary audit copies, not new learned capacity. Research checkpoints live
under `artifacts/20261114T173119Z-history-ablation-p0-v1/`, with incompatible
research envelopes; raw source/arrays/results are in the run directory.

[Live-state audit](../../runs/20261114T173119Z-history-ablation-p0-v1/live-state-audit.json)
confirms all saved neural arrays and learned metadata/RNG/replay remain equal,
all55 RC8 payload files still match, and VLM/vision, ASR and ALIF remain GPU.
Localhost received actual per-step loss/delta and the failure diagnostic with
actor `orchestrator`, never added to live counters or called model thoughts.

## Antigravity contribution and limits

`gemini-3.8-flash-high`, effort high, completed an inline public-spec review in
43.7s. Its prompt/result are retained in the run; no private data or weights were
sent. Useful concerns were reachability, preserving alpha1, explicit checkpoint
identity and a persistent precharged budget. Some recommendations were rejected:

- Bisection can hit alpha1 after eight exact halvings of[0,256]; the blanket
  “cannot hit1 in20 steps” wording is wrong. An explicit shortcut still avoids
  returning an approximate first-step match.
- Its asserted norm-reduction drift magnitude was not measured. This failure
  concerns an observed hard-spike threshold, not proof of that assertion.
- Torch/autograd examples and suggested production-loader changes do not fit
  the approved scope. No Torch import, new backward path or live loader patch.
- CUDA RNG state is not used by the deterministic local forward here. Existing
  actual RNG streams are preserved; an external budget intentionally cannot be
  rolled back with a model to refund attempts.

Antigravity did not run these tests or approve a learning rule. The result is
**STOP/PIVOT for P0 certification**, with history benefit still untested. The
registered quality/acquisition/retention gates and earlier negative SNN-vs-RAG
results do not change.
