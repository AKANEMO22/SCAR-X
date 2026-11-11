# Lessons learned — antigravity — campaign `scarx-neural-round1-20261111`

Models: Gemini 4 Argon (High) for prototype/protocol/run; Claude Sonnet 5.5 (High) for cross-review and the final analysis. Environment: Antigravity.

## 1. What was wrong or weaker than I first said

1. **I called the development screen "3/3 pass" and it did not hold.** `order` passed on 3 seeds and then lost to frozen by 18.5 pp on 8 seeds. Three-seed bootstrap intervals have almost no resolution; a screen can open confirmation, but it is not evidence for the claim.
2. **The executed run did not match the registered protocol** (different tasks, seeds, sizes, P2 stream, gates, no same-run baseline). I wrote the runner by reusing Report 35's harness and never reconciled it with `protocol-v1.md`, and I discovered it only while writing the report. Fix: a post-hoc Amendment 1 that leaves results untouched and downgrades the claim. Lesson: diff the runner's constants against the protocol before the first formal trial, ideally with an automated check.
3. **The mechanism is shaped to the benchmark.** The zero-diagonal coincidence and the query-gated product were designed knowing `strict_binding`'s key×value×query structure. On `deepseek`'s `pair_binding` the same locked mechanism scored 50.5% while its own ablation (`no_gate`) scored 78.4%. A positive result on the design target is weak evidence of a general mechanism.
4. **"Frozen" is not a floor.** Frozen DC-ALIF already scored 75% on `strict_binding`. The architecture contributes before any learning, so the honest credit-assignment number is correct − frozen, not the raw accuracy.
5. **A 10-point part of the gain survives shuffled credit**, so I must not describe the whole gain as proof the signal reaches the right neurons.

## 2. What held up

- The root-cause argument for why a 1-compartment linear voltage cannot do the order-3 conjunction, and the matching `no_gate` ≈ baseline (62.4%).
- Exactness engineering: finite-difference audit over 64 synapses per task, cold-process resume of 198 trials, rollback, loader isolation.
- Control discipline: zero == frozen bit-for-bit, shuffled, `no_gate`, an isolated BPTT control, and a report that keeps the negative results next to the positive one.

## 3. What the peers got right that I would adopt

- `grok`: measure information in the untrained state **before** training (fast 0.484 vs slow trace 0.9375). I did not do this for DC-ALIF. A pre-training probe of `d_j` would have shown whether the dendritic trace holds the product before any learning, and would have flagged the `pair_binding` and `order` failures earlier.
- `deepseek`: run the dumbest control first, and treat any task you introduce as unvalidated until the weakest readout fails on it. Its generator-defect story (bag shortcut) is the reason I added a bag baseline to every task.

## 4. Where I disagree

- With `deepseek`'s "the task needs no product term": uniform spike sums are not order-1 in the input (the raw tape reaches only 0.53). The product basis still adds +23.6 pp at matched width.
- With reading `grok`'s slow-trace gain over the fast feature as learning: it is a readout-memory change; the hidden update is not required (zero == frozen).

## 5. What can be combined, honestly

Slow readout memory (`grok`), a product/coincidence path (mine, `deepseek`'s §9 proposal) and a pre-training information probe are three distinct pieces. A fair combined test needs a **task-agnostic** gate: a single locked configuration scored on `order`, `xor`, `strict_binding` and `pair_binding` together, with the best single baseline on each. Current evidence says each mechanism helps on some tasks and hurts on others; no configuration has been shown to be uniformly better.

## 6. Process notes

- Memory cap: four parallel PyTorch BPTT children exceeded the 2 GB process-tree commit; fixed by running NumPy jobs first and BPTT children one at a time. Interrupted run is preserved and marked.
- Windows tooling: no `gh` on this machine, so a Draft PR could not be created from here; the branch is pushed and the PR must be opened by someone with access (see handoff).
- Peer worktrees were only read, with bytecode disabled and `-p no:cacheprovider`.
