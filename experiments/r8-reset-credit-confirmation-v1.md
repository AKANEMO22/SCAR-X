# P1 confirmation V1 — approved execution registration

The user explicitly approved **"Duyệt confirmation 12 seed"** in reply to
call_c9e5cfcca5aa4a7cac0e937a8b852134. Authorization is recorded separately in
`r8-confirmation-approval-20261112.json`. The approved proposal at commit7f53824
remains unchanged, including all12seeds,204jobs, data budgets,7Holm tests,
non-inferiority protections,60minute/120second/2GB/CPU1 limits and no P2/deploy.
The old PENDING notice in that proposal records its pre-approval state.

Implementation clarifications (no changed success criteria):

- Freeze candidate-source hashes and source607ffca before opening the bank.
  Technical tests use only prior/synthetic fixtures, not the new seed bank.
- Worker is supervised with a real process handle and hard wall watchdog. The
  global60minute clock starts before preflight; reserve30seconds for finalization
  rather than extending the wall budget. Stop preserves completed/partial cells.
- Sample the supervisor/worker tree plus the already-owned read-only observer,
  deduplicating process IDs. Shared pages may be double-counted conservatively;
  per-process OS commit cap is separate from sampled total-tree observations.
- Validate all48task/seed datasets and split hashes before training any cell.
  Run12seeds x(3primarytasks x5arms + protectedorder x2arms)=204jobs sequentially.
- Write initial/final checkpoints, every loss/applied-norm step, every ID/OOD
  direct/probe prediction, all three weight-revert predictions/scores, hidden
  invariance and raw-feature checks for zero/readout-only. No oracle metadata
  enters neural inference. Refit head stays evaluation-only.
- Statistics: 10000paired-seed bootstrap resamples with RNG919, percentile95%CI;
  exact two-sided sign-flip4096vectors; Holm over exactly the7approved ID tests.
  No samples-as-seeds, missing-data PASS, ceiling exception or after-test tuning.
- Master P1 and the particular XOR reset-superiority claim have separate results.
  Passing P1 requires>=2/3tasks with all approved criteria and protectedorder NI.
  Any result remains research-only; P2/QA/production are outside this approval.
- Re-score predictions/counters/checkpoint hashes independently before final
  gate publication. A missing/failed cell or technical audit blocks PASS.
  No rerun/rescore disguised as a fresh bank; any correction must be disclosed.

Ponytail tools/context were searched for and unavailable. No external teacher,
API, private history/weights, topology growth or whole-network backward is used.
