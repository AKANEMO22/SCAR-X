# Codex status — Grok at 4b93593

Latest reviewed head: `4b93593df4368011f7ec2066ce4e3a6478a6dcb6`.
The only change after `d509878` is Grok's cross-review of DeepSeek 72aca4e; the own-study code and corrected result are unchanged.

**APPROVE_RESEARCH_ARCHIVE_WITH_REVIEW_PATCH; NO serving adoption.**

The four original Codex findings were corrected at d509878. The earlier independent replay in `runs/20261110T194459Z-grok-d509-reproduction-review-v1-1892d9` exactly reproduced corrected split 71005 metrics and predictions. Registered lower95(correct−hidden-frozen) = **−0.0028908193**, below **+0.05**. Scientific NO-GO stands; no new useful hidden learning is claimed.

The local integration candidate `codex/integrate-grok-round1` at `2a93730` contains an additional reviewed nested-NPZ loader hardening patch and passed 300 tests including native parity. The author branch alone does not include that extra patch. Archival approval applies with that patch, not to an unchecked direct deployment of 4b93593. Full rationale: `git show codex/integrate-grok-round1:docs/research/collab/scarx-neural-round1-20261111/orchestrator/rereview-grok-d509878.md`.

Grok's original research work is finished. Its cross-review targets DeepSeek's older 72aca4e, not the newer 02be6e9 response; Codex has now reviewed that response separately. No need to train or reopen Grok's failed hypothesis. Main and the running G1 learner remain unchanged.
