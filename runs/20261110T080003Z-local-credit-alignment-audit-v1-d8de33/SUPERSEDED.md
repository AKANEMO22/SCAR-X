# Superseded run - do not cite

This run measured NLL deltas correctly but snapshotted the applied-weight norm AFTER the update, so every `applied_norm_*` read 0.0 and norm-matched controls could not be verified. Superseded by `20261110T080426Z-local-credit-alignment-audit-v1-db01fc`, which fixes the snapshot order and adds the permuted control.

Kept for history per project rules; results must not be cited.
