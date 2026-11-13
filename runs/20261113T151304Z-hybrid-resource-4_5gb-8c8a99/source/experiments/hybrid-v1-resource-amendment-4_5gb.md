# Hybrid v1 — user-approved 4.5 GB resource amendment, 2026-11-13

User: “cho phép khoảng chịu lên 4.5GB”. Effective for new service runs:
hard whole-system maximum **4,500,000,000 decimal bytes**; soft background
threshold **4,250,000,000 bytes**, retaining 250 MB for foreground work.
Includes gateway, workers, capture, Hermes process trees and sampled shared GPU.
Windows aggregate private-commit job and sampled RSS guard remain active.

No quantization, pruning, precision reduction, LoRA or ALIF-learning-rule change.
Context remains 16,384 tokens. A larger RAM allowance does not itself validate
a longer working window, better reasoning, hidden learning or retention.

The completed context run `20261113T144438Z-hybrid-context-v1-a7312f` used a
4,000,000,000-byte limit and <=3.85 GB deployment-headroom condition. Preserve
its manifest, source bundle, answers, timeouts, original receipts and NO-INCREASE
decision. Do not relabel it PASS under the new cap. Its measured peaks provide
only a resource estimate for future tests: they do not prove 4.5 GB video/audio
or long-term qualification. No new benchmark or training is launched here.

Earlier 3 GB/4 GB receipts and scientific protocols remain historical. Quality,
hidden-synapse, forgetting, seed/split and online-comparison thresholds do not
change. This amendment supersedes the operational 4 GB default only.
