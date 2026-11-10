# Input events v2 — hash collision repair, registered before fresh audit

2026-11-09 after v1. Preserve v1 raw/source:2of128pairs still collided because
the SINGLE signed hash bucket can map two distinct numeric tokens to exactly
the same vector. That is a concrete representational collision, not failed SNN
learning. Do not overwrite the v1 receipt/report.

Replace one-bucket token identity by128signed digest bits (Blake2b16bytes), each
scaled1/sqrt128, plus the same0.25weight character-trigram digests; normalize.
Dimensions/memory/event limits stay unchanged. Still not mathematically injective
for arbitrary strings; hash collisions remain possible in principle. No learned
semantic embedding, pretrained model or label-specific exception.

Validate same diagnostic pairs as regression (already viewed, not heldout),
and NEW independent seed29107,256pairs for confirmation. Keep text corpus/order
task generator fixed, no reranking/tuning by these outputs. Test same red/green
image fixture separately; coarse-RGB information is not vision understanding.
Budget<=60s/2GB, no updates or deployment. One repair only for this information
compiler audit; future learning must use independently registered task data.
