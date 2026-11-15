# D1/D2 provenance-only recovery, before any neural/probe execution

Original run `20261115T192329Z-rc-d12-v1-c22e39` ABORTED at source-byte guard,
0 neural forward, 0 probe fits, 0 hidden updates. FAILED.json and local stdout/
stderr are preserved. Root source commit c22e395d remains a failed intake,
not a result to replace. No parameters/data/seed/thresholds were observed.

Cause verified read-only: src/scarx/neural/core.py and hybrid/cuda_alif.py
working copies have CRLF because Git checkout honored core.autocrlf; Git blobs
have LF. Their bytes are exactly equal after CRLF→LF and no production files
are edited. Other four registered sources already match byte-for-byte.

New unique run may execute the original diagnostic without changed numerical
code/data/gates/budgets. Source guard records BOTH actual working bytes SHA256
and committed blob SHA256, requires canonical LF bytes equal, and saves copies
of actual executed source. This is provenance recovery, not model rescue or a
new learner. Total300s budget includes failed intake's<1s reservation (reserve2s).
No old run writes, P1b hidden learning/TEST access or live model restart.
