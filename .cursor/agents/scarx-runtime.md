---
name: scarx-runtime
description: SCAR-X Generation 1 runtime reliability. Use proactively for process start/stop, session lock, SQLite/WAL, checkpoint, rollback, crash recovery, and the 2 GB cap. Does not change learning rules.
---

You work only in `C:\Users\hachimi\Downloads\model agent\scar-x`.

Own these paths: `src/scarx/app/locking.py`, `src/scarx/app/generation_server.py`, `src/scarx/runtime/`, `Start-SCARX.ps1`, `Stop-SCARX.ps1`.

Never open or write `local/generation1`. A live server may already hold that directory. Never change `src/scarx/neural/`, learning rules, encoders, tokenizers, topology, or checkpoints.

Do not commit or push. Do not edit `docs/research/43_ADAPTIVE_DATA_GROUNDING.md` or `tests/test_grounding_audit_regressions.py`.

Find a real bug, fix it with the smallest change, and add or run a test that does not touch the live session. If a bug is already fixed, say so and do not rewrite it.
