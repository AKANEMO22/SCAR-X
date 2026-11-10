---
name: scarx-product
description: SCAR-X local product and Windows release. Use for the localhost UI, honest vision/chat limits, start/stop scripts, and smoke tests. Does not fake memory, vision, or reasoning the backend lacks.
---

You work in `C:\Users\hachimi\Downloads\model agent\scar-x` on UI static files, `README.md` user instructions, `Start-SCARX.ps1`, `Stop-SCARX.ps1`, and `scripts/smoke_release.py` only.

Never write `local/generation1`. Never change neural code, learning rules, or checkpoints. Do not commit.

The UI must say digits-only vision and experimental word generation. Do not add a claim the API does not return. Prefer a smoke test on a temporary data directory and a port other than 8765.
