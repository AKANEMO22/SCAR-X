# RC7 → RC8 upgrade source custody

Two concrete maintenance problems were identified during audio-repair release
preparation. Historical artifacts remain unchanged.

1. `protected_run_gate` required live files to match the cancelled endurance
   baseline forever. RC7 already contains authorized changes, so a subsequent
   real update could be rejected after a valid cancellation.
2. The packager took `target_before_sha256` from the already edited candidate
   checkout. A previous release would fail source CAS instead of being eligible
   for the advertised update. Matching the candidate to itself did not exercise
   this upgrade path.

The current allowlist pins the RC7 archive's full SHA256, version and source
commit. The packager verifies all previous archive members against its manifest
and derives before-hashes from that exact payload. A separately trusted source
checker verifies the upgrade metadata as well as executable contents. No Git
fallback, guessed SHA, private config or downloaded executable is substituted.

Before mutation, `plan_files` accepts only the pinned predecessor bytes or files
already matching the next payload. An unrelated edit is rejected. The terminal
run's source gate permits historical, pinned predecessor or exact next payload
bytes only for source paths included in this code package. Other research/source
files keep their historical pin. Actual PID/create-time liveness is checked
first; cancellation request/audit/result hashes and the not-PASS decision are
still verified. This exception cannot turn a cancelled run into an eight-hour
success, change its logs or permit updating a running protected worker.

Safety coverage includes corrupt/mismatched predecessor, archive path escape,
old/new source acceptance, rejected user edits, tampered cancellation custody,
live-worker blocking, source CAS, interrupted writes and code rollback retaining
new user data. **292 Python checks passed**, including the audio/CPU/native/CUDA
suite. `scripts/accept_hybrid_rc_upgrade.py` separately reconstructs the actual
RC7 payload on an owned clone, reproduces the old refusal and checks all three
changed payload paths through source gate, CAS and rollback. It never mutates
live source or calls a model; actual receipts are published after execution.

Antigravity CLI was also asked for a narrow readonly review using
`gemini-3.8-flash-high`, effort `high`. Its process terminated with print timeout
after 117.9 seconds and no final review. Logs show attempts to read nonexistent
guessed filenames; no review approval or execution of our tests is attributed
to it. The sanitized four-file snapshot hash is unchanged. No retry loop or
permission bypass was used. These engineering changes have independent fixture
coverage and explicit limits; they do not prove neural intelligence or production.
