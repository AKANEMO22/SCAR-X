# Learning-channel engineering acceptance

2026-11-09. Dedicated Hermes plugin route/sidebar entry; same SCAR-X model.
No neural rule change or research quality claim. Check real registered backend
plugin routes with normal session-token auth against an isolated live SCAR-X
HTTP server. Verify ordinary messages are not labels, numeric false feedback is
rejected, correct feedback persists, retries deduplicate, and unknown main-chat
questions are routed as a notice instead of an inline blocking clarify.

Frontend behavior tests: answer preserves source session/question identity;
network failure retains draft and reuses client_id. Native UI must be inspected
separately; API/Vitest success is not a claim of native visual verification.
Bound history512, keep all open questions visible, source-label controller vs
neural output. Conversation log lives in SQLite separately from neural snapshots;
rollback retains transcript while displayed lesson state follows restored core.
Test RAM/save/restart; no user-private inputs in test fixtures.
