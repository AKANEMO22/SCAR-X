# Hermes stateful-chat engineering acceptance — 2026-11-09

Adapt Hermes to the stateful backend. Do not modify SCAR-X neural code, weights,
learning rules, RAM limit or declare a64K learned context. Keep4096 as a host
transport buffer, input limit2048characters as actually implemented.

Explicit named-provider mode must bind endpoint+model. Normal LLM agents retain
the64K rule. Stateful chat has no agent tools, injected coding/system prompt,
history replay, fallback model, MoA or auxiliary compression. Transcript remains
append-only UI history, while backend neural state stays on the SCAR-X server.

Test real AIAgent imports, init, two calls to the real local backend, public
messages hi/chào bạn, append-only SessionDB, stream/nonstream as supported.
Record observed API payload, selected endpoint, zero tools, real4096 host budget
and token usage availability. No model capability/scientific success claim.

Back up exact modified Hermes source; preserve unrelated dirty working tree.
Restart only owned idle Hermes backend when needed, then verify native model
picker and session request path. Do not kill active user work or start9B.
