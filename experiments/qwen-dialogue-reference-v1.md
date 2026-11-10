# Qwen dialogue reference v1 — exploratory qualitative comparison

2026-11-09 before reference generation. Event-dialogue model finished its locked
trial and failed exact utterance improvement, no deployment. Compare24development
prefixes to pinned Qwen3-4B-Instruct-2507 Q4_K_M from the earlier numeric reference.
Same public prefix roles/text, no new system/task instruction, no tools/retrieval.
CPU4threads,context1024,max64tokens,temperature0,seed919,<=180s,4.5GBobserved cap.
No SCAR-X training job concurrently. Qwen process must stop afterwards, never
become a Hermes/SCAR-X proxy. Memory/training budgets differ and must be reported.

Exact match to one synthetic response is descriptive only: other natural
answers can be correct. Save both outputs for every prefix. Review criteria,
locked before Qwen outputs: coherent with prefix, follows explicit instruction,
no unsupported action/capability/knowledge claim, correct language, no excessive
repetition. Source-known Codex review is exploratory, NOT independent/blinded
human evaluation. No global normal-chat or4B-equivalence declaration.
SCAR-X unseen words outside its train-vocabulary are representability limits;
report which target utterances are expressible, never expand vocabulary from
development. No model selection or weight replacement from this comparison.
