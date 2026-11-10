# Generation1 decoder audit — development only, 2026-11-09

Frozen release selected in generation1-joint-bd6c92. Compare EOS-only decoding
against stopping after the first neural-generated sentence punctuation.
Run on training/development prompts only; do not reopen final acceptance data.
No weight updates, no model selection from test, no answer matching/reranking.
Report exact development score and outputs; this is decoder behavior, not
representation or learning improvement. Engineering regression prompts hi/chào/
chào bạn remain training examples, not generalization evidence.
