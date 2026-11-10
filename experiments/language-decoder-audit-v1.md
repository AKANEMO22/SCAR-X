# Language decoder audit v1

Registered after V2language exact18.75% failure, before this decoder comparison. Keep trained checkpoint V2 selected solely on development, no weight updates. Hypothesis: teacher-forced loss vs free-generation gap partially comes from greedy decoding/exposure errors. Beam search does not create new learned representation or remove data/capacity limits.

One fixed decoder width4,lengthpenalty0.6,max32words; generic neural probabilities only. No answerbank/allowed-sentence list/intent template or corpus-guided reranking. Compare greedy vs beam on same16developmentprompts; do not tune another beam parameter. Open fresh16testparaphrases only if developmentexact≥.75 and no loops; same testexact≥.5 and knownrequestedgreeting criterion before limited production candidate. Otherwise log audit fail without applying old test again or replacing production.

Budget1CPUthread,2GBprocesslimit,60s. Record every output. Check restored state, deterministic decoding and no weight update. Changing decoder does not prove successful training by itself; quantify against the same frozen checkpoint. Native screenshot understanding and general chat remain unachieved regardless of eight-family corpus result.
