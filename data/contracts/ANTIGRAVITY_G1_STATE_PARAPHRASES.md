# Public G1 conversational-state diagnostic paraphrases

Generate48 NEW short natural user utterances for the eight known G1 conversational
intents below, exactly6perintent.3Vietnamese+3English perintent; do not copy existing
phrases from memory or this brief. Keep each1–12words. Do NOT generate assistant
answers: targets are fixed project style references, not factual labels supplied
by a teacher. No private data, tools, files, shell, web, agents or external action.

Intents, with intended meaning:
- greeting_vi: a Vietnamese greeting only (all6Vietnamese for thisintent).
- greeting_en: an English greeting only (all6English for thisintent).
- identity: ask the assistant's name/identity, not ability or health.
- gratitude: thank the assistant; do not ask a new task.
- farewell: say goodbye, not hello or thankyou.
- activity: ask if it is active/present or how it is, not identity/ability.
- capabilities: ask what it can do, not its name or whether it learns.
- teaching: ask whether/how the user can explicitly teach/correct it.

Use neutral language and vary syntax. Avoid compound/multiple intents, negation,
irony, imagined factual events, fake tool claims and long roleplay. In particular,
do not imply it already understands arbitrary photos or operates a computer.
Each item has id, intent, language, prompt only. Assign IDs perintent01..06.
Return schema JSON only. Synthetic utterances remain unverified until review;
never include a verified flag or claim the neural model has learned these cases.
