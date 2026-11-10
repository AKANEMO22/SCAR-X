# Antigravity dialogue-event curriculum v1

2026-11-09. Small corpus repair for the ordered input prototype under masterP1,
not a release qualification or large language pretraining. Existing supplement
had all-Vietnamese train/all-English development, so language/split confounded.
Do not rewrite it; generate a new version and preserve both.

Exactly40 dialogues,3user/assistant exchanges each. Train32 =20vi+12en;
development8 =4vi+4en. Each example has id,language,split,scenario_id,template_id,
turns. Roles alternate user/assistant; every assistant response1–12wordtokens,
no unsafe/current factual claims or private user data. Total target vocabulary
<=200tokens (SCAR-X tokenizer). No finaltest; original test prompts unavailable
to the generating CLI. Do not claim generated labels independently verified.

Scenarios: ordinary greetings/conversation, short instruction following,
clarification, explicitly stated preferences, correction of a previous answer,
and acknowledging uncertainty. Keep the conversation natural, useful and short.
Specific language templates must not cross train/development; scenario families
may occur in both. A dialogue is indivisible: never split its turns. Do not
flatten memory-dependent responses to isolated last-user strings.

Teacher may create text only; no files/tools/web/shell/subagents/private data.
Record CLI/model/version/usage/hash. Validator checks all counts,roles,token
limits,language balance,template/duplicate overlap and source metadata. Human
style/rubric review remains separate from machine arithmetic proof.
No live auto-import, neural weights or API keys included.
