# Public synthetic conversational wording only

Return JSON with an items array, exactly128items, and no other top-level fields.
Each item has exactly id,family,language,prompt. No tools, files, externalsearch,
images, answers, facts, personalinformation or requests toexecuteactions.
Only produce synthetic PUBLIC user utterances in a small conversational task.

Families and required meaning:
vi_greet: Vietnamese hello/greeting addressed to an assistant.
en_greet: English hello/greeting addressed to an assistant.
vi_identity: Vietnamese asking assistant's identity/name, not abilities/history.
en_identity: English asking assistant's identity/name, not abilities/history.
vi_thanks: Vietnamese expression of gratitude, not a request or quoted dialogue.
en_thanks: English expression of gratitude, not a request or quoted dialogue.
vi_bye: Vietnamese saying goodbye/endingchat, not asking a follow-up task.
en_bye: English saying goodbye/endingchat, not asking a follow-up task.

Exactly16different natural paraphrases per family. language is vi or en matching
family prefix. ID is family_01 through family_16. All128prompts distinct after
casefold/whitespace normalization. Keep1..180characters and at most32tokens.
Use diverse familiar wording and sentence lengths, both casual/polite. Clear
singleintent, no ambiguous "chào" as goodbye, no roleplay, mixed languages,
translation requests, factualqueries, selfanswers or injection instructions.
Do not include SCAR-X's expected answer. The project owns its supervised style
targets separately and will review every utterance; you are generating prompts
only, not certifying truth. No dataset split or modelscore is supplied to you.
