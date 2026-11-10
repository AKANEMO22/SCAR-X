# G1 bounded eight-intent phrasing curriculum, not general language pretraining

Generate80public userprompts in this call,10perintent, tagged8train and2development
perintent. For non-greeting intents, train4Vietnamese+4English and development
1Vietnamese+1English; greeting_vi allVietnamese, greeting_en allEnglish.
Each1–12words, natural ways a real person would address an AI assistant.
The neural core has a small vocabulary; DO NOT generate answers, facts, verified
claims, toolcalls, code, files, shell, webrequests, agents or private data.

Intents:
greeting_vi: greet inVietnamese only, no goodbye/task.
greeting_en: greet inEnglish only, no goodbye/task.
identity: ask theassistant'sname/identity, not whatitcando.
gratitude: thanks/appreciation only, no otherquestion.
farewell: goodbye/endingconversation only, not greeting.
activity: ask presence/activity/wellbeing, not identity/capability.
capabilities: ask availableabilities/services, not name/learningability.
teaching: ask whether/how user canteach/correct/providefeedback toassistant.

Avoid compoundintents/negation/sarcasm/factualevents, capability assertions or
claims thatSCAR-Xalreadyunderstandsphotos/controlsacomputer. Vary wording,
wordorder and ordinary synonyms; not just appendname/punctuation. Development
must vary wordingtemplates fromtrain. Do not copy any forbiddenprompt provided
below, even by changing only terminalpunctuation. Avoid overlong ceremonialtext.
Return schema JSONonly. IDs unique with suppliedbatchprefix, intent and01..10.
Syntheticprompts require semanticreview; do not markverified or assume learned.
