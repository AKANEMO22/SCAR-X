# Public wording contract — SCAR-X development candidates

Return the required JSON object with exactly128 items. Generate wording only,
no answers, no files, browsing, tools or commands except native structured finish.
No private data is needed. Do not inspect your environment. This is development
data preparation, not a claim that the model understands language or images.

Each item has exactly id, modality, family, language, prompt.

CHAT:96items: families greet,identity,thanks,bye,learning,capability.
For each family produce8Vietnamese and8English prompts; IDs
chat_<family>_<vi|en>_<01..08>. modality="chat".

Intent definitions:
- greet: user naturally greets assistant, expecting a greeting.
- identity: asks the assistant's own name/identity, not the user's identity.
- thanks: user thanks assistant for help already given, expects acknowledgment.
- bye: user ends conversation and says goodbye.
- learning: asks whether/how user may teach or correct assistant.
- capability: asks assistant what it currently does; no assumed unsupported skills.

Keep each prompt single-intent, natural and short (3–24words, <=180characters).
Avoid exact common prompts "hi", "hello", "chào", "chào bạn", "bạn là ai",
"bạn tên gì", "cảm ơn", "tạm biệt", "bạn khỏe không", "bạn làm được gì".
Do not put answers into prompts. Avoid 8copies differing only by punctuation,
names or please. Some can use ordinary informal phrasing but no ambiguous slang.
Do not ask to greet/thank a third party or translate a quoted phrase.

VISION:32items: families single_color,single_shape,object_count,left_color.
For each family produce4Vietnamese and4English questions; IDs
vision_<family>_<vi|en>_<01..04>. modality="vision".
Questions will later accompany project-owned synthetic scenes, not private photos.
- single_color: scene contains one geometric object; ask its color.
- single_shape: scene contains one object; ask its geometric shape.
- object_count: ask the number of objects, not number of colors or shapes.
- left_color: scene contains2objects at distinct left/right positions; ask color
  of the object farther left. No hidden numeric/spatial assumption beyond this.
Do not include a color/shape/count answer or assume an exact shape/color in text.
Natural questions, 3–24words and <=180characters. Do not use yes/no alternatives.

Use valid Unicode Vietnamese diacritics. JSON only. No markdown or explanations.
