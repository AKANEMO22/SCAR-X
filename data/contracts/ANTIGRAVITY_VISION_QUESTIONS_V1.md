# Public synthetic visual-question wording only

Generate32 short natural questions, exactly8perkind and4Vietnamese/4English
perkind. Do not generate answers, images, facts, scene descriptions or tools.
No files, shell, web, agents, private data or external actions. SchemaJSONonly.

Kinds:
* single_color: ask the color of the sole visible shape in an image.
* single_shape: ask what geometric shape is the sole visible object.
* object_count: ask how many geometric objects/shapes are visible.
* left_color: ask the color of the leftmost object among multiple visible shapes.

Each row: id=kind_01..08, kind, language=vi/en, question. Keep question1–16words,
≤180characters. One attribute perquestion. No color names, specific shape names,
numbers, answer hints, comparisons, negation, tool claims or implied modelability.
Do not combine intents. Refer to this image generically; no instructions toclick
or access files. A deterministic renderer will supply synthetic geometric images
and source-of-truth attributes separately; generated text will be reviewed.
No verified flag, no claim the neural model has learned or sees arbitrary photos.
