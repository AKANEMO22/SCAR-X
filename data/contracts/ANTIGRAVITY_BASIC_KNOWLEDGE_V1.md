# SCAR-X public source-backed candidate factory

Generate only the requested JSON. Use no tools or external sources except the
native structured finish. Do not inspect files, environment or private history.
The appended facts are from a fixed PUBLIC source snapshot, with AST-verified
values. You generate questions and CANDIDATE answers; an independent verifier
checks them. Do not assign split, verificationstatus or new facts yourself.

For each supplied fact produce exactly8variants:
1,3,5,7 Vietnamese (vi);2,4,6,8 English(en). Each item has exactly:
fact_id,variant,language,prompt,canonical_answer.

Known domain facts (known_domain=true):
- Ask for the indicated value in the stated scope, never a different field.
- Write the exact field name in ONE pair of backticks. No other backticks.
- Config questions must explicitly say Config and default/mặc định.
- UnifiedCore questions must explicitly say UnifiedCore and initial/constructor/
  khởi tạo/thiết lập/setup, so current runtime values are not confused with defaults.
- For maximum=true, explicitly say maximum/limit/cap/tối đa/giới hạn/trần.
- Ask an open value question, never yes/no, negation, comparisons or explanation
  of another field. No number, guessed value or answer may appear in the prompt.
- Avoid written-out numbers too (zero/one/two, một/hai/ba/không, etc.), including
  narrative filler. Do not mention ALIF when asking what the neuron enum is.
- Canonical answer is JUST the source value plus terminal period: e.g. a numeric
  literal or an enum. Preserve decimals exactly; no added units, grouping, prose
  or invented claims. Source string alif can be written ALIF.

Outside-domain API facts (known_domain=false):
- Ask a natural question about that API topic and explicitly mention API.
- Do not put the actual source value, route, digits or parameter answers in text.
- Our learning domain intentionally excludes these facts. Candidate answer must
  be exactly "Tôi chưa biết." for vi and "I do not know." for en. This expresses
  missing training-domain coverage, not that the fact is false in the world.

All prompts:12–220characters, real Unicode Vietnamese diacritics, no control chars.
Variants1/2: direct question.3/4: concise request to state the value.5/6: question
from someone reading the configuration.7/8: distinct contextual question from
someone preparing/explaining the initial setup. Vary the syntax substantially;
not just punctuation or a few politeness words. No private information is needed.

Return exactly8records per fact, no missing or duplicated variant. JSON only.
