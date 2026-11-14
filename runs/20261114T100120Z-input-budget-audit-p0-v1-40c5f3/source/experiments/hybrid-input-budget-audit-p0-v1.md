# Input boundary audit — before execution, no neural calls

Purpose: inspect whether current relevance/incident encoders and FTS search
retain discriminating observations for valid long user inputs. No new encoder,
learner/rule/weight/model changes, no GPU allocation/forward/update, VLMcall,
webrequest or private user data. Current smallmixed8h36frozenfiles untouched.

Read actual LocalALIF.relevance via an explicit input-only forward tap that
captures its argument and returns an unreported transport placeholder. No tap
value is a model score/prediction or training label. Recompute captured array
using actualfeatures as a consistency check; keep full input arrays/hashes.

Relevance pairs: same query and same length factual evidence "lamp color blue"
versus "lamp color pink". Query asks to select blue, padded deterministically
to token counts16/64/120/123/124/125/126/127/128/160 using "note" words. Labels
come from explicit documentcolor equality, not model output. Compare exact
array equality within each pair. Short controls must be reported regardless
of whether signed hashing happens to alias. Do not select a convenient subset.
An opposing-label identical-input bound is input distinguishability only for
these fixed-state paired classifications, not neural or full assistant accuracy.

Incident researchV2: identical actual flags/probe and question, change only
error_type missing_file/permission_denied. Query lengths16/64/120/128/160, each
within1000char. Check whether appended lexicalerror survives; labels/absolute
time/IDs are not inputs. This is not proof exact root cause can be inferred
from reported error alone. No production encoder promotion.

FTS: actual Store/SQLite in one owned temporary directory; evidence blue/pink.
Place query keywordblue at positions1/15/16/17/32/129, all preceding wordsnote
absent from evidence. Record actual returnedIDs/rows and ordered query terms;
no repaired query or response generation. Separate FTS first16 cap from SNN
first128 cap and VLM16Ktoken working buffer. Byte/word/token units explicit.

Lock complete source and cases in unique run before any measurement.<=30s
CPU preprocessing/SQLite only, no new neural quality metric/gate. Compare
historical032f30 predicate text lengths descriptively; do not claim this
long-input counterexample explains an old short-input failure. Data-only audit
rederives arrays/equalities/SQLite hits from lockedcases and sourcearchive;
publish measuredcounts on localhost as technicalaudit, never as thought or
modelanswers. No main/live/R13/history changes. Preserve failures immutable.
