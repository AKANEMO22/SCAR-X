# Full-query SQLite retrieval repair P0 — isolated, before measurements

Repair inputbudget40c5f3: Store.search ignores every lexicalword after16.
Do not change active source while32cd13smallmixed8h is live. Exact-source
patch recipe creates one shadowStore.py; no new service/table/schema/model,
embedding, tokenizer, ALIF equation/weights/encoder, hiddenrule or live config.

Keep requestbound6000Unicodecharacters. For<=16lexicalwords retain exact old
MATCH string, ranking, sourcefilter, timefilter, limit behavior. For longer
requests collect all words, case-insensitive duplicate removal, quote each
literalword and build a balanced OR tree. Every distinct word participates,
including last; no learned/gold-based term choice or prefix/tail-only quota.
Balanced tree limits parse nesting, duplicate removal prevents repeated filler
from creating thousands of identical terms. Longquery BM25ranking may change;
record cost and actual outcomes, do not call retrieval gains neural learning.

Candidate accepts no query longer6000chars or nonstring. No query is executed
if malformed; full query expansion is bounded by requestchars, not an infinite
context claim. Empty/punctuation-only retains empty result. No SQLinterpolation
of usertext: all literalMATCHexpressions passed as SQLparameters. SQLiteunicode61
and regexwordboundaries are unchanged; no claim semantic/paraphrase understanding.

P0 data locked before measurements: two equal-length colorfacts plus1024owned
fillerrecords, no private usermemory. ActualSQLite index and coldbackup for
baseline/candidate identical. Six original positions1/15/16/17/32/129 plus
tail at word~1200 in valid6000charquery. ExtraVietnamese diacritics, punctuation,
literaloperators, all-no-hit, empty and short repeatedword controls. Compare
orderedIDs/ranking on all<=16wordcases and verify stream/before/resultlimit.
Adversarial<=6000char many-unique and repeatedword queries must not parserfail;
report actual latency, SQLtrace MATCHsize, Pythontracemalloc peak and processRSS
separately.<=30s total CPUsetup/SQL, no new neural/GPU/LLM/webcalls.

Technical gate: all required sourcehit/filter/shortcompat/boundary cases pass,
no unexpected exceptions, coldSQLite dataset/indexdigest equal, <=6000charquery
not silently truncated, bounded resultcount<=32, preprocessing peakPythonheap
<=10MB perworstquery and perquerySQLwall<=2s on the owned1028-rowfixture.
These are new engineering acceptance limits declared before execution, not
changes to any research quality/acquisition/retention/5ppgate. No claim matching
compute oldprefix16; disclose extra cost. No repeat/tune after failed output.

Run/source/cases/actualSQLiteoutputs immutable, audit independently reexecute
SQL and inputstructure. No liveactivation before frozen8h terminal+audit and
appropriate source/runtime regression. End-to-end semantics/realcorpus quality
still need their own heldout comparison. This is memory/retrieval repair only;
signedhash128 truncation, hidden representation/credit/retention remain open.
