# Dated public questions and short followups: RC15 → RC16

Reported screenshot: `việt nam có bao nhiêu tỉnh thành 2026`, then `là`.
Read-only inspection of the actual `Greeting #13` transcript found0toolcalls.
The model answered34plus5cities, then refused web access it actually had.

The existing route covered latest/current/release wording but not a year in a
public factual question. Without the runtime route, the small frozen VLM chose
to answer from weights and invented a capability refusal. This incident does
not establish that SearXNG is unavailable or that the SNN learned tool selection.

## Exact repairs

RC15 recognizes dated public factual questions with bounded public subjects,
while keeping math, creative tasks, personal inputs, local files/workspaces,
explicit no-search and explicit tool choices outside automatic export. Search
query comes from actual user text; no fact answer is hardcoded. Only routing
module changes in its58-file code package.122targeted checks pass.

Actual canonical Hermes/SearXNG first question succeeded: one search with the
exact user query, five public results, native answer `Việt Nam có **34** tỉnh thành.`
However its short followup generated an unrelated YouTube-like query three
times before the existing no-progress guard stopped it. Preserve this failed
followup receipt; a successful first-turn search was not full acceptance.

RC16 resolves bounded clarification phrases (`là`, `cụ thể`, `which ones`, etc.)
to the preceding eligible public question. It cannot skip a newer substantive
local/private task or accept a clarification alone as a new public topic.
For this short clarification route, one real search is followed by tool_choice
none on the next native call; it does not give model-created unrelated queries
more retries. Other public requests retain the existing three-call next-request
boundary. Original user messages are unchanged. No fake tool result or answer.

132targeted gateway/routing/evidence/package checks pass, including six new
clarification repros that failed before repair. No renderer/precision/model
weights/ALIF plasticity or learning criteria changed.

## Actual validation and limits

[Canonical receipt audit](evidence/ACTUAL_HERMES_DATED_WEB_20261115.json) preserves
both RC15's failure and RC16's result. Each RC16 turn called exactly one actual
`web_search` with the original public question, received five public results,
and completed with the native34answer. No local-file or paid extractor calls,
unrelated YouTube query or repeated tool-loop in those two turns. The followup
repeats the count: richer explanation/listing or proper source citation remains
unproven. These are two engineering acceptance turns, not a general tool-use
benchmark, neural learning or evidence every followup is understood correctly.

Live restart used a maintenance lease and verified only owned gateway processes.
All saved neural arrays, learned metadata, RNG and replay remain identical;
liveupdates5/hiddenupdates5, VLM/vision+ASR+ALIF stayGPU. Rollback bytes come from
the exact pinned predecessor archive because this worktree is also the already
edited serving source. Do not misdescribe restart as overwriting an old on-disk
file. [RC16 activation](evidence/RC16_ACTIVATION_20261115.json) and
[package-to-install verification](evidence/RC16_INSTALLED_ARTIFACT_20261115.json).

Current34total was checked independently against public government sources.
The2025breakdown28provinces/6cities is dated: Đồng Nai became a centrally governed
city effective30April2026 under [Resolution30/2026/QH16](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/4/nq30_nqthanhlaptpdongnai.pdf).
The [Hanoi planning authority's current account](https://vqh.hanoi.gov.vn/index.php?language=vi&nv=news&op=tin-lien-ket%2Fxu-huong-phat-trien-do-thi-thuc-hien-mo-hinh-chinh-quyen-dia-phuong-2-cap-3342.html)
lists27provinces and7cities, total34. These sources were not inserted as training
labels or expected native answers. Real SearXNG snippets included both older
and current breakdowns; search alone does not solve time-aware fact validation.

## GitHub scope

The108completed research commits through6e0f929b were pushed to
`codex/scarx-hybrid-20261113`; main stayed37749041. Subsequent RC15/16 bugfix
commits/evidence continue on the same branch, never silently merge into main.
Unrelated untracked video-metadata intake files remain local and untouched.
