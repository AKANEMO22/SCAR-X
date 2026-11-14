# Latest-information questions incorrectly searched local files

Reported query: `jojo ra tập mới nhất là tập bn`, then `tìm kiếm đi`.
The actual RC9 logs show all eight tools supplied, including `web_search`.
The first answer refused a capability it had; the follow-up called
`search_files` with `go|journey|travel`. Neither request used web search.
The stream field was null, so private-media tool suppression was not the cause.
A direct native SearXNG request returned ten real results in about1.2s, with no
unresponsive engines. This was model tool selection, not a missing search service.

## Repair

`web_routing.py` recognizes bounded public freshness/release queries in Vi/En
and bare follow-ups. It takes a query only from actual user turns, never model
refusals, tool results or retrieved memory. It does not jump over a newer local
topic to reuse an older public one. Explicit no-search/tool-choice constraints,
private-media context, local paths and conservative private-content exclusions
prevent automatic routing in those cases. No user message is rewritten.

For this public search turn the controller requests the real existing Hermes
`web_search` with the user's topic. The native model sees the search tool for
this turn, not unrelated file tools; normal tools return on the next workspace
task. SearXNG is search-only. The independently configured `web_extract` backend
is Exa, so this automatic search route does not escalate into that service.
Explicit page-reading turns retain the normal Hermes integration; full-source
reading is not qualified by a SERP-only acceptance.

The first request is marked `runtime_public_web_prelude`, with zero generated
model tokens and a `web_search_prelude` runtime event. Hermes still executes the
tool normally with its permissions and loop guard. After three already recorded
search calls the next native request gets tool_choice none; this is a next-call
boundary, not a claim that a parallel batch cannot exceed three calls. Results
and final wording are real, not hardcoded JoJo facts or a fabricated search card.
The actual native tool names are logged separately from the host's original
schema names.

## Validation and limits

323 full release Python checks passed. A final narrower negation refinement
(`không biết ... tìm giúp tôi` is not `không tìm`) has102 targeted gateway/
routing/package checks. Coverage includes follow-up carryover, assistant/tool
injection exclusion, newer local topics, declined search, unavailable tools,
private media and new-module install/rollback safety. No renderer, model weights,
ALIF equations or learning rules changed.

Actual canonical Hermes acceptance is recorded separately after activation,
using a neutral owned hidden session and the exact reported question/follow-up.
Read [CURRENT_RELEASE](CURRENT_RELEASE.md) for installed version and receipts.
Code-level checks alone are not presented as an end-to-end or model-quality PASS.

Remaining limits: intent rules are conservative, not a general learned router;
ambiguous “tập” can refer to anime or manga. A search result does not verify the
latest episode count, and model0.8B can still misread it. This is a runtime fix,
not evidence that SNN learned web use, root causes or retention. Original P0
failure, all research thresholds and the cancelled eight-hour test remain intact.
