# Pipeline harness v2 — request isolation correction

The first v1 run and repair run were served at the usual loopback port so the
user could watch actual trace. Existing Hermes traffic could also enter that
port. In repair run d468f9, 12 external streamed requests and 11 tool requests
arrived during the idle stage. This preempted review and changed the isolated
controller's last-presence time. Record that stage as CONTAMINATED / INCONCLUSIVE,
not a scheduler PASS or FAIL. Keep fixed fixture results and original receipts;
do not label the run a clean end-to-end qualification.

V2 wraps only the engineering server's HTTP handler. ALL POST mutations require
an ephemeral per-process audit nonce, including presence, upload and inference.
Other callers receive503 maintenance, without model execution or state change.
GET trace remains available. Normal product authorization is unchanged; no
runtime/learner/mock output is substituted. Prepare a fresh run, preserve all
old source bundles and locked labels. Test the filter on an actual local HTTP
server without invoking a model. No third full inference repeat is launched
solely to make the existing report look clean.

V1 event batches are retained in ignored local storage because they may contain
unrelated private conversation/tool data. Public run receipts contain only fixed
fixture cases, counts, hashes and explicit contamination notes. Their exact raw
hashes/paths are indexed. This is privacy separation, not removal of a failure.

The Hermes-owner probe had a separate harness error: display_projection rows
carry `text`; the first parser read only `content`, reporting a false timeout.
Readonly reinspection of the same turn confirmed PIPELINE_OK. Preserve the
false-timeout receipt plus appended correction; no new inference/retry/latency
claim. The probe now accepts both documented fields.
