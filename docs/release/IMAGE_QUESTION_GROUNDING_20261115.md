# R Shiny image question: RC17 failure and scoped RC18 recovery

**Exact reported case repaired in canonical Hermes; general vision/coding
quality remains unqualified.** No newSNNlearning, VLMtraining, adapter, scale,
precision change or external upload of userpixels. All actual inferenceCUDA.

The screenshot asked what server binding is needed for `plotOutput("distPlot")`.
The original `Greeting #14` answer transcribed the question/options but called
the functions PythonPlotly and rejected the correct alternative. Hermes logged
native image attachment and gateway prepared a large image-bearing user message:
the attachment was not absent. One later directGPUimage probe picked C but
invented incorrect `ui$...` server code; literal text-only transcription also
gave confused framework explanations. Perception and factualreasoning are
distinct. Do not call a readable question evidence the solution is understood.

## Source-grounded assistance, not learned capability

`image_question.py` reads current attached pixels locally with the same frozen
GPUVLM into bounded `visible_text` JSON. It never follows imagepaths/URLs or
instructions in image text. Original user pixels/text and hosthistory remain
unchanged. Invalidtranscription keeps the original request path; no guessed
option substitutes for a malformed result. Explicit tool requests, editing/
execution tasks, structuredoutput, video and ordinaryphotos retain their routes.

One explicit referencecard matches the callable symbols `plotOutput` and
`renderPlot`, with manually sourced principles from official Posit docs:

- [plotOutput](https://shiny.posit.co/r/reference/shiny/latest/plotoutput.html)
  declares the UIoutput location/identifier.
- [renderPlot](https://shiny.posit.co/r/reference/shiny/latest/renderplot.html)
  provides the server reactive plot assigned to the matching outputslot.
- [Display reactive output](https://shiny.posit.co/r/getstarted/shiny-basics/lesson4/)
  explains the UI/server distinction, matchingnames and evaluation/update flow.

This is **one authored documentation card**, not an automatic general web/RAG
service or a newlylearned SNNskill. The frozenVLM chooses an option actually
transcribed from the image; no C-label or `distPlot` key is stored in the card.
Runtime then renders the cited factual explanation, instead of letting the
smallVLM invent more free-form rationale. Output says “Đối chiếu với tài liệu”;
trace explicitly records `native_freeform_explanation_replaced=true` and the
producer. This does not prove the native model independently knows Shiny.
Selection can still be wrong, and matching two functions is not a general
semantic verification of arbitrary questions, negation or code requirements.

## Failure preserved, then actual end-to-end result

RC17 direct candidate probes picked C; first canonicalHermes acceptance instead
picked A despite matching referencefacts, and the following greeting repeated
that answer. The147unit checks did not establish realanswerquality. That
receipt is kept asFAIL; do not quote candidateprobes as Hermesacceptance.

RC18 isolates the bounded local choice task from irrelevant host-agenthistory
and tool schemas: only current userrequest, transcribed question/options and
referencefacts enter the tool-free selector. Normal requests retain their
original authority/permissions. A fresh greeting after imagehistory also gets
an explicit turn-boundary reminder; its reply is still generated, not hardcoded.

Actual canonicalRPC admission with the exact original image and exact query:

| Turn | Measured outcome |
|---|---|
| Imagequestion | GPUtranscription and selector, chosenC; explanation renders correct UI/server/identifier roles with sources,0Hermestoolcalls; about2,06s |
| “chào bạn” afterimage | Generatedgreeting, no repeated option; still mentions Shiny, so perfecttopicreset/persona is not claimed; about2,36s |

[Receipt/trace audit](evidence/ACTUAL_HERMES_IMAGE_QUESTION_20261115.json)
recomputes native usage, verifies transcribed syntax/chosenlabel, producer and
delivered response. Two local native calls occur inside one gatewayresponse;
the HermesAPIcall count1 is not “one VLM inference”. Reported completiontokens
count native transcription/selection, not the documentation words rendered by
runtime. No fabricated native explanation/loss/spikes or learning claims.

The first testharness attachment was refused before admission because it named
a retained input belonging to an earlier turn, rather than uploadstaging.
The harness now copies exact same bytes to a fresh profile imagecache entry and
uses normal Hermesattachmentadmission. Noownershipguard weakened, no original
session rewrite, no imageediting. The failed test and intermediatewrong answers
remain in private `local/` receipts; userpixels are not committed toGitHub.

## Release evidence and limits

149targeted image/schema/gateway/streaming/web/packagechecks passed. Tests cover
current-image scope, privateinstruction isolation, invalidschema fallback,
unchanged submittedtext/images, allstubchoiceletters to detect C-hardcoding,
explicit toolpriority and actual-native-token accounting. They are contracttests,
not149independent imagequestions. Shared332checks fromRC11 stay historical,
not silently counted as rerun. No neuralCPUtraining was started.

RC18 has59codepayloadfiles/884815bytes, sourcefc2aa9c09b852ac44c891ac3833aa8b69d89299a,
SHA2568096923f07833e9f38e9622cac7f80e1675d8a0464adfc3e7d1925be49a8f16e.
[Installedverification](evidence/RC18_INSTALLED_ARTIFACT_20261115.json) matches
allpayload/model/runtimepins. [Activation](evidence/RC18_ACTIVATION_20261115.json)
preserves savedweights/synapsestate/RNG/replay, liveupdates5; VLMvision/ASR/ALIF
remainGPU. Exact predecessor code retained forrollback, privateexperiences are
notrewound. Main/R13 remainuntouched. No new neuralquality/retentionPASS.

Knownlimitations: only this reportedcase was answered end-to-end; the card
covers a narrow Shiny API distinction, OCRcanhallucinate, nativechoicecanfail,
morethan6000visiblecharacters is rejected, a general-sourceverification/vision
benchmark is still missing. Do not market the workaround as SNNstrongerthanRAG
or a general fix for the0.8Bmodel's knowledge/reasoninglimitations.
