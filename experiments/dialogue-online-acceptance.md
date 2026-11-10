# Online dialogue + idle review + asking the user — product acceptance

2026-11-09. User explicitly requests learning during dialogue, idle review of
conversations against neural knowledge, and proactive questions in Hermes when
uncertain. This supersedes the earlier product default of idle weights disabled.
No BPTT, pretrained proxy, unverified self-labels or full-model retraining.

- Chat retains source/session/neural output. Unsupported coverage/unstable
  generation may trigger a controller question; these heuristics are NOT
  calibrated confidence or proof of semantic understanding.
- Explicit correction or answer to a pending controller question provides a
  desired response label, provenance retained. Arithmetic labels independently
  checked when simple operands/operator can be parsed. A verifiedflag is ignored.
- Local micro-updates between turns, at most32attempts/lesson. Guard against loss
  of previously correct authored development probes/accepted other lessons.
  Clone-only canonical evaluation may reset diagnostic state; live core never
  resets. Rollback wholemodel/controller on rejected updates or commit failures.
- Idle reviews compare live predictions to explicit targets; old generated text
  is not a target. A drift without evidence asks for help rather than promoting
  its own old or new answer to truth. Replay only approved labels, max3bounded
  replay rounds. No claim that guard solves catastrophicforgetting.
- Persist pendingquestions/queue/reviewcursor/provenance/counters along with
  neuron/topology/RNG. New sessionformat reads legacy; old runtime rejects new
  bundles rather than silently dropping learningcontrollerstate.
- One pendingquestion/session, max16total; bounded64episodes/64lessons. Hermes
  uses its existing clarification UI and an opt-in single watcher, no fake
  reasoning_content and no independent second serving model. Timeout/noanswer
  never becomes a learninglabel. SessionIDs prevent cross-chat answer mixups.
- Test falsearithmetic feedback, pendingquestionrestart, idleweight invariance
  with no accepted labels, actualweightupdates from humanfeedback, guard rejection
  and DB faultrollback, headless transport and native Hermes question UI.
- Historical researchforgettingmean≤3pp/worst≤5pp unchanged and stillfailed.
  Data generation/bootstrap is separate from this persistent online lifecycle.
