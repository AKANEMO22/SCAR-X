# Local magnitude protection prototype P0 —14/11/2026 before numerical checks

Reason: Hybridaudit9d1545 proves stored importance/stable arrays inert in current
update. User's lifelong/local-synapse research direction includes metaplasticity/
fast-stabletimescales. Reuse the documented PlasticCore._update magnitude rule as
an ISOLATED research prototype, not servingimport or proof it solves forgetting.
No new credit path, backward/BPTT, neuron size, pretrainedVLMupdate or LoRA.

Keep LocalALIF forward/eligibility/readout-error feedback and its clip unchanged.
After originallearn, get actual per-weight FP32delta from before→after. Filter it
using OLD importance/anchor, not futureupdatedimportance:

    protected_delta = actual_delta/(1+5*omega_old)
                      - .005*omega_old*(weight_before-stable_before)
    weight = clip(weight_before + clip(protected_delta,-.025,.025),-3,3)
    omega = min(20, omega_old+.001*abs(actual_delta)/(mean(abs(actual_delta))+1e-6))

Same local-magnitude heuristic as historicalPlasticCore, applied to actualclipped
candidateupdates to avoid changing credit. Hidden: parentimportance matrix;
readout+bias:new boundedFP32importance arrays, because oldparent has none for them.
Ignore groups withzero candidateupdate; readout_only never learns hidden state.
EMAstable toward weights rate.02 every64successfullearnattempts (a timescale, no
task boundary). Plainmodefrozen doesnot update weights/protection. Replay uses
existingreservoir and verifiedevidence; consolidation onlyexplicitcalls.

This is a changedresearchalgorithm, namedscarx-magnitude-protection-prototype-v1,
not claimed exact gradient, causalimportance, newliteraturemethod or production.
SeparateZIP_STORED envelope bindsalgorithmconstants,innerpayloadSHA andextra
importancearrays; currentlive loader cannotload it. Save fullanchors/importance/
traces/stats/RNG/replay, reject duplicates/compressed/unbounded/badtype/shape/
algorithmversion. GPU neuralcalculation; CPUonlyinit/RNG/serialization.

P0numericfixtures only, no taskdataset/DEVselection or accuracytraining. Fixed
3seeds260113/137/171,4owned randomfeaturesequences8×64,32repeatedupdates to
exercisebranching; targetindicesfixedexternalfixture. Check importancechanges,
firststepzeroimportance matches original withFP32tolerance≤3e-7, positiveomega
attenuatesactualdelta withanchor=before, at64EMAanchorschange, modefrozeninert,
readout_onlyhiddeninert, coldGPUresume/nextupdate/fullcheckpoint exact. Explicit
reward sign remainsoriginal, no teacher/modelgeneratedselflabel. These are
algorithmunit stimuli, not acquisition/retention/hiddenrepresentation evidence.

≤2minutes,≤300totalnumericupdates,extraarray/checkpointbytesreported. Do not touch
active8hsource/live/main/R13, do not overwritehistoricalcheckpointformat or claim
scientificGO from P0. Any learningDEV needs its own freshbank/protocol/matched
delta+anchor-only/importance-only/replay/normalization/readout/hiddenablations
and unchangedacquisition/3-5ppretention/5ppCIgates. Do nottrain196043bank whose
finiteoraclegain over simpleprototypecontrol is4.17pp<5pp.
