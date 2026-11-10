# Pilot v2 — registered after v1 failure, before v2 training

V1 language stopped at16epochs after dev degradation; bestdev exact6.25%, finaltest0%, 'chào bạn' → 'Bạn,'. Digits full/readout both test66.39%, best at epoch1. Do not overwrite those results or deploy v1 as chat.

Diagnostic: with uniform44-way softmax, eta.2 and componentwiseclip.025, bias delta sum becomes approximately−.17045 instead of0. The positive target update is clipped while many negative updates are not; this changes the direction of the intended readout error. V2 sets readoutlr.015 so readout/bias gradient components remain belowclip.025 for features in[0,1]. weightclip20 instead of3; this changes numeric range, not neuron count/precision/RAM. No change to backward constraints.

Compare two fixedconditions full-local vs readout-only for language and digits. All other v1 initialization/encoder/time constants remain. H48/K8/R12. Language200epochs max, earlystop30stale after epoch30; choose checkpoint by development exact then loss, choose condition by same development rule before opening v2test. Vision24epochs max, choose epoch/condition by devaccuracy then loss; open test once for selected condition, disclose v1test reused as diagnostic, not a fresh independent research claim. No further hyperparameter search within v2.

Language train/dev same v1. Testv2 uses16new paraphrases saved in data/dialogue/pilot-v2-test.json before run. Same canonical answer vocabulary; output must come autoregressively from learned readout without lookup or intent routing. Same productcandidate criteria: devexact≥.75, finalexact≥.5, no generation loops, explicit greeting prompt yields greeting. Testing screenshot/general-language capability remains outside the evidence of this limited corpus.

Vision uses the exact v1train/dev/testindices. 8x8 digits only, threshold≥.75test. Bounded CPU one numeric thread, below-normal priority, process2GB, run deadline360s. Save source/datahashes and checkpoints, rawoutputs. Do not call a readout-only win evidence for hidden representation or fullcontinuallearning. Keep production model/checkpoint untouched until a limitedchat candidate qualifies.
