# P1 input-information audit v1

Registered2026-11-09 before measurements. Precondition to the master P1
representation/credit-assignment work, not a replacement quality gate.
No learner, live weights, framework, topology or deployment change.

H1: the G1 sum-of-token/character-trigram context encoder maps some
different temporal statements to exactly the same vector, making them
indistinguishable from the same initial neural state.
H2: grayscale digit preprocessing maps different visible colors to the same
64 neural pixel values, so color QA cannot be recovered by this downstream core.

Text:64development pairs, seed9107;64confirmation pairs, seed19107. Each prompt
has a fixed prefix/query and two state assignments with different values. Reverse
the two complete assignment clauses; ground truth is last assignment. English
and Vietnamese balanced. Values1..999, distinct within each pair. No target
channel or fact retrieval. Report exact float32 context equality, L2difference,
same-state output equality across existing public G1 checkpoints17/29 and fresh43.
Do not train or select checkpoints. If vectors are identical and targets differ,
any deterministic downstream response with the same starting state has at most
one correct answer per pair. This is an information bound for those prompts,
not a general claim about all text or all SNNs.

Vision: same synthetic hollow-digit mask, foregroundRGB(255,40,40) vsRGB(0,178,0),
black background,64x64PNG. Their intended color labels are red/green; verify
actual PIL luminance equality, pass both through the current real pixel loader,
report image-hash difference/pixel equality. The digit task remains supported;
this is a color-information counterexample, not a natural-photo benchmark.

Additional white-box boundary: verify current vision returns10digit outputs
and never runs neural word decoding. Cite exact implementation paths. No paper
claim or static UI flag substitutes for the data-path measurement.

Budget: CPU1thread, <=60s,2,000,000,000byte cap. Save raw prompts/results and
source snapshot in a unique run. Preserve previous results. Failure to confirm
the hypotheses is a negative result, not a reason to edit thresholds/pairs.
