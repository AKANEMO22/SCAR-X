# Byte dialogue learnability controls — isolated backprop ONLY

Registered2026-11-09 before control training. User allows BPTT only as an
independent research control. All backward/autograd code MUST remain in
src/scarx/evaluation/controls.py; never import its weights into SCAR-X core.
No code path in serving/neural/learning imports Torch or this control.

Question: does the same small ALIF/codec/data learn more when its optimization
and temporal credit assignment are changed? This is NOT an alternative SCAR-X
model to deploy or a way to satisfy the local-learning goal using backprop.

Only run AFTER local byte trial is terminal, not concurrently. Same96train/
24development public Antigravity prefixes, same H48/K8/D454/O268 ALIF forward,
3ticks/byte, ordered/RGB compiler, UTF8host mask. Training-only calibration via
ByteDialogue; fixed presynaptic normalization as in byte_normalized_full.
New independent seeds7103,7121,7133; no weight/token/vocab import from Qwen.

Controls: Adam_readout (hidden frozen) and Adam_BPTT (all current synapses/bias
train). Both Adam lr0.001, grad-normclip1, outputweights/biasclip3, hiddenclip20,
4passes, same dialogue-order RNG per seed. One mean categorical loss over each
response, one optimizerstep/response; local trial updated after eachbyte, so
not equal-update/equal-compute. Report actual bytes, steps, walltime and memory.
Triangular surrogate derivative from the existing Spike control is not the
local-target candidate's derivative. Forward equations match, gradients differ.

Detach recurrent state at each response boundary but retain numerical values;
BPTT spans its input-prefix events and response. No reset across responses,
past state before that horizon has no backward graph. Evaluation is no-grad
and restores training state afterwards; no dev labels used by optimizer or
teacher-forcing inference. Autoregressive128byte outputs and teacher-forced
byteNLL/accuracy at0,2,4passes; retain all outputs, no model selection.

Descriptive learnability evidence: train byteNLL declines and teacher-forced
train accuracy>=65%; a >=10ppdev exact gain is stronger evidence in this small
set but is NOT a global natural-chat gate. Failure under this limited budget
is inconclusive about architecture impossibility. Never tune after outcomes.
Same target representability/OOV0 for both control conditions. No retention,
independent human/blind rubric, newimage generalization or release claim.

CPU1Torch/BLASthread, no CUDA allocation, per-condition180s/round1200s,
sampledprocessRAM<=2GB, Windows committedcap2GB. Save final numerical arrays,
forward-parity preflight and resources under a new control-only run. Check no
inference weight/statistics updates. These checkpoints never enter G1 release.
