# Ordered text / RGB event compiler prototype v1

Registered2026-11-09 before measuring candidate encodings. This follows masterP1
and the information-loss audit; no topology change, extra model, pretrained
encoder, new framework or live-checkpoint migration. Experimental only.

Text: preserve each Unicode token as its own chronological event in the same
128hash channels used for text. Token identity + its own character trigrams,
signed Blake2 hashes, normalized per event. Speaker/start/end/position markers
in the existing6control channels. No clause parser, answer lookup or target
information. Reversing clauses must reverse event order. Do not advertise
injectivity for arbitrary text:128hashchannels can still collide.

Vision: decode bounded PNG/JPEG, retain RGB, resize16x16 and stream16nonoverlap
4x4RGBpatches in raster order.48color channels +2normalized patch coordinates
fit within the existing64image channels. No occupancy/digit-domain claim. All
images bounded500KB/1Mpixels/aspect ratio4:1. Questions are text events followed
by image events. This preserves coarse color/position, not arbitrary image
detail, OCR or object understanding. No labels/input features from filenames.

Same channel layout as G1: reserved previous-word one-hot V +128text +64image
+6control. Compiler accepts V as metadata, and no neural model or labels.
Max192text tokens,12conversation messages,256total events; reject oversized
input rather than silently remove old turns. UTF8text input<=2048chars/message.
Bounded full dialogue is an experimental compiler input, not a revived1Mcontext
goal. Live shared core/memory remains unchanged.

Counterexample validation uses SAME locked128text pairs and red/green fixtures
from p1-input-information-audit-v1. Measure sequence equality/L2, pool-summed
equality separately, RGBsequence differences, and whether output is finite/bounded.
Do not select parameters/checkpoint from these examples. Input0knowledge/0learning
updates,<=60sCPU,2GB. Passing means only information is PRESENT for a learner.
Task success/hidden representation/retention/general conversation still need
independent registered training and held-out evaluation. Local-targetP1 failed;
this encoder result cannot turn that failure into a pass.
