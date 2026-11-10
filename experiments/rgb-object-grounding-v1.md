# P1 RGB object grounding diagnostic v1

Registered2026-11-09 before image retrieval/split/training. Main goal includes
normal-photo perception, so a color-information counterexample or digit recall
cannot qualify. This is a SMALL learnability/precondition diagnostic, not normal
vision completion. No pretrained vision/LLM/proxy or new framework.

Source: official CIFAR10 binary dataset URL
https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz (redirect permitted to
official dataset host), MD5c32a1d4ab5d03f1284b67883e8d87530. Dataset reference:
AlexKrizhevsky2009, Learning Multiple Layers of Features from Tiny Images.
Official description https://www.cs.toronto.edu/~kriz/cifar.html. Download max
180MBcompressed, memoryboundedstream, do NOT unpickle or unsafe extractall.
Read ONLY data_batch_1.bin,3073byte records; originaltest remains unopened.
No dataset redistribution in productZIP, localresearch files excluded.

Split locked seed10103: per10classes64train/16development from the first training
batch, disjoint imageindices. Source-providedlabels (not reverified perphoto),
detect train/dev image-hash duplication before measurements; stop if found,
do not silently move test into train.640train/160development is intentionally
small. Images32x32coarse; not an OCR/screenshot/real-world deployment benchmark.

Fresh independent seeds10111,10133,10139. Same byte-backbone48ALIF,K8,R12,D454,
no hidden training/topologygrowth. Input same fixed userquestion followed by
RGBpatch events via InputEvents; no filename/classchannel in input. Readout
classification has10classes (diagnostic output), not natural-language generation.
Do NOT claim class-name formatting is a learned fluent response.

Imagefeatures: mean48spikefeatures over16imagecontentpatches (skip boundary
events), not just finalbottomrightpatch. Readout uses existing ConditionedReadout
raw vs train-calibratednormalized. Calibrate640train features only;4passes,
lr0.08/clip0.025/weightsclip3, localoutputdelta. All conditions same class/samples/
steps. ScalarhiddenSTDP not added because the current learning-family gatefailed.
Use actual core reset on INDEPENDENT diagnostic images to remove orderbetween
different photos; clearly distinct from livecontinuousstate. No privatecore reset.

Final classify untoucheddevelopment, report overall/perclass/confusion,train
accuracy. Blank-imagecontrol uses allzeroRGB pixelchannels but exact same
patchcoordinate/control markers; mismatched-imagecontrol circularly permutes
encodeddevelopmentfeatures. Channel inputs stay targetfree. Correct-label
scores should exceed blank/mismatched by>=10pp AND be>=30% overall (10class)
before calling the featurepath informative; this is not a broadvision gate.
Any advantage must be confirmed on8freshseeds/newimages/class-wordgeneration/
question/paraphrase/OODimage before release. No modelselection by test.

Primary gap attribution: if raw/normalized classhead fail, worddecoderfailure
cannot be solved just by captions. If classhead succeeds but wordheadfails,
decoder/credit remains a separate issue. This diagnostic never chooses weights
to replace currentG1; P1representation/P2retention requirements unchanged.

CPU1thread, sequential<=120s excludingdownload<=360s,2GBcommittedcap and measured
RAM, no CUDA. Save source hashes/sampleindices,labeldistribution,predictions,
local-onlyimage receipts; no query tocloudteacher with privateimages. Generate
PNGbytes from indexedpixels only for encoder (no hostvisionanswer heuristic).
