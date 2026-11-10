# RGB object diagnostic — source amendment before any image measurements

2026-11-09. Original university binary download exceeded360s after only~18MB;
process exited1, no dataset/training/holdout opened. Failure receipt kept.
Replace transport with University of Toronto Computer Science HFdataset
https://huggingface.co/datasets/uoft-cs/cifar10, revision
0b2714987fa478483af9968de7c934580d0bb9a2. Trainparquetfile
plain_text/train-00000-of-00001.parquet,119705255bytes,SHA256
8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883.
No testparquet downloaded/opened, no dataset loader remote code or pickle.
License status on dataset card is unknown; local research only, no image
redistribution in SCAR-Xpackage. Preserve dataset attribution/citation.

Read FIRST10000trainingrows of pinnedparquet via existingpyarrow, decode32x32
PNGbytes withPillow, convert to the same label+3072planarRGBrecord format for
the diagnostic. Row order may differ from officialbinarybatch; do NOT claim
same exactsampleindices/data as the never-completedbinarytrial. Sampling rule
seed10103/perclass64train16dev is unchanged and applied BEFORE measurements.
Hashduplicate guard stays; do not tune split/conditions/thresholds from results.

Everything else fixed in rgb-object-grounding-v1:48ALIF/K8/R12/D454, raw and
normalizedoutputlocalhead,640train160development,threefreshseeds,4passes,
imageblank/swappedcontrols,30%accuracy/+10ppgrounding screen, no normalphotoQA
claim, no hiddenrepresentationgain or livecandidate replacement.2GBCPU1thread,
download<=360s/180MB, experiment<=120s. Pyarrow/Pillow already installed; no new
framework/model/APIcharge/privateinputs.
