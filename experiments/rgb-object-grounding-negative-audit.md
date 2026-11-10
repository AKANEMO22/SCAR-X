# Additive negative-pairing audit

2026-11-09 afterv1classifier measurements. Source trial's mismatchcontrol used
roll1 on development images ORDERED BY CLASS (16perclass).150/160swapspreserve
class, so this is mostlysameclass perturbation, not a true wrong-object test.
Keep raw results and source unchanged; never call that1ppdifference a valid
wrong-class causal bound. Thresholds/split/trainingdecision remain unchanged.

Recompute pointwise classifier mismatched predictions using a deterministic
bijective cross-class mapping: sortlabels,rollbyoneclassgroup16,scatterback.
Assert100%differentclass/noidentity. The saved cleanpredictions can be permuted
directly because readoutforward is pointwise/no state, so no retraining/new
testlabels for modelselection. Existing blankcontrol and accuracy remain.
Persist corrected receipt/report separately. This audit can show sourceimage
alignment, not specific learned semantic features or naturalcaption/QA ability.
Normalgrounding screenstillrequires>=30%accuracy AND>=10ppoverblank/mismatch;
do not lower it after20%actualaccuracy.
