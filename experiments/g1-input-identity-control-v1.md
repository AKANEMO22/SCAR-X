# G1 fixed-budget input identity information control v1

Registered2026-11-10 BEFORE labelperformance measurements. MasterP1 input
representation diagnosis after wordscoretransfer/beam/stateviews failed. No
coretraining, new learningrule, scale, response lookup or deployment. Entirely
public181train/31dev/prior48 from reviewed sources; no privateweights/feedback.

Fourpredefined views, no gridextension:
legacy128: existingG1word+charhash, terminalsurface normalization as fixed
surface_expanded source ofrepresentationaudit.
dictionary128: top128WORDtypes by trainingdocumentfrequency only, tie lexical;
wordtoken=existingtokenizer containinganyalphanumeric, punctuationexcluded;
counts×IDF(log((1+N)/(1+df))+1), perpromptL2normalized, unknownignored/padded.
hash_same_dictionary128: EXACTsame128selectedwords/counts/IDF, but existing
blake2signedhash128; isolates collision effectvsinjective wordIDs.
full_dictionary_reference: ALLtrainingwordtypes/counts/IDF, L2norm. This has
moreinputfeatures and is an INFORMATIONUPPERREFERENCE, not matchedcapacity/
permissiontoenlargeneuralmodel or bypass P2scalegate. No pretraining/embeddings.

Selection/IDFonly trainprompts, neverclasslabels/dev/prior. Unknownwordcoverage
perview/split reported, not an implicittruth/confidencescore. TokenIDs fixed
forthisdiagnostic; future runtime codebookgrowth/retention is unimplemented.
Bagviews don't solve order/binding/general-language understanding, even if
closed8intent labels become predictable.

Train diagnostic8classridge.01 (train-onlycenter/std, dualsolution exactly
same regularizedobjective as priorprimalprobe), and existingoutput-localsoftmax
20passes/lr.08/clip.025/weights3/orderseeds1901/1907/1913. Samefeatures/labels,
no neuraltemplate answers; all predictions areINTENTLABELS only. Threehead
orders are not independentmodelseeds; no inflatedCI/generalqualityclaim.

Descriptiveinformative dictionary128 if devlocalheadmean>=65%ANDgainvslegacy
>=10pp AND priormean>=65%. Ridgealone doesnotpass. This isn'tP1/P2success;
positive only motivates separatelyregistered freshneuraltest with same128input
budget and privateguard beforeanymigration. Negative stopthisviewcontrol;
no extra vocabularysize/hashseed/featuregrid or labelsupervisedtokenfiltering.

No source protocol/result rewrites, oldSTOPfamiliesstaystopped. Save matrices/
metadata/predictions, publish actualclassification/progress with explicit
nonneuralinputcontrolscope. CPU1thread,2GBcap,120sround; stop NaN/leakage.
