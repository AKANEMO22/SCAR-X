# Literature-first reference DEV v1 — trước mở kết quả, 12/11/2026

Nguồn chỉ thị: attachment3367eb19-6038-4670-b463-9f7044841f31. Base
950718bd410371f8ef15dab2d78d722b04ab506d; nhánh nghiên cứu riêng. Main/Hermes/
SQLite/live giữ nguyên. Không fresh confirmation, không P2. ACR có55job hoàn tất
trước gián đoạn credits; không thêm biến thể ACR trước đối chiếu reference.

## Câu hỏi hữu hạn
Reset Jacobian nào và filter nào giải thích tradeoff hiện có? Chỉ5learner:
own và stopped pinned607/bc686500; transformed eprop_matched/eprop_aligned;
BrainTrace block D-RTRL adapter. Thêm readout_only; uniform frozen alias đã kiểm.
Không gọi adapter thay forward là nguyên bảnpaper. Tất cả vẫn một H48 D8K8
ALIF, hardspikes, topology/init, local pre-update feedback và source optimizer.
Không TensorFlow/JAX/framework trong candidate; full sensitivity oracle chỉ là
control frozenweights, không train và không serving. Không chọn phương pháp
dựa vào confirmation bank đã mở.

## Đăng ký nguồn và biến đổi
- E-prop efd02e6879c01cda3fa9a7838e8e2fd08163c16e, BSD3:
  https://github.com/IGITUGraz/eligibility_propagation. Original resets constant
  threshold with delayed spike. SCAR same-tick adaptive threshold retained;
  eprop_matched detaches resetspike but retains -z*beta*ea; matched gate/filter.
  eprop_aligned additionally no sensorygate and gamma=actual feature_decay.
  Both retain SCAR logistic surrogate (original triangular only numericalfixture).
- BrainTrace fafa25753b743a2e2d5b47209c4ee947abc02cfb, Apache2:
  https://github.com/chaobrain/braintrace. NumPy explicit within-neuron3x3block
  D-RTRL, recurrent mixing cut including self; no sensorygate, actualfeaturefilter.
  Not execution of upstream compiler. Zero additional persistenttracearrays.
- SI helper frompathint86c5e07c5603d6a44c2f53585c19ee70773ef15c, MIT, technical
  tests only. Explicitconsolidate is boundary-assisted, not stricttaskagnostic.
- DECOLLE latestactuallicense includes noncommercialreservation; do not copy.
  DGN changes neuronconductance/dynamics, outside unchangedprimaryforward.

## P0 trước train
Independent equations, suppliedlocalJacobians and anchored surrogatefinite-diff;
actualupstream epropAST recurrence withNumPy primitive shim; hardforwardexact,
sourceoptimizer/readout/RNG unchanged, no autograd/BPTTcandidate. Strictcheckpoint
allarrays/settings/RNG, coldresume+nextupdate. RAMincludes checkpointbuffers.
Failure blocks DEV; fixcode beforeopening, never changethresholdafterresults.

## Paired schedule
Reuse exploratoryDEVseeds140011,140017,140023 (not fresh), four unchangedtasks
strict_binding,xor,distractor_binding,order. Split1024TRAIN,256refit,512ID,512OOD.
Calibration first32TRAIN. Same1024updates in sourceorder, sameinitweights/topology.
Six methods ×4tasks×3seeds=72primaryjobs. Fit probe only on refit; standardize
using refit only. Direct/probe, OOD and revert_in/revert_rec/revert_both saved
separately, allrawpredictions, perupdate loss/appliedhidden/readoutnorm andCPU.
Revert_BOTH must recover readout-only probe exactly. Originalown/stopped use
singletrace: CPUcomparison reflects actualimplementation, unlike dualACRendpoints.

Nominate at mostONE globalreference fromeprop_matched,eprop_aligned,braintrace;
rankminimum meanIDdirect/probe difference vsbetterown/stopped across3primary
tasks, thenmeanprimarydirect difference, alphabeticaltie. No per-taskpolicy.
Runone nominee shuffledsignal ×12cells. Then equalCPU own,stop,nominee,readout
0.5processCPU seconds/episodeboundary, cap4096updates ×12=48secondaryjobs.
Maximum132jobs. No coefficient,lr,surrogate,gate or seedsearch.
Methods are trained separately on eachtask; this does NOT demonstrate a single
accumulated skillmodel or retention. Joint/curriculum/continualP2 not opened.

## OOD/interpretation
UnchangedprimaryOODgeneratedsplits. Additional posthoc-fixedweightsstress on
nominee+own+stop, first128ID pertask: insert12zero delayticks beforequery/end;
nonfinal6:8 background+distractorpulses x3; swapnonfinal6/7 preservingendmarker;
semantic eventrowchannels0:6 x.7; bindingqueryflip pairedsamewrites/relabel.
No fitting/tuning ontheseoutputs. See ACR stressdefinitions; reportexacttransform.
Frozen-parameter directionoracle on fixedsynthetictechnicalfixtures is separate
from accuracy; normalizeddirectionalignment is not proof of learningadvantage.

## Locked exploratory report rules
No P1PASS from3seeds. Report perseed effects, descriptive95% pairedclusterCI
and exact8sign-flips (minp=.25), no pseudoreplication. No replacing direct withprobe.
REFERENCE_DEV_PROMISING only if >=2/3primaryfamilies beat betterbaseline mean
direct by>=.02, direct>=.65 each, probe gainvsreadout>=.05 and causal direct
revert_bothdrop>=.03 onimprovingfamilies; neitherIDdirect/probe regresses>.03
vsbetterbaseline, neitherOODdirect/probe regresses>.03vsown, orderdirect>=.65
andregression<=.03. Nominee-vsshuffle probe>=.02 on>=2families andnoprimary
directregression>.03. Otherwise REFERENCE_MIXED/NO_CLEAR_GAIN asfactsdictate.
Theseeffectrules do notestablish norm-independentadvantage: rawupdateamplitude
isreported; no matchednormproof claimedinthisfirstreferencecomparison.
Ifreference strongerthanACR on availablepairedcells, prioritizepublishedmechanism;
do notrescueACR. Missingcells neverPASS. FutureP1gates unchangedrequiresapproval.

## Limits and evidence
One numericalthreadperworker, <=2,000,000,000Bprocess/tree RSS+commit sampled;
max50minworkerwall plus separately bounded10minaudit (60mincompute total),
max120sec/job. Once-reservedrun, uniqueoutputs,
checkpoint/hash allsource andpeer/upstreambodies. No production imports from
externalrepos. Externalclones liveunder scar-x/local/references, pinnedreadonly.
Observer separate8772, never8767. Outputs actualresults not simulatedthought.
Stop onnonfinite/oracle/split/hash/resource/timefailure; preservepartialdata.
MainSHA e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541 verifiedbefore/after.
