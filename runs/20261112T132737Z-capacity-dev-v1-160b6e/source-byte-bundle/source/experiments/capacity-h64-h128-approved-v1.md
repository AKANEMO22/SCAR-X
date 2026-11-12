# Capacity diagnostic — H64/H128 đã duyệt, 12/11/2026

Phê duyệt trực tiếp của người dùng: “ok tôi duyệt thử H64 và H128, bắt đầu chạy
đi, giữ nguyên ALIF và học synapse cục bộ, cho quá trình học với kết quả lên
localhost để tôi xem luôn”. Đây là quyền chạy research secondary ở2size đã
nêu; không merge/main/Hermes/liveSQLite, không confirmation/P2 hoặc language.
Ponytail không có trong tool inventory. Base22bb9781f4ed8f60f7180bdb56b6570e64e1f503.
Proposal cũ giữ nguyên lịch sử; approval này supersede “chưa phê duyệt” ở đó.

## Hypotheses và invariants
H48 có thể thiếu capacity; hoặc estimator/readout/optimization là nút thắt.
Không suy ra SNN vô dụng từH48, không suy ra hiểu tốt từ RAM thấp/tăngneuron.
Giữ nguyên forward SpikingCore ALIF (hardbinary, subtractive adaptive reset),
D8/fan8/O2/R12, input/data/readout/learningrates/clips/statistics/optimizer.
Không Transformer/pretrained/autograd/BPTT/RTRLcandidate. One-step readout
error feedback+own-neuron eligibility theo ngoại lệ research đã duyệt trước.
H64/128 là fixed allocations của trial, không tính năng tự grow theo task.

Port constructors/trace allocation theoH; kế thừa nguyên source step/learn
từ own-reset và ACRgroups_up. Gamma vector vẫn linspace(.25,.92,H); groups
hai nửa .25/.75. Fan8/H thay đổi mật độ khi tăngH; báo như structuralcovariate,
không gọi đây là biến đổi chỉ số lượng neuron tuyệt đối. R12 chỉ là legacy
core trace blocks; source research optimizer thực sự cập nhật tất cảHrows.

## P0 và checkpoint trước mở quality
H48 losses/weights/active traces/normalizer/momentum/energy/RNG khớp exact với
matched own, ACRgroups_up, readout vàshuffles qua technicalfixture vàcoldresume.
Forwardbitwise invariant cùngH; allconditions/init sameH exact; crossH same
seed/distribution chứ không giả weights khácshape là bitwiseidentical.
Fullboundeddata-onlycheckpoint cho48/64/128, distinctformat, strictshape/dtype/
ZIPmembers/hash/config/policy, allstates/RNG/coldresume/rollback; không padding
H48 thành128 hoặc thay oldformats. Sourcecore production không sửa.
P0failure blocksquality. Independentreal reviewer trước quality.

## Finite DEV grid và ngân sách khóa trước scores
Seeds140011,140017,140023 là reusedexploratoryDEV, KHÔNG freshconfirmation.
Tasks strict_binding,xor,distractor_binding,order; exactfrozen oldsplits
1024TRAIN/256refit/512ID/512OOD, calibrationfirst32TRAIN; labelsindependentoracle.
Arms own(singletrace),groups_up(dualtrace),readout_only(singletrace) và
groups_shuffle(cùnggroups_up, chỉ shuffledneuron learning signal) primary.

Primary1024updates:3H×4arms×4tasks×3seeds=144records. Reuse48H48records từ
ACRfinal59ced4 (own alpha0,groups_up,readout_only,groups_up-signal_shuffle)
sauP0parity, copyhash provenance, không trainlại vìscore. H64/H12896newjobs.
H48own/readout sources dualtrace nhưng forward/update/predictions phải exact;
không dùng cachedCPU cost của dualtrace để claim ownsingletrace efficiency.

EqualCPU secondary0.5processCPU seconds, checkepisodeboundary, cap4096updates,
3armsown/groups_up/readout ×3H×4tasks×3seeds=108newjobs. H48timing được đo mới
trong adapter hiện tại để so CPU đồng thời; đây là timingcontrol có ngân sách
khác primary, không chọn lại qualitywinner. Giữ negatives/overshoot/counts.
TOTAL252records,204newtrainjobs+48reused; no profile/lr/seedsearch.

Đo direct/probe và ID/OOD riêng; refit chỉ dùng refit data cho center/scale/ridge.
Revert input/recurrent/both hidden, giữ learnedhead, refitprobe đúngintervention.
Fullrevert phải recoverreadout-onlyprobe cùngH/seed/task chính xác. Không dùng
probe tốt thaydirectskill. SameHsignalRNGdrawschedule matched ởprimary.

## Decision trước xem dữ liệu
SoH64/H128 với H48 trong CÙNG arm, không chọn bestarm theo task. Chỉ báo
CAPACITY_DEV_SIGNAL cho mộtH nếu >=2/3primaryfamilies có meandirectgain>=.02,
không family nào IDdirect/probe hoặcOODdirect/probe regress>.03, orderdirect
>=.65 và regression<=.03; learned-vsreadoutprobe advantage tăng>=.02 ở>=2families,
causal directrevert_bothdrop>=.03 ở các family cải thiện. Đây chỉ effectgateDEV.
Rawfrozen/readoutimprovement riêng gọi CAPACITY_ONLY_SIGNAL, không neurallearning.
EqualCPU thắng/thua báo riêng, không thay primary. Không tiến hơnH128 từvòngnày.
Không đạt gọi MIXED/NO_CLEAR_CAPACITY_GAIN, không kết luận architecturalceiling.
Candidate broad/P1 cũ vẫn cần2/3primary đầyđủdirect65/CI5pp/Holm/shuffle/OODNI;
không formalPASS với3seeds (exacttwo-sided signpmin.25). Không đổithreshold.
P2retention chưa đo; acquisition/final65%,mean3đ/worst5đ theo seed giữ nguyên.

## Resources, stop và trace
One numericalthread, one worker. Max60min từ workerstart gồm data/calibration/
training/evaluation/checkpoint,120sec/newjob. Tính riêng bounded10minaudit sau
worker; không claim60minincludesaudit. 2.000.000.000B/process vàtree RSS+commit
sampled; Windowscap nếu khả dụng. Không pagefile/SSDweightstreaming. Không GPU.
Beforelongrun: technical throughputfixture (khônglabelsbenchmark) và budget
ước lượng, không thu nhỏ samples sau khi scores xấu. Nếu estimatedgrid vượt
workerbudget, failpreflight/revise BEFORE scores, không tự bỏ seed/cell.
Stop nonfinite,oracle/split/hash/P0/resource/timeviolation; preservepartial,
missingcannotPASS; noautomaticrerunseed/hyperparam. Checkpointssaved percell.

Passive observer8773 riêng:actualinputevents,neuralprediction/oracle,loss,
updatecount/norm,hiddenH,spikes/resources,coldsave/rollbacklogs; khôngfake chat,
Qwenoutput hoặc suy nghĩ bằng lời. Không chiếm8767/8771/8772 hay processkhác.
Đưa cả technicalactivity thật lên log, không gọi nó là neuron thinking.
Commitprotocol/code trước quality; source/peer/data/approval hashes trongrun.
