# Trạng thái SCAR-X

## Audit 38: bài học còn hẹp, có khoảng cách tiền tố — 2026-11-10

Sau câu hỏi người dùng về khácbiệt SNN/Transformer, [protocol38](research/38_WORD_TRAINING_OBJECTIVE_AUDIT.md)
chỉđọc15checkpoint36: taskprefixstructure/FREEvsGOLDPREFIX/đổicâuhỏi và
potentiallocalcredit, khôngoptimizer/learningrule mới hoặcneuralupdates.
201 kiểm thử đầy đủ và 4 kiểm thử chẩn đoán bổ sung qua. Audit15sources×128
cases/7.680forwardsegments hoàn tất,0neuralupdates/khôngTorch. Corpus8mẫu có
8từđầukhácnhau,27/35targetpositions là các từsau; chưa là ngônngữ tổngquát.
Correctdev cùngvịtrí: GOLDPREFIX77,21% vs neuralprefix47,87%, gap29,34đ
CI26,69..31,90,89/648laterpositions bịdừngsớm/censored. Cờprefixreliance
khôngđạt ởcorrect; khôngkếtluận modelchỉdựatiềntố hoặc tựlọcbỏlatercredit.
FREEexacttoken+EOS32,81% khớp source36strict;35,94% cũ lànormalizedstyle,
không thaymodel/metricgate.18,84s/RSS52,74MB/commit37,49MB. Các STOP35–37,
mainALIF/2GB/P1/P2 giữ nguyên; chưa selective learner hoặc training mới.

## Directvisual37: chưa có representation/groundinghữuích — 2026-11-10

[Protocol37](research/37_DIRECT_VISUAL_BRIDGE.md) giữsame32ALIF/input/cache,
eligibility35 mappedonto actualpooledfeature, khôngTransformer/mainchange.
P0`20261110T064337Z-direct-visual-preflight-v1-927d70`:21same32calibration/
forward/headparities/16TRAINresponses mỗi nhánh khớpbytes, pooledsensitivity64
finite-diffmax1,78e−7, midimage/endword/Adam/RNGexact. Forecast1327,90s/21trials,
array485832bytes/model (cảpendinginput/cachetraces), peakcommit987,24MB.
5directvisualtests và2archive/metrics tests qua; fullsuite197passed/48,00s với
BLAS/OMP/MKL1thread trướcformal, guard2GB khôngđổi.21trial đãcompleted/coldexact:
correct25,10%dev/51,53%train, frozen15,05%, shuffled21,32%, BPTT26,30%.
Correct−frozen10,05đ CI7,81..13,32 nhưngcorrect−shuffledCI−6,16..11,00 và
resetheadprobegain−9,26đ CI−11,46..−7,29. Swappedimagedrop2,18đ CI−4,69..11,89;
shape1,74%, chỉfrozengainqua. STOP_DIRECT_VISUAL_BRIDGE_V1, khôngnormalvision/
P1/P2/deploy/scale hoặcretuning.330,14s, treeRSS643,47MB/commit1042,43MB.
UIactual48neurons/RGB/wordoutputs vàcompletedfilearchive864cases/96ảnh;
mainALIF/privateweights khôngđổi, Qwenactualtext36 giữriêngkhônggiảvision.
3freshseed37313/37321/37339 ×7conditions×1536experiences đãkhóa trướcscore.
ReusedactualAGY32questions/owned192PNG, khôngnewteacher/Qwenvision hoặcfakeoutputs.

## Wordbridge36: hidden cóđónggóp nhưng tổngquáthóa chưađạt — 2026-11-10

[Protocol36](research/36_DIRECT_WORD_BRIDGE.md) là pilotword đãnêu35 sauP0,
khôngconfirmation35/P1waiver/scale. AGYCLI đãsinh128prompt thật/62,66s; review
toànbộ tìm1mixedlanguage, giữgốc/cáchly và1AGYrepair đãđọc.36A sửa ownedtarget
apostrophe trướcneuralcode/measurements, giữtokenizer.36B sửa languageinvalid
trướctraining; cảhaiamendments giữhistory/criteria. Dataset64train64dev,3prompt
trùnghistorical,6caveats, không gọi freshfinaltest hoặcteachertruth.
Wordprototype khớpG1legacycontext/3tickinputs,4tests codec/rollback/control qua.
Preflightseed36303/64TRAINresponses:zero=frozen/graphparity/continuationexact,
forecast472s/15trials, peakcommit983,07MB. Formal15trials completed:
correctdev35,94%/train99,48%, frozen16,67%dev, shuffled17,19%, BPTT36,46%dev.
Correct−frozen+19,27đ CI18,75..20,31; resetheadprobegain14,06đ CI0..25 nên
chưagate, macro/familyaccuracy cũngfail. STOP_DIRECT_WORD_BRIDGE_V1, không
tuning/scale/deploy/P1waiver.15coldprocessresumesexact,68,28s/RSS627,22MB/
commit1024,74MB.190tests qua1BLAS/OMP/MKLthread; default2fullruns1legacyguard
HTTP507 retained, khôngnớiguard. Qwen64samepublicprompt đãđo:64/64,14lengthcut,
320,99s, RSS3,363GB/commit4,047GBreferencebudgetriêng, ownedserver đãtắt.
Rawoutputkhôngthànhnhãn/gradeexactstyle; GUI tựhiện64pair/archive và bảng5methods,
có cảcâu sai. Mainprivateweights/core giữnguyên, chưa generalchat/vision/lifelong.

## Nhánh30 v1 đã thử: XOR có tín hiệu, chưa qua P1 — 2026-11-10

[Protocol35](research/35_DIRECT_READOUT_ELIGIBILITY.md) khóa equation own-neuron
forwardeligibility × one-stepreadouterror, chưadeployHermes hoặc scale.
6targetedtests qua:64synapsefinite-difference anchoredsurrogate, forwardbytes,
partialstate/optimizer/RNG, zero=frozen, locality/clipcutoff và transactionrollback.
P0pass maxfinite-differror2,59e−8. Screen45trials/3freshseed đã hoàn tất:
order98,44%direct/+2,34đprobe; XOR91,80%/+10,94đ; binding63,41%/−1,95đ.
Chỉ1/3qua screen, STOP_DIRECT_READOUT_ELIGIBILITY_V1, không8seedconfirmation.
XORcorrect−frozenaccuracy+10,55đ (CI3,52..18,75), đúngtínhiệu tốt hơn shuffled;
3seed chưa đủP1, khôngnormalchat/vision/retention. BPTTbinding63,54% cũngfail.
184tests qua. LỗiWinreadheartbeat đãsửa bằngboundedretry/filelocktestthật;
38completedmodels/coldproof giữbyte,7thiếutrain, partial832+countkhôngbiếtđủ.
Round624,82s tính cảrepair; recoveryRSS621,75MB/commit1055,83MB, originaltreepeak
chưa đủsốđo. MainALIF/241363updatesgiữ nguyên; master1.25researchngoạilệđãduyệt.

## Audit đường thông tin34 đã hoàn tất — 2026-11-10

[Protocol34](research/34_VISUAL_INFORMATION_PATH.md) chỉđọc6checkpoints32/33,
đo actualinput/state/cache/BOS/readout và fixedprobe ngoàicore. Đúng1536train
exposures/seed, không thêm nhãn/coreupdate, Torch hoặc mainchange.3tests
diagnostic đãqua; fullsuite176passed/64,97s trướcaudit. Run
`20261110T000339Z-visual-information-audit-v1-882afb`:10368BOSobservations,
373248ticks, neuralupdates0. Rawsemanticprobe50,22%dev, headfeatureprobe8,30%
frozen/11,60%control; controlgainCI95%−4,19..8,14đ, chưa diagnosticgate.
Firstwordprobe kémactualhead16–19đ; khôngBOSreadoutgap. Các fullpathstages
khôngcontradictoryexactcollisions, clip0,157%/1,079%, khôngnewclipcollapse.
Flagcomponentcollapse chỉquestion-only nhánhthiếuRGB, khôngdiscoverycoreloss.
Không normalization/stateview/decoder/encodergrid mới. PeakRSS133,39MB/
commit150,06MB,47,60s; main/proposal30 giữnguyên. Representationgeneralization
chưađạt; learner mới cần chốtallowedsignal vàprotocol, không tự chọnprobe làmAI.

## Đối chứng visuallearnability33 đã kết thúc, chưa đạt — 2026-11-10

[Protocol/báo cáo33](research/33_VISUAL_WORD_LEARNABILITY_CONTROL.md) giữ48ALIF,
input/head/data/seeds32; hiddenBPTT chỉđốichứng côlập, khôngmain/proposal30.
Preflight`20261109T233806Z-visual-word-control-preflight-v1-805213`:3×64train
responses khớpfullstatebytes/sourcecalibration; graphstateerror3,82e−6,
featureerror5,96e−8. Timing64responses0,85s, forecast317s/round; peakRSS565MB/
commit986MB, cap2GB. Codec/optimizer/nextupdate/generation khớp. Chưa formal
qualityscore trongpreflight. Sau đó173tests qua trướcformaltraining.

Formalrun`20261109T234349Z-visual-word-learnability-control-v1-e51fe9`:3paired
seeds/1536experiences, meanmacrotrain48,43%/dev25,67%, baselinefrozen18,88%dev.
Gain+6,79đ (CI95%0,30..10,76), novel27,44%, shape5,21%; mọiqualitygate fail,
CONTROL_LEARNABILITY_NOT_DEMONSTRATED_IN_BUDGET. Khôngfittraingate65%.
3coldresumesexact;97,82s, treeRSS639,85MB/commit1025,50MB, cap2GB/process.
Hiddencontrol đổi thật nhưng khônglà main/locallearning. Stopcontrol, không
tune/input/epoch/scale hoặc importHermes. Main241363updates/sourceweights
giữnguyên. Localhost tựhiện864dev/96ảnh và actuallearning cónhãnCONTROL.
Tiếp theo ưu tiên readonlyinformation/rootcauseaudit theo master, chưa learner mới.

## JointRGB/question→neuralwords v1 chưa đạt — 2026-11-10

[Báo cáo32](research/32_JOINT_VISUAL_WORD_READOUT.md):9cấu hình/3freshseeds,
hiddenfrozen/localwordreadout,96train/96devscenes và864QA/split,720noveldevcases.
Fullmacrodev18,88% so noimage17,97%/noquestion21,50%; gainvsnoimageCI95%
−4,69..5,99đ, novelmacro16,94%. Causalimageswapdrop3,26đ, chưa20đgate. STOPv1,
không normalize/tuneLR/cachemixture/epochs/scale/mainimport saufailure.
Đã nốijointinputs và sinh từthật trongprototype, chưa learnedvisualgrounding;
khônghiddenlearning/normalvision/retentionclaim.9coldworkingstateresumesexact.

LỗiNaNemptydiagnosticgroup vàwrapperarg đãgiữlog/sửa reportingonly;2completed
models được táichấm khônghọc lại,7cấuhìnhthiếu trainmới. Mainprivateweights/
architecture/proposal30 chưađổi.170testsqua. RecoveryRSS197,39MB/commit150,34MB;
originalfailedattempt peakmemory khôngđođủ nên khôngkhẳngđịnhfullprogrampeak.
Đềxuất30 vềnguồntínhiệu vẫn chờquyếtđịnh rõ của người dùng, khôngsuyraapproval.

## Visionquestionpath đã kiểm thật — 2026-11-10

[Báo cáo31](research/31_VISION_QUESTION_PATH.md):AntigravityCLI sinh32câu hỏi,
đãđọcreview (3wordingcaveats).240actualGeneration.vision calls trênpublicclone,
12ownedrenderedscenes; cùngảnh/state/48neurons thì đổiquestion vẫn cùnginput/
probabilities, contextchannels luôn0. Câu hỏi bịbỏ trước neuralcore; hiện chỉ
10digitoutputs, chưa jointimagequestion→language. Colorloss6cặp là replication
của13, không discovery hoặc visionqualitygain. RGBprototype giữ khácinput,
chưa learnedgrounding. Không training/mainweightschange hoặc proposal30approval.
PeakRSS59,06MB/commit42,59MB,6,94s. Qwen4Btext-only không là visionreference;
không tạoảnh-as-text hoặc tải/bậtVL/9B. Trace cóảnhRGB/8×8gray/rawneuralJSON.
165testsqua, source/mathematics/protocol/240outputs/12PNGhashes đãkiểm;
UI đãquan sáttrực tiếp. Mainprivateweights/topology/vocab/encoding giữ nguyên,
counter241363, RAMlive≈65,0MB tại mẫu đo; mộtmodelID, khôngllama-server kèm.

## Observedassociation v1 chưa đạt — 2026-11-10

[Báo cáo29](research/29_OBSERVED_RESPONSE_ASSOCIATION.md):12trials/3freshseeds,
labelconditionedHebb chỉ contextinputsynapses, khôngbackward. Paired/frozen đều
2,15%dev; pairedprobe29,03% so frozen31,18%, gainCI95%−6,45..3,23đ. Thay đổi
7168transactions/model không tạoquality/representationbenefit. STOPv1; không
autoLR/encoder/epoch/normalization hoặcround2 tune saufailure, khôngmainimport.
Baselinecũngyếu, không suy ra mọiHebbianSNN bấtkhảthi.12coldresumesexact,
frozen=erase-trace/tick/sampler khớp. PeakRSS155,74MB/commit179,01MB,14,23s.

[Đề xuất30](research/30_PROPOSED_LEARNING_SIGNAL_BOUNDARY.md) đang chờ người dùng
chốt có mở nhánh readout-error→neuron một bước hay giữ ranhgiới tínhiệu cũ.
Chưa code/huấnluyện nhánh đó; main vẫnLIF/ALIF+locallearning, cấmTransformer/
pretrainedproxy/BPTTmain. Các công việc audit/UI/docs không phụ thuộc câu trả
lời vẫn thực hiện; chưa coi quyềnnghiêncứu là quyềnđổi allowederrortransport.

## Audit nhiễu tín hiệu học — 2026-11-10

[Báo cáo28](research/28_CREDIT_SIGNAL_NOISE.md):144conditionalinputs×128pairs,
9immutablecheckpoints/3pairedseeds, không training. Tạilocalcheckpoint, scoreEMA
cosinetái lập≈0,202 so surrogate≈0,999; crossfit/EMAnoiseenergyratio≈0,227,
CI95%0,104..0,382. Baselinecrossfit chỉvariancecontrol, haihalfmeans bịcoupled;
không coi cosinecrossfit là lặp độc lập hoặc kết luận qualitygain. Surrogate
biased, khônggroundtruthgradient. STOPword-scorev1/v2 giữ nguyên, khôngscorev3.
9sourcehash/144weight-learningstate-RNGchecks khớp; source/mathprotocolbấtbiến.
PeakresearchRSS594,03MB/commit994,04MB,159,05s; không mainweights/architecture
change hoặc teacherAPI mới. Representation/credit/retention/normalvision còn mở.
159testsqua gồm nativeparity; bảng9tínhiệu/CI đã kiểm trực tiếp trênlocalhost.
Privateweights/topology/vocabulary/encoding/counter241363 giữ nguyên; appmột
coreALIF, RAMlive khoảng57,8MB tại mẫu đo, khôngllama-server chạy kèm.

**Khóa định hướng theo yêu cầu người dùng:** giữ lõi Spiking Recurrent Neural
Network LIF/ALIF và local synaptic learning; cấm tự chuyển thành Transformer,
pretrainedLLM hoặc proxy. Qwen chỉ là đối chứng độc lập. Bản cập nhật trace
không sửa thuật toán neural hoặc thay trọng số đang phục vụ.

## Trace tự động và cập nhật kết quả — 2026-11-10

[Đối chứng27](research/27_ISOLATED_WORD_LEARNABILITY.md) đã xong6lượt/3pairedseeds:
ALIF48+dense+BPTTcontrol dev44,09% so frozen8,60% và localscore7,53%; train16
91,67% chưa là normalchat. Tín hiệu học có ảnh hưởng rõ trong bài8styles,
nhưng globalcontrol không đáp ứng localmainconstraint và có compute thêm.
Mainword-score vẫnSTOP, tổng thểPIVOT; không importweights/backward/Transformer.
6coldresumesexact,156testsqua; mainprivateweights/topology/vocab/encoding/counter
241363 giữ nguyên. PeakcontrolRSS623,37MB/commit1,015GB. Sự cốWindowsatomicwrite
đã tái hiện và sửa boundedretry; thời gianlỗi/sửa tính trong deadlinegốc.

47Qwenreference mới đã đo sau khiTorchdừng; đủ95prompt reference gồm41bịcắt64tokens,
không dùng làm nhãn. PeakQwen3,236GB báoriêng với4,5GBreferencecap, serverđãtắt.
Mục `#answer-archive` tự hiện đủ95case đãlưu và Qwenkhớpprompt, có nhãncontrol/
archivaltime. Không chạy model để tô dữ liệu hoặc mở lại family đãSTOP.

[Vòng word-score cuối26](research/26_FIXED_INPUT_NEURAL_TRANSFER.md) đã xong:
27trials/3freshseeds, sameH48 và rule. Densefull dev7,53% so densefrozen8,60%,
gain−1,08đ CI95%−3,23..3,23; priorfull9,72% so frozen13,89%. STOP, đóng hướng
word-score application sauv1/v2, không tune/scale/import.27/27coldresumes khớp,
152tests qua, peakresearchRSS163,45MB. Maincore và privateweights giữ nguyên.
Readonlyinputinterventionv1 thấy có sensitivity nhưng chưa devbenefit rõ;
không giải thích mọi thất bại là model bỏ qua câu hỏi. Goalnormalchat/vision
và P2retention vẫn chưa hoàn thành; ưu tiên learnability/creditassignmentcontrol.

Mở [nhật ký công việc](http://127.0.0.1:8765/#agent-work): tự hiện việc Codex
công bố, câu trả lời thật, loss/ΔW/guard và tiến độ. Không chọn model hoặc bấm
chạy. Nhật ký kỹ thuật tách khỏi neural trace. Acceptance thật:148tests qua,
inference21ticks, baGET giữ updatecounter241363,100logloss cũ còn đọc được,
RAM live56,6MB tại mẫu đo. Không huấn luyện mới hoặc thay liveweights.

Các lượt trước đã hoàn tất và được bổ sung vào lịch sử localhost:
[representation/search24](research/24_REPRESENTATION_DECODER_AND_SEARCH.md) và
[word-score/input25](research/25_WORD_SCORE_TRANSFER_PROPOSAL.md). Beamsearch
và chuyển paired-score sang từ chưa đạt; stateview và inputcontrol cũng STOP.
Không gọi cải thiện token teacherforcing hoặc inputclassifier là normalchat,
không mở lại các family đã dừng. Goal nói/nhìn bình thường vẫn chưa hoàn thành.

## Controller stop-loop đã kiểm thật — 2026-11-10

Hai ví dụ chào hỏi từ fixture dự án có 4 cập nhật được giữ và 60 rollback,
nhưng câu trước/sau không cải thiện. Controller đã sửa để dừng sau hai đề xuất
giống hệt về model state, prompt và target cùng bị guard từ chối. Kiểm tra live
đã quan sát 2 rejection rồi dừng; trọng số/counter giữ 241363. Không đổi guard
hoặc quy tắc synapse. Hậu kiểm 48 ảnh chữ số giữ 81,25%, không mất câu đúng cũ
hay hai bài toán đã học trong phép đo hẹp này.

140 kiểm thử qua, gồm native baseline parity. 48 câu công khai mới đã chạy
trên lõi live, ghép đúng prompt với Qwen đã lưu; counter không đổi. RAM live
khoảng 60 MB, Qwen reference đã tắt. Không có candidate curriculum nào được
nhập vào live, không bật optional surface normalization. Normalchat/vision và
retention rộng vẫn chưa đạt.

## Hội thoại thật và curriculum nhỏ — các candidate chưa đạt

[Báo cáo23](research/23_REAL_DIALOGUE_STATE_AND_CURRICULUM.md):Antigravity đã sinh
48diagnostic +160curriculum prompts; rà soát giữ117train/31dev. Qwen4B đã trả
lời48câu mới, serverđãtắt; referenceRAM3,2GB được ghi riêng, không gọi dưới2GB.

Stateaudit cho thấy resetclone không sửa được bảnpersonal; rehearsal phục hồi
câu quen nhưng cách nói mới còn kém.12curatedtrainingclones dùng dataAGY thật,
hiddenfrozen/readoutlocal, lựa chọn dev trước acceptance; tất cả chưađạt để
thay live. Optional terminalnormalizer chưa bật. Source-aware projectteaching
được tách riêng, không giảhumanfeedback hoặc importfailedweights.

Timescaleview60tổ hợp đãSTOP, khôngfreshtraining từscreen. P1/P2gate lịch sử
không đổi. Normalchat/vision, broadskill và lifelongretention vẫn chưa đạt.

## P2 baseline chưa đạt; cần xử lý representation/state trước protection

[Báo cáo21](research/21_P2_RETENTION_AND_CUE_INFORMATION.md):18lượt/3seed,20context
trộn dần, same5760updates. Full+reservoir final34,74%, acquisition57,64%,
forgettingTB29,09đ; mọi condition fail joint gate. Frozen hidden vẫn quên27–32đ,
head overwrite có bằng chứng; offlinejoint/refit cũng dưới65%, representation
hiện chưa đủ. Không mở importance tuning hoặc confirmation từ kết quả này.

Audit immutable checkpoints cho MC64/tick6 probe khoảng67–68%, endpoint chỉ
56–60%; inputproductcontrol100% không là neural skill. Chưa giải noise/state/
association. Maincore/P1historicalgates giữ nguyên, khôngscale/deploy.

## P1 paired-score qua gate hẹp — 2026-11-09

[Báo cáo20](research/20_SCORE_ELIGIBILITY_AND_VARIANCE.md):108development+120
confirmation trials, 8 seed xác nhận mới/task. Selected raw-independent Adam
0,002 dùng local pre/post score × scalar return, không backward. Gate P1 qua
trên order/XOR: gain18,33/20,00 điểm, lowerCI14,65/15,06; direct91,99/74,56%.
Binding direct49,56%, gain−1,29 điểm: thất bại. GO_P2_RESEARCH_ONLY, không scale
hoặc deploy. Retention/quy tắc mới, normalchat/vision và broad skill còn thiếu.

24/24 cold continuations khớp,129 tests qua. Peak nghiên cứuRSS165MB/commit188MB.
Reward đúng hơn shuffled trên2tasks, nhưng chưa chứng minh vượt mọi clipped
baseline. Paired sampling đối ngẫu không được chọn. Các family đãSTOP giữ nguyên.

## P1 audit optimizer/coverage — 2026-11-09

[Báo cáo19](research/19_CONTROL_OPTIMIZER_AND_COVERAGE.md):45isolatedcontrols,
3freshseeds, sameforward/head/data, factorialAdam/SGD×all48/rotating12.
Bindingmean: frozen65,10%; Adam-all82,42%; Adam-rot12:63,28%; SGD-all61,07%;
SGD-rot12:62,76%. Optimizer/coverage tương tác; không quy hết gap cho credit.
Chỉ thăm dò, Holm12contrasts chưađạt; không mở lại familylocal-classifier.

DIAGNOSTIC_ONLY, khôngnewlocalcandidate/deploy/scale. Loss/ΔW/tensorpredictions
tự lênlocalhost. Khoảng302s, peakRSS751MB/commit1,25GB. Liveupdatecounter
giữ241347; maincorekhôngBPTT.122testsqua, gồmnativebaselineparity.

## P1 local classifier — đóng family sau vòng cuối

[Báo cáo18](research/18_LOCAL_CLASSIFIER_FINAL_AND_LEARNABILITY.md):108development
trials,3freshseeds, conditioning/timing có ablation; selected hidden probe gain
order+2,60đ, XOR+3,78đ, binding+0,26đ. Không qua gate5đ trên2tasks, không mở
confirmation hoặc deploy. **NO_GO_LOCAL_CLASSIFIER_FAMILY**, không vòng3.

18isolatedBPTTcontrols cùngforward/head cho binding91,67% so với62,37%readout;
đây là learnability diagnostic, không đưa backward vào lõi chính.9/9readout
control predictions khớp,9/9cold-process continuation khớp. Peak nghiên cứu
RSS750MB/commit1,25GB. Livecore không đổi sang candidate. Normalize có ích
thăm dò choorder/XOR nhưng khôngbinding; timing không cải thiện nhất quán.

Lượt và learning/answer measurements đã tự lênlocalhost khi chạy thật.
Protocol/result lịch sử giữ nguyên. Mục tiêu hội thoại/vision, retention,
Experience-to-Skill vẫn chưa đạt; PIVOT theoP1/P2, khôngscale.

## Trace thụ động, inference và learning thật — 2026-11-09

Theo yêu cầu mới, bảng localhost **tự theo dõi**; đã bỏ chọn nguồn/bấm chạy khỏi
observatory. `#test-answers` tự hiện input thực, generated, target và archived
measured Qwen; `#learning-observer` hiện loss/ΔW/changed weights/guard thật.
Log cũ thiếu metric ghi thiếu. Không dùng mock model/teacher proxy để điền bảng.

Đã chạy [24 inference + một lời dạy số học](../artifacts/live-observatory-latest.json)
trên chính phiên đang dùng.17 accepted micro-updates,34 token steps,
loss7,155→1,144; last training attempt sinh15 cho7+8.0 rejection mới; không
giả loss cho93 rejection lịch sử. Peak RSS quan sát khoảng70,5MB trong lượt.
Targets development không được dùng làm nhãn; Qwen full-prefix và serving
latest-user-only được ghi rõ là khác điều kiện. Chưa language/retention success.

119 tests qua, gồm native baseline parity; UI đã thấy kết quả thật tự hiện
không bấm chạy. [Hướng dẫn](OBSERVING.md), [ADR0012](decisions/0012-passive-real-work-trace.md).
Sau đó idle review tự thêm2 accepted updates nữa (không bấm dạy lại), được
trace ghi loss0,9957 và0,7706. Đây là số đo các giao dịch tiếp theo, không sửa
kết quả lượt17updates ban đầu. Restart vẫn đọc lại25answers/19lossrecords.

## Quan sát neuron và công việc nghiên cứu — 2026-11-09

Website có **Neuron theo task / Đang thử nghiệm / Lõi đang trò chuyện** (nay
bổ sung hai mục trace thụ động ở trên).
Đã quan sát UI và chạy 5 task trên core live: 64 → 64 → 96 → 128 → 64 neuron;
128 đã cấp được giữ lại, nền có thể dùng 48. Có checkpoint trước lượt và nút
dừng; không thêm lesson hoặc câu hỏi giả. Bản sao public đã đo dãy
48 → 64 → 96 → 128 → 48. Đây là chọn theo tải, chưa learned difficulty hay ACT.

[Hướng dẫn](OBSERVING.md) giải thích trace, RAM, job state và lịch sử qua restart.
Lượt nghiên cứu mới tự đăng ký; trạng thái chạy xác minh PID/create-time, không
tự restart. Trace thí nghiệm tách khỏi model live, giữ cả kết luận thất bại.

[Local classifier adaptation](research/17_LOCAL_CLASSIFIER_ADAPTATION.md) đã chạy
54 development trials và hiện tiến độ/mẫu neuron trên trang. Chưa qua gate
representation; không mở confirmation/deploy hoặc nâng quality claim.

## Real RGB grounding diagnostic — 2026-11-09

[Báo cáo16](research/16_REAL_RGB_GROUNDING.md):640train/160heldoutRGBobjectảnh,
3freshseed×2head, same48ALIFhiddenfrozen. Raw18,96%/normalized20,42%accuracy,
blank10%, crossclassmismatchđãaudit10–10,83%. Chưađạtgate30%, khôngnormalVQA
haylearnedcaption. Originalroll1mismatchgiữ150/160sameclass; đượcauditadditive,
khôngretrain/thaygate. Datasetpinnedhash, nooriginaltest/privateinputs/redistribution.
Livevisionvẫndigits, khôngthayweights. ByteworkingfamilyđãSTOPsau2vòng.

## Byte codec và working-state — 2026-11-09

[Báo cáo15](research/15_BYTE_CODEC_AND_WORKING_MEMORY.md): UTF8codec fixed bỏOOV,
nhưng12localtrials vẫn0devutteranceexact; syntaxvalidity dohostmask, khôngIQ.
Isolated6Adam/BPTTcontrols cũng0exact, trainbyteaccuracychỉ26–28%trongbudget.
Không đủ quy mọi lỗi cho localrule hoặc nói SNNbất khả thi.

Audit6model/32freshpairs thấy inputdistinctioncollapse ở127forcedbytes; thử
heldactivation192byte trên9trials không tạo taskgain,0devexact. **STOPbytefamily
sau2vòng**, coldresumequa nhưngqualityfail. Không đổi modelHermes hoặcweights.
Visionảnhthật đang có protocol nhỏ riêng, không gọi RGBfeature là đãhiểuảnh.

## Event dialogue + Qwen đối chứng — 2026-11-09

[Báo cáo14](research/14_EVENT_DIALOGUE_AND_QWEN_REFERENCE.md): đã thật sự train
96Antigravity responses trên3freshseeds×4conditions, chronological input→một
ALIF48→word decoder. Dev exact0/24mọi condition; outputvẫn lặp/vô nghĩa, dừng
theo protocol, không deploy.30dev words không biểu diễn được bằng word dictionary
train-only. Không lấy classification95% để gọi normalchat. Cold resume12checkpoints
khớp; là bằng chứng persistence, không quality.

Qwen4B đã trả lời cùng24dialogueprefix, có câu có nghĩa hơn nhưng11bịcắt và có
lỗi tự nêu ngày/khả năng tra cứu. Server Qwen đã tắt; không proxy vàoSCAR-X.
Corpus curatedđã dùng **trong experiment**, chưa vào phiên cá nhân. Mục tiêu
chat/visionbình thường còn thiếu representation/codec/retention; goal chưa xong.

## P1/P2 input và learning audit — 2026-11-09

[Báo cáo13](research/13_INPUT_AND_LOCAL_LEARNING_AUDIT.md): encoder G1 mất thứ tự
ở128/128cặp last-write, grayscale mất màu. Prototype ordered/RGB events cùng
channel budget sửa các counterexample; v2không trùng256cặp mới, chưa serving.

Hai local-target rounds108development trials không qua representation gate:
**NO-GO nhóm ứng viên**, không mở confirmation/scale. Output-local normalization
được xác nhận8fresh seed trên order/XOR (95,53%/86,99%), hidden frozen. Tuy
nhiên20context retention fail; normalization làm forgetting tăng35,14pp. Đây
là readout acquisition gain, chưa hidden learning hoặc solved lifelong.

Antigravity40dialogue mới đã qua repair/editorialreview:194targettokens,
train20vi/12en,dev4vi/4en. Rawfalse/actionclaim labels giữ và không train. Không
tự nạp corpus hoặc candidate weights vào phiên người dùng; model live giữ nguyên.

## Tab Hỏi để học và ba chế độ — 2026-11-09

Đã cài disk plugin Hermes qua cửa desktop/backend sẵn có, route
`/scarx-learning`, nav **Hỏi để học**. Native GUI đã quan sát hai chiều: gửi
QA4+4 → nhận câu hỏi → trả lời8 → thẻ chuyển Đã trả lời, có acknowledgement và
nhãn kiểm bằng phép tính. Hai câu hỏi không liên quan còn giữ. History512tin,
reply theo source IDs, clientID dedup, feedback theo verifier/guard hiện có.
Main chat mới chỉ routing notice, không blocking inline clarify cho mode này.

98 SCAR-X tests (gồm4native parity),2 UI behavior tests và real Hermes plugin
registration/auth → SCAR-X HTTP acceptance qua. Screenshot QA giữ riêng trong
`local/hermes-learning-tab-qa.jpg`, không đóng gói private transcript.
[ADR0010](decisions/0010-learning-tab-and-cognitive-modes.md).

Người dùng đã làm rõ3chế độ cùng core: Fast Processing / Deep Thinking /
Autonomous Idle Learning. [Tài liệu12](research/12_ADAPTIVE_COMPUTE_AND_IDLE_COGNITION.md)
đối chiếu nguồn và nối vào P0–P5. ACT/halting và tự khám phá có verifier vẫn
Proposed, không được nói như đã chạy. Mục tiêu chat/vision bình thường chưa đạt;
giữ PIVOT, không mở rộng data/model để che acquisition fail.

## Tăng tải, retention và Qwen4B — 2026-11-09

[Báo cáo11](research/11_KNOWLEDGE_LOAD_RETENTION.md): đã thử ladder128bài tối đa,
5condition×3orderseed từ1public initialization. Các nhánh học dừng ở16bài theo
stop rule: paraphrase0%, chưa có useful acquisition. Full/readout/replay quên
oldskill trung bình25pp; guarded8,33pp sau audit, vẫn vượt3pp/5pp. Replay giữ
vài đáp án nguyên văn, chưa đạt kỹ năng. Cold restart khớp không chứng minh nhớ
ngày/tháng. Negative-answer decoder limitation được lưu ở run abort riêng.

Qwen3-4B-Instruct-2507 Q4_K_M đã chạy thật:160/160bài số học, đã tắt server.
SCAR-X yếu hơn rất rõ trong phép đối chiếu hẹp; không chứng nhận4B-equivalence.
QwenRSS peak2,924GB, SCAR-X diagnostic102,7MB; budgets không tương đương.
Không thay liveweights hoặc đổi cấu trúc. Người dùng đã làm rõ fast/deep/idle là ba chế độ cùng core;
master/code hiện vẫn một pool hidden, không phải ba model riêng.
Không dùng kết quả flatpool để kết luận cho kiến trúc phân tầng khác.

## Học xuyên hội thoại và dữ liệu Antigravity — 2026-11-09

Một learner giữ trạng thái xuyên lượt, nhận lời sửa/đáp án, cập nhật cục bộ
và idle review có guard. Lưu controller/lesson/questions/RNG cùng neural core;
rollback giữ object core tại chỗ. Không tự dùng output làm nhãn đúng. Guard hiện
chỉ bảo vệ bộ probe phát triển nhỏ, chưa chứng nhận retention.

[Kiểm tra mới](../runs/20261109T064249Z-continuous-dialogue-acceptance-ce1b2c/report.md):
9 lesson,15 microstep chấp nhận/116 từ chối, exact resume và neural/RNG continuation
khớp. Câu `1+1` vẫn sinh “Bạn có..” sau review. **Acquisition chưa đạt**; cập nhật
trọng số và giữ trạng thái không đồng nghĩa đã học kỹ năng. Không thay release
weights bằng candidate thí nghiệm. Những số đo cũ bên dưới giữ như lịch sử.

Antigravity CLI đã tạo240 ví dụ; validator giữ217 (178train/39development), trong
đó53 nhãn arithmetic kiểm độc lập.23 ví dụ cách ly. Bổ sung40 hội thoại/120response
và16 visual QA +2caption từ2 ảnh công khai. Nhãn style/ảnh vẫn là candidate;
train/dev hội thoại bị confound ngôn ngữ nên chưa dùng kết luận tổng quát hóa.
Không tự nạp corpus vào phiên cá nhân. [Báo cáo và giới hạn](research/10_DIALOGUE_DATA_AND_ONLINE_LEARNING.md).

Hermes có câu hỏi controller riêng và terminal event request-correlated;
không chờ LLM tool continuation.6 Python integration tests và5 UI tests qua;
GUI đã quan sát: trả lời6 cho câu QA3+3, thẻ settle; gửi tiếp hi nhận Hello.,
không còn thẻ bị khóa hoặc spinner. [Bằng chứng](../artifacts/hermes-online-resolved.jpg).
Footer là lượng nạp thực + neuron,
không là thước đo intelligence. General chat/normal photo QA vẫn chưa đạt.

## Capacity linh động và tải từng lượt — 2026-11-09

Theo phê duyệt mới,48không là trần sản phẩm. Prototype mộtcore48–128 có actual
selected-row/readout-column compute, lazy dormant decay, growthsnapshot và
topology/activecount/RNG/clock/ID persistence. Budget dựa inputvolume/modality
hoặc override; chưa có detector độ khó ngữ nghĩa. Newreadout0cần học; tăng neuron
không là tăng năng lực đã chứng minh. Foreground không tắt các cột readout đã
nối, chưa có causalimportance/pruning. [ADR0008](decisions/0008-adaptive-neurons-and-request-load.md).

Hermes lấy requestload theoID: inputchars/UTF8bytes/images, vectorchannels/bytes,
neuron đang tính/đã cấp/trần. Thanh chính hiển thị **Nạp X ký tự · N neuron**.
Memorycapacity và retainedskills vẫn chưa đo, không dùng load làm intelligence.

Đã mở explicitlesson với từ đầu ra mới≤256token và migrationđúng. Engineering
clone học được `1+1→2` sau32lesson, có exactresume; đây là trainingpoint, không
general toán và không âm thầm thêm vào trọng số phiên cá nhân.

[Vòng cải thiện3seed/9condition-runs](../runs/20261109T015554Z-generation1-improvement-fdd0ba/report.md)
đã kết thúc: lexicalencoderfull so originalfull **−3,125điểm** textconfirmation;
full so lexicalreadout **+6,25điểm**, nhưng accuracy tuyệt đối thấp. Chưa có
representation/qualitygain hữu ích; không phát hành candidate kém hơn, giữ
trainedweights G1cũ. Dataset mới32promptkhác18promptcũ, cùng8families, không
phải bộ đo generalchat hoặc lifelong. Giữ PIVOT và forgettinggate lịch sử.

Cần data mới cócurriculum/verifier; đã chuẩn bị
[brief cho Antigravity](../data/contracts/ANTIGRAVITY_DATA_BRIEF.md); việc tạo/kiểm tra
dataset đã hoàn thành theo cập nhật trên, chưa nạp corpus chưa xác minh vào phiên cá nhân. Có thêmdata chưa đủ
để cam kết sửa localcredit/representation.

## Hồ sơ triển khai trước prototypecapacity

## Hermes có trạng thái — 2026-11-09

Sửa Hermes theo yêu cầu người dùng; không sửa neural core/weights/topology G1.
Mode `stateful_chat` named-provider giữ buffer4096, bỏ floor64K dành cho LLM
agent, không tools/prompt injection/history replay/fallback. Lịch sử UI vẫn
append-only, backend giữ neural state riêng. Real AIAgent + HTTP + SessionDB
đã kiểm ở [nonstream](../artifacts/hermes-stateful-check.json) và
[stream](../artifacts/hermes-stateful-stream-check.json). Đã khởi động lại Hermes
và thử hai lượt trong cùng phiên desktop: không còn lỗi64K, lịch sử giữ nguyên,
`hi → Hello.`. Lượt `chào bạn` vẫn sinh câu sai/lặp; không nâng quality claim.
Footer đổi thành **Backend state**, popover phân biệt input2048ký tự và memory
capacity chưa đo.48 Python tests có liên quan và10 UI tests qua, không phải toàn
bộ suite Hermes. [Bằng chứng GUI](../artifacts/hermes-stateful-ui.jpg) và
[hướng dẫn](../integrations/hermes/README.md).

[Memory/learning measures](research/09_MEMORY_LEARNING_MEASURES.md) thay mục
tiêu độ dài context bằng đường cong retention, working-memory horizon,
acquisition,transfer và skill capacity có điều kiện. Đây là thiết kế phép đo,
chưa chạy benchmark mới, chưa có giá trị capacity/horizon được xác nhận.

## Hiện hành — Thế hệ đầu, 2026-11-09

Một core 48 ALIF chung cho text/ảnh/nền, một checkpoint và một model Hermes
`scarx-generation1`. Điểm vào `run_app.py` dùng `generation_server.py` và
`Generation/UnifiedCore`; các mục V0.1/V0.2 bên dưới là hồ sơ lịch sử.

Đã có neural idle loop, foreground priority, topology/spike/voltage/probe trace,
pause/speed, explicit local teaching và full-session restore/RNG/controller.
Không reset neuron trong runtime, không tự học weights lúc rảnh hoặc grow.
Trace không giả độc thoại nội tâm. Cả48neuron vẫn được tính mỗi tick.

Huấn luyện chung hai seed: greeting50%/50% (fail80% target), text33,33%/22,22%,
digits88,33%/89,44% diagnostic. Seed17 chọn bằng development. Các câu `hi`,
`chào`, `chào bạn` là training points trả lời đúng; không là bằng chứng general
chat. [Run](../runs/20261109T000625Z-generation1-joint-bd6c92/report.md) và
[audit protocol deviation](../runs/20261109T000625Z-generation1-joint-bd6c92/PROTOCOL_DEVIATION.md).

72 tests đã qua, gồm native parity và 13 test G1. Formal Phase2.2, useful hidden
representation, retention gate, general vision/actions và useful consolidation
vẫn chưa đạt. Nghiên cứu PIVOT_NO_SCALE; xem [master1.2](RESEARCH_PLAN.md).

[Engineering acceptance](../runs/20261109T005218Z-generation1-engineering-9111b3/report.md)
đo khoảng57MB, idle ticks tăng trong khi weight fingerprint bất biến, shared
image/text và exact continuation. Decoder dừng cuối câu thử trên development
giảm37,5%→25%, nên không bật. Gói `SCAR-X-Generation-1.zip` có kiểm manifest và
khởi động từ thư mục giải nén mới; không chứa localdata/Hermesconfig/secrets.

Hermes provider config và native model-picker API chỉ có `scarx-generation1`;
model overrides tắt nhãn reasoning/tool chưa có. Existing chat có thể cần phiên
mới hoặc `/model custom:scarx-local:scarx-generation1`. Qwen9B autostart vẫn tắt.
Không có plugin `ponytail` khả dụng trong phiên cập nhật; không giả lập công cụ.

## Hồ sơ các sản phẩm đã ngừng phục vụ

## Huấn luyện và V0.2 LAB — 2026-11-09

- PilotV2 đã kết thúc, không còn process training: languageexact3/16=18,75% (fail), digitclassification321/360=89,17% (diagnostic pass, không phải screenshotunderstanding).
- Decoderbeamfixed4 không cải thiện đủ; không tiếp tục tune trên test hoặc thay threshold để gọi languagepass.
- Tạo modelLAB opt-in riêng trong app/Hermes: `scarx-language-lab-v0.2` sinh text từ trainedwordhead, `scarx-digits-lab-v0.2` phân loại ảnh chữ số. V0.1không bị thay.
- Một lesson trực tiếp cho ví dụ `chào bạn` đã loại lặp trên trainingprompt bằng localupdates; metadata vẫn ghi languagequalitygatefalse. Không gọi đây là generalization.
- LAB có explicitteaching, vocabulary expansion≤256token/48hidden, SQLitecheckpoints và exactrestore/restart. Chưa có generalchat, ảnhgiao diện, OCR, agentactions hoặc learnedreasoning.
- [Pilot V2](../runs/20261108T190315Z-language-vision-pilot-v2-6a889e/report.md) và [release descriptor lịch sử](../artifacts/pilot-release.json) giữ checkpoint/giới hạn. Hướng dẫn hiện hành ở [README](../README.md); llama.cpp autostart vẫn tắt để tránh Qwen9B chạy kèm.

## Product-first — MVP V0.1 chạy được

Người dùng đã thay phê duyệt docs-only bằng scope triển khai MVP. Chạy `python run_app.py` và mở `http://127.0.0.1:8765`; [README](../README.md) là hướng dẫn dùng/cài/teach/restore duy nhất.

Đã có: UI localhost, PlasticCore48ALIF/864synapse thật, dạy nhãn theo verifier A/B, kiểm before/after và A→B, spikes/weightdrift, SQLitememory/history, atomicfullsessioncheckpoint/restore/restart và resource display. Windows processcommit cap2GB áp khi Job attach được. Không cloud/LLMpretrained, không đổi rule/topology hoặc tăng scale.

[Acceptance demo](../runs/20261108T175211Z-mvp-v01-acceptance-3d887d/report.md): A50%→99,2% sau hai đợt; sau hai đợtB, A89,8%/B74,2%; chưa có mẫu exact trong train cho tập chấm. Restart/restore exact được kiểm. Đây là product regression sau build iteration, không phải researchgeneralization claim. Scope còn yếu:2tác vụ có cấu trúc, chưa natural-language, chưa solvedforgetting, chưa customverifier/importUI hoặc consolidationnền.

Xem [ADR0005](decisions/0005-product-first-mvp.md). Các mục nghiên cứu bên dưới giữ kết quả và gate; **không còn là điều kiện chặn việc dùng/phát triển MVP**.

**Hermes backend:** `/v1/models` và `/v1/chat/completions` nonstream/SSE cho structuredJSON đã có, provider `scarx-local` được thêm vào configHermes và giữ nguyên Qwen9B đang chọn. Không MCP, không GGUF. Automaticcompute chọn4/16passes, chưa là learnedreasoning; vision/actions chưa được hỗ trợ. PackageZIP có source và checkpointdemo; xem README mục8 và capabilities.json.

Cập nhật **2026-11-09**. Định hướng: **Lifelong Adaptive Intelligence**. Mục tiêu 1M context và độ dài token đã được loại bỏ.

**Kế hoạch chính duy nhất:** [RESEARCH_PLAN.md](RESEARCH_PLAN.md). SCAR-X giữ mục tiêu neural architecture, không chuyển thành memory/RAG cho LLM. Bản1.0 chỉ cập nhật docs; ADR0005 nay phê duyệt MVP/code/UI. Protocol và kết quả nghiên cứu cũ vẫn giữ nguyên.

**Quyết định sau Phase 2.1: PIVOT_NO_SCALE.** Giữ mạng nhỏ 48 neuron. Chưa có bằng chứng đủ cho lợi thế representation và continual learning của các quy tắc đã thử.

## Kết quả hiện tại — Phase 2.1

- [Báo cáo kết quả và câu trả lời nghiên cứu](research/08_PHASE21_RESULTS.md)
- [Kế hoạch đăng ký trước](../experiments/phase21-v1.md) và [bổ sung stream không task boundary](../experiments/phase21-lifelong-amendment.md)
- [Phản biện khả năng học dài hạn](research/07_LIFELONG_LIMITS.md)
- [Audit hiệu lực phép đo](decisions/0003-phase21-evaluation-audit.md)
- [Danh mục run](../runs/INDEX.md), [kết quả máy đọc](../artifacts/phase21-summary.json)

Đã audit 15 trial/checkpoint Phase 2 và chạy 303 trial mới: eligibility score, STDP, metaplasticity, importance decay, replay/consolidation factorial, readout-only, SI tại readout, modulation nhiễu và BPTT control riêng. Có 3 seed phát triển, 8 seed xác nhận và 8 seed mới cho strict binding audit; không tính các condition ghép cặp như seed độc lập.

Stream có 1.280 trải nghiệm, không task ID/boundary cho learner, 20 cửa sổ đánh giá và 20% nhãn nhiễu. Fast weights, stable anchor và consolidation định kỳ đã triển khai. Neural/RNG/importance/replay continuation được kiểm tra.

Kết quả: hidden chưa đạt gate representation; metaplasticity giảm forgetting nhưng accuracy gần 50%; không condition nào đạt đồng thời mean forgetting ≤3 điểm phần trăm, worst ≤5 điểm và chất lượng tối thiểu. RAM quan sát trong trần 2 GB, chưa phải OS hard cap.

Hướng còn hợp lý: importance gắn với mức hữu ích, ổn định readout, credit assignment và bài representation hợp lệ. Cần protocol/seed mới trước vòng tiếp theo. Không tăng tham số, huấn luyện ngôn ngữ hoặc tối ưu C++/iGPU trong vòng này.

## Roadmap nghiên cứu Phase 2.2 — không chặn MVP

1. **P0 kỹ thuật:** compatibility `plasticity-v1`, whole-state checkpoint/rollback cho learner/replay/RNG, evidence verification và budget RAM toàn phiên. Save/load riêng đã có; transaction cho lõi mới chưa hoàn chỉnh.
2. **P1/P2 ưu tiên khoa học:** representation/credit assignment và readout stability/usefulness-based importance. Không dùng replay tối ưu để che các vấn đề gốc.
3. **P3:** Experience-to-Skill với tình huống giữ kín, reset/restart và retrieval-off; semantic/procedural memory vẫn là mục tiêu chưa chứng minh.
4. **P4:** offline consolidation thủ công có matched compute, verifier/guard và rollback; chưa mở automatic idle scheduler.
5. **P5:** độ bền20→50→100giai đoạn có gate, giữ capacity và plasticity reserve.

Chi tiết giả thuyết, code, dữ liệu, baseline, metrics, pass/fail, rủi ro và phụ thuộc ở [roadmap master](RESEARCH_PLAN.md#7-experiment-roadmap). Tất cả stage trên cần duyệt triển khai/protocol trước. Ngưỡng mean forgetting≤3 điểm phần trăm và worst≤5 trên từng seed vẫn giữ; kết quả cũ không chấm lại.

Chưa có Git trong project; [master mục8.3](RESEARCH_PLAN.md#83-quản-lý-phiên-bản-và-dữ-liệu) chỉ đề xuất phương án version control local, chưa init/commit/push hoặc đổi cấu trúc. [ADR0004](decisions/0004-master-plan-and-phase22-docs.md) ghi phạm vi quyết định mới.

## Hồ sơ Phase 2 lịch sử

- Cấu trúc src theo domain, project map, ADR, protocol và run index.
- ALIF recurrent, block-local reward-modulated STDP, readout local, stable weights.
- Checkpoint có checksum, atomic save, exact resume; guard và rollback.
- SQLite exact episodes, FTS5, provenance; byte generation plumbing gắn nhãn chưa huấn luyện.
- CLI init/generate/observe/search/learn/rollback/benchmark.
- C++ CPU SIMD kernel, build bằng MSVC, kiểm tra parity.
- 15 trial local/analog qua3 seed,20 task/chặng; run xác nhận cùng seed.
- 3 trial surrogate BPTT control riêng; runtime/memory/retrieval smoke.
- Sáu báo cáo nghiên cứu trong research/.

## Kết quả Phase 2 cần nhớ

Học được liên kết nhỏ, nhưng readout-only cũng đạt100%. Hidden plasticity chưa tạo lợi ích đo được ở bài này. Forgetting trung bình full20,8–23,9 điểm %, cao hơn ngưỡng3 điểm. Vì vậy dừng mở rộng.

## Chưa thực hiện — giữ rõ trong backlog

- Huấn luyện ngôn ngữ10M token và model10M/30M/100M tham số.
- Bộ đánh giá đa năng600+ bài và matched commercial comparison.
- Low-bit/pretrained LLM+RAG baseline trên cùng tác vụ ngôn ngữ.
- Runtime C++ độc lập, hard memory cap, quantization, iGPU scheduling.
- Kiểm thử feedback noise thực tế ngoài giả lập Phase 2.1, 8h endurance, điện năng đo thật.

Lý do: gate nghiên cứu bắt buộc đã không đạt; không được diễn giải những mục này là hoàn thành.

Tra cứu: [Phase 2.1](research/08_PHASE21_RESULTS.md), [Phase 2 lịch sử](research/06_FEASIBILITY_VERDICT.md), [run index](../runs/INDEX.md), [bản đồ](PROJECT_MAP.md).
