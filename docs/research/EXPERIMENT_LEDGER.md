# Sổ thí nghiệm và kết quả SCAR-X

Ngày hợp nhất13/11/2026. [Sơ đồ](EXPERIMENT_MAP.md) · [Tra theo mã](experiment-catalog.json)
· [Bản gốc có hash](archive/README.md). Cleanup không huấn luyện lại và không
đổi scientific gates. Phạm vi kiểm từng nguồn được ghi riêng; không chấp nhận
PASS chỉ từ tiêu đề của một agent.

## Foundation và G1

[Reports01–44](experiment-catalog.json) vẫn ở đường dẫn gốc `docs/research/NN_*.md`.
Các số bên dưới thuộc đúng bộ đo trong report, không so chéo generator/forward
khác nhau như một scaling law.

| Mốc | Đã thử | Kết quả và cách đọc |
|---|---|---|
| [01–06](06_FEASIBILITY_VERDICT.md) | Literature, ALIF/STDP, RAM/runtime, associative/recall/stream | Prototype có update/persist; recall100% không chứng minh representation/general intelligence |
| [Phase2 audit](08_PHASE21_RESULTS.md) | Kiểm15trial, head/hidden restoration, coverage replay | Full acquisition45,36%, final34,17%; max-to-final forgetting22,29đ. Khôi phục head cũ cứu task0 nhiều hơn hidden cũ: readoutoverwrite là nguyên nhân nổi bật ở scope đó |
| [Phase2.1](08_PHASE21_RESULTS.md) | Eligibility, modulation, metaplasticity, replay/consolidation, strictbinding, BPTT control riêng | 303trial; chưa continual advantage đáng tin. Strictbinding direct50,20%, probe49,66%; initialbinding có shortcut75% bị loại khỏi claim. Giữ acquisition khi so forgetting, không chọn model chỉ vì ít học nên ít quên |
| [09–12](12_ADAPTIVE_COMPUTE_AND_IDLE_COGNITION.md) | Memory/knowledge load/Qwen4B reference, online dialogue và idle processing | Có prototype/controller/trace; SSD retrieval không đồng nghĩa synapse học skill; chưa Qwen-equivalence, không suy activity thành lời nghĩ |
| [13–22](21_P2_RETENTION_AND_CUE_INFORMATION.md) | Inputinformation, localclassifier, scoreeligibility, optimizer/coverage, retention/cue | Phân biệt đủ thông tin đầu vào, credit đúng hướng, head đọc được; không gộp probe với direct, no hidden success từ weight movement |
| [23–29](29_OBSERVED_RESPONSE_ASSOCIATION.md) | Real dialogue state/curriculum, word transfer, observed-response association | Nhiều STOP/pilot yếu; phép so learnability/encoder/readout còn giới hạn, chưa sinh ngôn ngữ bình thường tổng quát |
| [30](30_PROPOSED_LEARNING_SIGNAL_BOUNDARY.md) | Chốt đường `Wout.T@error` một bước + own-neuron eligibility | Research exception được duyệt; không quyền whole-network/BPTT hoặc pretrained proxy |
| [31–34](34_VISUAL_INFORMATION_PATH.md) | Image-question information, RGB-word bridge, learnability controls | Có chẩn đoán ảnh/câu hỏi; không claim screenshot/OCR/robot hoặc vision đa năng |
| [35–37](35_DIRECT_READOUT_ELIGIBILITY.md) | Direct eligibility, wordbridge và visualbridge cùng controls | Có code/data/raw và learner local thực; vẫn pilot/research-only, không nhập vào Hermes từ các lượt này |
| [38–40](40_DEEPSEEK_REVIEW.md) | Objective/credit alignment và review parameter/guard/rollback | Audit cục bộ không tự là improvement chất lượng; lỗi norm/result được đánh dấu không dùng |
| [41–43](43_ADAPTIVE_DATA_GROUNDING.md) | Curriculum/data grounding, Qwen đối chiếu độc lập, usefulness P0 | Chỉdata/reference/P0 được duyệt tương ứng; không biến teacher output thành truth label. P0parity/rollback/normchecks qua không nghĩa quality/retentionPASS |
| [44](44_P1_HANDOFF_EVIDENCE.md) và [codec P0](OUTPUT_CODEC_P0_HANDOFF_V1.md) | Gói bằng chứng, fixedoutputvocabulary24→55 dưới256 | P0technicalPASS; chưa trainingdataset/newP1/P2/productionrelease |

## Các lượt đã kiểm gần nhất

### R8 repaired local credit và confirmation

[DEV](R8_REPAIR_AND_DEV_RESULTS_20261112.md) · [12seed confirmation](R8_P1_CONFIRMATION_20261112.md)
· [snapshot](forensic-r6-20261111/R8_SNAPSHOT_REPAIR_HANDOFF_V1.md).

Pin equations `bc6865003cc63dd846b4321d5050a0efbe27ef24`, peer
`607ffca7c9222c5e3fd9741c3ebdf5939e10bedf`. Own-reset giữ hai thành phần reset
trong voltageeligibility; stopped bỏ chúng ở trace bước sau. Neural forward
ALIF giữ nguyên. Control matching, inference/head và source/RNG được sửa trước
confirmation; không coi không-BPTT là local hoàn toàn nếu learning signal có
one-step readout feedback.

| Task | Own direct | Stopped direct | Stopped probe | Kết luận12seed |
|---|---:|---:|---:|---|
| Strict binding |82,55%|76,69%|82,19%|Fail OOD protection |
| XOR |97,18%|99,28%|99,71%|Đạt đầy đủ gates trong bộ đo |
| Distractor binding |66,28%|63,64%|66,68%|Fail direct65%/OOD |
| Order protected |99,85%|99,98%|—|Protected, không dùng ceiling làm hiddenproof |

12freshseeds,204jobs/408conditions,417.792predictionrows và50statisticalscalar
checks đã đối chiếu. Kết luận `P1_FAIL_RESEARCH_ONLY`, vì cần≥2/3primarytasks
đạt toàn bộ gates. Hidden gains đã có bằng chứng trên ba eventtasks, nhưng
không có universal learner hoặc continual retentionPASS. Không reuse bank
đã mở để tune/đổi ngưỡng.

### ACR — trộn own và stopped eligibility

[Full results](ACR_DEV_RESULTS_20261112.md) · [independent audit](ACR_STATISTICAL_HARNESS_REVIEW_20261112.md)
· [router trace](ACR_ROUTER_ANALYSIS_20261112.md) · [cơ chế giới hạn](ACR_CREDIT_SPACE_LIMITS_20261112.md).

345jobs,375modelrecords,3DEVseeds,3.072.000predictionentries được chấm lại;
30.720actualnormpairedsteps,0invalid,144diagnostic tapes/36stressfiles. Source
`c54ee689126a79e695df98eb81eb60053acc2621`, final archival branch tip
`22bb9781f4ed8f60f7180bdb56b6570e64e1f503`. Các quãng mất process/reuse203cells
và128..1.024lostupdates được ghi, không tính thêm thành independent evidence.

| Global profile | Binding direct/probe | XOR direct/probe | Distractor direct/probe |
|---|---:|---:|---:|
| Own alpha0 |85,42/86,98%|91,86/96,68%|56,97/56,97%|
| Stopped alpha1 |82,62/85,29%|99,35/99,54%|55,66/63,41%|
| Groups-up nominee |85,03/88,54%|98,11/98,44%|59,24/60,48%|
| State heuristic |80,47/84,18%|99,09/99,54%|59,24/59,77%|
| State-shuffle control |88,28/89,39%|96,61/98,31%|59,83/61,52%|

Endpoint0/1 khớp exact baseline qua1.024updates và checkpoint. Nominee chọn
**một profile toàn cục**, không alpha theo tên task. Normmargin2/3 và
shufflemargin3/3 có tín hiệu; directDistractor<65% và jointguards vẫn fail.
`ACR_SPECIALIZED_GAIN`, **không ADVANCE**. Statealpha biến thiên thật ở192frozen
rollouts nhưng không chứng minh policy học hoặc suy nghĩ bằng lời. State-shuffle
thắng state trên Binding là phản biện cần giữ, không chọn control hậu nghiệm
làm winner. Mỗiisolatedtask fit model riêng; jointIID chưa là lifelongP2.

### Literature-first

[132job results](LITERATURE_REFERENCE_RESULTS_20261112.md), [e-prop](literature/eprop.md),
[BrainTrace](literature/braintrace.md), [trained oracle](literature/trained-credit-oracle.md),
[reuse matrix/SI/EWC/DGN/DECOLLE](literature/continual-and-local-signals.md).

| Method | Binding direct | XOR direct | Distractor direct |
|---|---:|---:|---:|
| Own |85,42%|91,86%|56,97%|
| E-prop matched adapter |78,58%|98,24%|60,03%|
| E-prop readout-aligned adapter |69,92%|99,54%|55,01%|
| BrainTrace block adapter |68,62%|99,22%|53,32%|

675.840predictionentries,132checkpoints, manualforward/SVD/eigenprobe kiểm riêng;
`REFERENCE_MIXED`, không formalP1/P2. Đây là transformedNumPyadapters cho
SCAR-X postreset/adaptation và sourceoptimizer, không execution upstreamcompiler
hoặc accuracyreproduction paper. Fullsurrogate oracle chỉ frozencontrol, không
candidateRTRL/BPTT. Alignment ban đầu gần1 không bảo đảm directquality;180trained
diagnosticrows còn thiếu cross-neuroncredit. Initial/trainedfixtures khác nhau,
không quy cosine giảm chỉ cho learning. SI mới helpertoán, chưa retention.

License theo pin thực: e-propBSD3, BrainTraceApache2; DECOLLEcurrentpin có
noncommercialreservation trái shorthand GPL ở một số report; không copy
code đó vào sản phẩm. DGN đổi conductance/dynamics, không tự gán guarantee của
DGN cho inputgate trong ALIF giữ nguyên.

### Capacity H64/H128

[Results](CAPACITY_H64_H128_RESULTS_20261112.md) · [audit](CAPACITY_RESULTS_INDEPENDENT_AUDIT_20261112.md).
204newjobs+48H48reuse=252records;1.290.240predictionentries,168contrasts và
192copyfilehashes được kiểm. BaDEVseeds; primary1.024updates, equalCPU0,5s riêng.

| H | Own Binding | Own XOR | Own Distractor | Groups-up Distractor |
|---:|---:|---:|---:|---:|
|48|85,42%|91,86%|56,97%|59,24%|
|64|81,77%|95,96%|66,67%|65,04%|
|128|87,30%|99,41%|69,40%|67,77%|

H128 có ích và có hiddencausaleffect, nhưng readout-only cũng tốt lên; EXTRA
learned-vsfrozen advantage chỉ đạtgrowthmargin ở1/3family. `CAPACITY_ONLY_SIGNAL`
không nghĩa không có hiddenlearning. Contrastgrowth còn có ceilingXOR khi
frozenprobeH128 đã91,34%, không dùng gatefail đó để phủ nhận capacitygain.
Giữ nguyên tiêu chí đã đăng ký, không đổi nhãn thành PASS hậu nghiệm.
EqualCPU khôngmonotonic; cùngfan8 làmdensity8/H đổi, gamma resample và aggregate
readoutvariance/norm cũng đổi. Không purecapacitylaw. Fit từngtask riêng;
chưa commoncheckpointretention/chat/vision. PeakRSS148,82MB/commit191,15MB,
worker996,40s; không suy performanceđa năng chỉ từ RAMthấp.

## Các nhánh tác giả

Tất cả sourceSHA/protocol/report/metricpaths trong [catalog](experiment-catalog.json).
Các bảng này phân biệt source/arithmeticreview với full independent rawaudit;
không nâng selfverdict thành productionapproval.

### Grok và Antigravity

| ID | Câu hỏi/đã thử | Kết quả cuối được giữ |
|---|---|---|
| GROK-R1 | Sloweligibility đọc delayedorder | Corrected71005 meanlearned−frozen+2,734đ, lower95−0,289đ<5đ;zero=frozen. NO-GO. Report cũ71004 vẫn được giữ như superseded |
| GROK-R2 | Dendritictransfer từDC-ALIF | Transfer không đủ, pairbindingchance/orderdamage; STOP |
| GROK-R3 | Temporalstress, code có/không đọc được | Diagnostic; không suy missingcode từ một head thấp hoặc leakage từ mộtshuffle cao |
| GROK-R4 | Onlinehead/conditioning/clipping | Chưa đủGO, ridge/head khác nhau phải tách |
| GROK-R5 | Cue/querytiming và safenormalization | Có correction trong shifteddata; headchange còn regression |
| GROK-R6 | ReproducecaseA, unclippedhead | CaseAdirect≈66,76% vsridge≈98,58%;no-clipCI bao0/order−18,75đ, STOP |
| AGY-MTB | Multi-timescale readout | PIVOT: informationaccess không chứng minh hidden cần học |
| AGY-DCALIF | Coincidence/dendriticgate | ScopedPIVOT; neuronvariant chưa justify universaltransfer |

Bản gốc [Grok](archive/grok/README.md), [Antigravity/Gemini](archive/gemini/README.md),
Codex [round1 reviews](collab/scarx-neural-round1-20261111/orchestrator/ROUND1_REVIEW_STATUS_20261111.md).
DeepSeek/Antigravity có disclosure dùng chung phiên ở giai đoạn đầu; không tính
sự đồng thuận đó như independentconfirmation.

### DeepSeek R1–R6D

| ID | Kết quả và điều cần tránh |
|---|---|
| R1 | Pairbindingscore không đủ;_sampler/hardspike/nonlinearity/headconditioning_ bị sửa claim qua review |
| R2 | Tác giả scopedGO cho câu hỏi readout/substrate, không wholeP1/P2 |
| R3 | Screenpass rồi confirmationFAIL, STOP |
| R4 | Relationalfeature/readout có information; temporalshift confirmationFAIL |
| R5 | Independentkey/value screen9/9, confirmationcaseAFAIL; abortedruns được ghi |
| R6 | Scaling/data hypotheses, stress/delayedtransitionclaim bịR6D rút |
| R6D | Sixrealdefects,selfaudit/forensic; correcteddataeffect≈23,05đ nhưngdistractor55,02% failgate; stressclaimwithdrawn |

[Full originals](archive/deepseek/README.md), [forensicR6D](forensic-r6-20261111/R6D_SOURCE_AUDIT.md),
[R5B/R6B reproduction](forensic-r6-20261111/R5B_R6B_REPRODUCTION.md).

### DeepSeek R7 — các vòng1–43 và chuỗi sửa sai

Tất cả41subround reportfiles hiện có trong snapshotc2742540 được catalog theo ID
`DEEPSEEK-R7-N`. Không tự tạo report cho một số vòng không có file. Đây là
roundreports, không41independentseeds hoặc41confirmationbanks.

| Round | Cơ chế/khám phá | Kết quả/disclosure cần đọc cùng |
|---|---|---|
|1–3|Benchmark/trace/hidden-localization|Có testbed sai phải sửa, không áp verdict trước sửa sang bản mới |
|4 và16|Replay/readout interference|Kết luận replay của4 bị16 retract vì WRONG LABELS; sửa xong replay giảmforgetting nhưng vẫn failgate |
|5/5b/6|Orthogonal/nonlinearhead và averaging|ClaimmeanF2,86đ bị sửa vì tính cả termzero; mọi arm vẫn fail3đ ởmetricđúng |
|7–11|Protection, exacttrace và direction|Shuffle có thể thắngprotection; exacttracealignment≈0,945 chưa thànhrepresentationgain;instrumentở9 từngbroken |
|12–15|Geometry/deadunits/threshold|“25,7/48dead” bị14 retract thành≈1,7;13VOID;thresholdgain ở15 có benchmarkartefact |
|17–20|Replaybudget/orders/uncertainty|MeanF7,68đ/worst16,02đ ở17 vẫn fail; seedoutlierclaim18 bị19void;18pairedcells ở20 không là18independentseeds |
|21–26|Acquisitioncost và verdictlogic|21 được22retract;24gate8/15 bị26rút do signedforgetting;correctedmean≈6,84đ,0/15gate |
|27–31|Unreachableclaim/headfunctionclass|27/28 bị29retract qua nonlinearheadcontrol; không suy architectureceiling từ một head |
|32–36|T=6taskscale/replay/rank|32–35 giải thích một benchmark cóidenticalinputs/conflictinglabels, bị36retract; giữ cả code/data cũ với VOIDstatus |
|37|CorrectedT=6|MeanF3,84đ,worst9,12đ,gateFAIL;không dùng27,57đ từtaskcũsai |
|38–41|Aging/order/replayallocation|38mean3,39đ/worst8,54đ cònfail;39falsifierfires/verdictwithdrawn;41không hết tradeoff |
|43|DECOLLE-style signal vs scalar/frozen/shuffle|Probe+3,69đ[1,38;6,00] tác giảtáitính từ10/15stdoutseeds sau summarycrash; perneuron−scalar+1,52đCI bao0. DEV/reconstructed, không mainverifiedP1 |

Round43 tự ghi không có artifact gốc đầy đủ, mất năm seed đầu trong captured
tail và epochbudget120→30 sau đo tốc độ. Không thể coi câu “first positive in
43rounds” thành bằng chứng đã giảiSCAR-X; cũng không suy scalar chắc chắn
không còn là bottleneck chỉ vì một transferredsignal chưa thắng. Các report
đề xuất BF/metaplasticity không là quyền chạy P2/đổi learner mới.

### Gemini R6–R13 và R13-family đang chạy (ảnh ban đầu R13.1)

| ID | Điều có thể tận dụng | Giới hạn/quyết định |
|---|---|---|
| R6 | Code/raw localassociation,erratum | Unsupported/fabricatedargumentclaims bịretract; dùngconfirmation/sourceđúng |
| R7 | Representation/replay vàcross-SNNsurvey | Scopepartial, chưa fullP1/retention |
| R8 | Sensoryencoding/gate/integrityrepairs, peer607ffca | Codexmatched/confirmation là lượt riêng; không gộp selfPASS |
| R9 | Factory/controlmatching và gatebookkeeping | REQUEST_CHANGES;stalescreencache/effectivesignatures/confirmationselection;finalFAIL |
| R9b/c/d | Snapshot/normalization/causalwaiver/gategradient/literature | Nhiều sửa có ích cho harness, chưađạtwholegate;3seed≠72rows;license/neurondynamics phải đọc pinthực |
| R10 | Tác giả báo independentreplication/paradox/dampedreset | Source/summary được lưu, authorPASS chưa được cleanup tái chạy thành fullindependentaudit |
| R11 | SharedIID4096SGD trênmộtcore | Directmean82,49/86,33/75,39/88,41%; taskprobeheadsfittedriêng. SharedIIDkhông continualretention; không chứng minhngônngữ |
| R12 | Reservoir16/64, equalupdate,replay/SSDepisode storage | Replaygain có ích nhưng max-to-final forgetting vàworstseed đều vượt gate;precalibfuturetasks là scopeviolation đã erratum |
| R13 cũf4272ba | Source/data để hiểu modularcompositionhypothesis | Symbolicbridge/twocores/invalidcontrol/checkpointpeakclaims chưahợp lệ; archive-only, không importlearner |
| **R13.1** | **ĐANG CHẠY — người dùng bảo vệ** | **Không dọn worktree/branch/file/cache/process; không ghi verdict hoặc kết quả mới trong sổ này** |

R12 arithmetic đọc lại từ phase_snapshots, không training: Order1 baseline
meanF37,13đ; equalupdate39,56đ;reservoir16=30,03đ;reservoir64=22,37đ/final78,01%.
Order2reservoir64=13,45đ/final80,70%. WorstseedworstF33,20/34,38đ≫5đ.
Interleavedforgetting phải `NOT_MEASURED`, không0; originalfullstudy dùng
futuretaskunlabelledcalibration chưa fullstrict-online rerun. Đọc
[erratum](archive/gemini/original/docs/research/R12_ERRATUM_AND_METRIC_REAUDIT.md).

R13legacy sourcebridge `Aargmax→patterns_y[pred_y]→B`, không raw learnedhidden
spikeinterface;controlI hardcoded0,125;controlJ thiếu forwardtrain_x. Engram
chưa giữ mọi neuron/eligibility/RNGstate nên reloadaccuracy không chứng minh
mid-statecheckpoint exact; memory_info cuối không phải peak. Đây là source
review trên snapshotf4272ba, **không đánh giá các sửa đang làm củaR13.1**.

## Được đưa vào main lần này là gì?

- Hợp nhất nhánh research Codex đã kiểm: audits, isolatedprototypes, tests,
  protocol/raw/checkpoint và compatibilitycodec. Research modules vẫn nằm trong
  evaluation, không import vào serving hoặc thayproductionweights.
- 227originalreports/protocol/metricreceipts của peer và immutablebranch/worktree
  tags, để tra cả falseclaims/aborts/retractions. Peer code chưa kiểm nằm trong
  archivedGitobject/tag, không được auto-promote vì “merge cho gọn”.
- Catalog120entries là navigation, **không120independentexperiments**. Kết quả
  đã đo và hypotheses/chỉđềxuất được phân biệt.
- Source sản phẩm `src/scarx/neural`,app/learning/integrations vàrun_app được
  đối chiếu với maintrướccleanup để không làm đổi model đang dùng.

Không tạo training mới, không nới gates, không P2/freshconfirmation từ cleanup.
Các PR research đã chấm/đã lưu có thể đóng như archived, không đánh dấu đã
mergelearner. R13.1 và các sourceworktrees liên quan là ngoại lệ bảo vệ.
