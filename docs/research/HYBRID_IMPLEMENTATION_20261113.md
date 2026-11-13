# Hybrid implementation —13/11/2026

**Verdict: engineering MVP / PIVOT chất lượng. Không scientificPASS.** Người dùng
duyệt hybrid, sau đó tăng3GB→4GB, yêu cầu model trong chatHermes thường và bỏ
toolbarVideo/Screen. TabSCAR-X chỉ debug. Nhánhmain gốc và activeGemini giữ nguyên.

## Đã triển khai và đo được

- VLM frozenQwen3.5-0.8B nativeBF16/F32, ASRWhisper-base directFP32, localALIFH128.
  Không quantization/pruning/LoRA/autograd trong runtime. Storageaudit4deffd kiểm
  sourceSHA và453GGUF tensors chỉBF16/F32;245ASR tensors readbackexactFP32.
- Original-reference8496c6 đọc lại tất cả tensors nguồn dùng trong reference;
  phải restore36tensorFP32 bị`dtype=auto` ép xuống. Các lần referencefail được giữ.
  Ba ca next-token parity với runtime dùng top1same/Δprob≤.03. Đây là parity
  kỹ thuật hẹp, không chứng minh chất lượng toàn bộ VLM/video.
- Một gateway8775, streamed OpenAI toolcalls/Hermes history giữ nguyên, diskplugin
  chuẩn của Hermes. Một exceptionfloor đúngroute16,384tokens và sử dụng bridge
  tìm công cụ có sẵn; không khai64K giả hay bỏ toolresults.
- NativeDesktop đã chọnSCAR-X trong modelselector, gửi “Chào bạn!” và nhận:
  “Chào! Tôi là Hermes Agent, hỗ trợ các công cụ và dịch vụ trên nền tảng này.
  Bạn có cần gì không?” Traceactualc33f6ba9… ghi2368inputtokens,28outputtokens,
  TTFT14.515s,total16.343s. Chưa đạt TTFT5s. Bản cũ13,182tokensTTFT224.437s
  được giữ, không thay số cũ bằng số mới.
- Toolengineering734e30: model tự phát`lookup_number({key:ALPHA})`, nhận kết quả
  fixture17 rồi trả lời đúng. Đây là tool-protocol fixture, không đủ kết luận
  model gọi mọi công cụHermes một cách tin cậy.
- Screencaptureb84a70: đúng cửa sổ syntheticfixture được chỉ định, actualWGC→VLM,
  dừngcapture thành công, không microphone/audio. PeakconservativeRAM3.721GB
  gồmHermes/sharedGPU. Chưa là test audio-loopback E2E hay video tự nhiên.
- Checkpoint giữ trọng số/stable/state/eligibility/optimizer/importance/RNG/
  replay/controller/questions/probes; coldnextupdate/parity/rollback/DBfailure
  có regressiontests. Audit/evidence không biến mất khi rollback.
- Publicweb1bfb3c: đọc được nguồnWikipedia công khai theo topic được phép;
  nguồn vẫn có statusobserved, không trở thành nhãn đúng hoặc auto neuralupdate.

## Các lỗi còn nguyên trong kết quả

| Receipt | Kết quả và phạm vi |
|---|---|
| P0`9a3745` |3GBFAIL_RESOURCE, dư threadpoolcommit; không xóa receipt |
| P0`25a284` |Giới hạnBLAS1thread; CPUchatVi/En, shapes vision và EnglishASR chạy; peakRSS2.462GB trước tínhHermes |
| Runtimeguard trước schedulingrepair |Peakconservative4.039GB; dừngworkers. Không gọi đây là4GBPASS |
| Livevideo`fd5931` |21s publicFLEURSVi speech + syntheticshapes; ASR hallucination, captions có lỗi; QA ban đầu sai và chọn nhầm modality |
| `fd5931/after_restart.json` |Stored sự kiện còn sau restart; câu trả lời vẫn nhầm square thành circle. Persistence≠QAquality≠synaptic learning |
| Nativeaudit`4deffd` |Storage/types/hashPASS; chưa fullsemantic equivalence |
| Screencapture`b84a70` |Đường capturewindow thật PASS trên fixture, không benchmarkP1 |
| Finalsmoke`0a60b0` |Reinspect rawframe giúp nhận đúng xanh/đỏ và hìnhtròn; vẫn gọi hình chữ nhật là hình vuông, không nhận qualityPASS |
| Livevideo`5537cb` |ASR/VLM cùng resident gây peak4.045GB; guardstop, receipt giữ nguyên |
| Livevideo`4ae684` |Gọi trước lúcnativequalificationready→503; harness thêm chờreadiness, không sửa receipt |
| Livevideo`4003aa` |Whisper-cli + tạm dỡ VLM khi thiếu headroom: video21s xử lý xong, peakconservative3.629GB; QA vẫn sai, không semanticPASS |

Schedulingrepair: chỉ loadFP32ASR khi có đoạn cần nghe, giải phóng trước lượtVLM;
giữ nguyên trọng số. Finalprofile dùngCLIaudio theo đoạn; nếu dự toán còn thiếu
headroom, dỡ VLMfrozen theo component và nạp lại khi cần, ghi đầy đủI/O/cachecold.
Latestqualifications được ghi theo run riêng. Capture/ASR/
visionfrontiers riêng; không nói đã xem mọi frame. RawTTL10GB/7days không xóa
originalfiles; nguồn hết hạn được đánh dấu. Reinspection đọc bản sao còn giữ,
không âm thầm đọc lại original đã thay đổi.

## Neural thực sự làm gì?

Một quần thể128ALIF, inputs64, recurrence sparsefan-in8, output8. Hai maskedheads:
relevance2 và action6(review/search/verify/ask/defer/consolidate). Encoder fixed
signedhashunigram/bigram48dimensions +16numericstate;8causalticks/decision.
H128D8toy gains không được mang sang H128D64 mới này.

Hidden update dùng forwardeligibility đa thang thời gian, pre-update`Wout.T×error`
và per-synapseRMS/momentum theo auditedR8Own. Không backward qua tầng/thời gian.
Feedback có identity/evidence mới được học; generalwronganswer không gán nhãn
cho mọi retrievedevent. Importance array được lưu đầy đủ nhưng chưa là cơ chế
metaplasticity đã chứng minh. Consolidation hiện là boundedreservoir replay,
chưa là experience-to-generalskill trong VLM. VLM frozen không tự tăng weights.

SNN policy trong runtime hiện **unqualified**. Có weightupdate không đủ để nói
hidden representation hữu ích. Chưa có bằng chứng thắngfrozen/readout-only/
non-SNN trên streamvideo. Tên“neural_policy” trong trace chỉ xác định nguồn chọn
hành động, không có nghĩa đã qua benchmark.

## Tự học chủ động và những giới hạn

VLM đề xuất gap/contradiction từ quan sát/conversation khi thực sự vắng; ALIF
chọn trong tập hành động runtime cho phép. Querylocal trướcweb,≤2rounds/goal,
≤12searches/h,≤3openings/h,mộtpending. Userpresent ưu tiên; không thông báo khi
vắng. `public_topic` cần approval; built-incurriculum gồm chủ đề công khai phổ
thông theo quyềnweb người dùng đã duyệt. Query gửi đúngtopic đã cho phép, không
gắn privateconversation vào query. Nguồnweb/linkedsource vẫn chưa là true label.

Verifier đầu tiên chỉ toán nguyên xác định được. Dữ kiện mở và lời sửa của người
dùng giữ provenance; chưa có generalfactchecker. Không thưởng self-confidence,
novelty, số câu hỏi hay số trang. Policy có thểdefer sai hoặc hỏi chưa hữu ích;
đây là lý do cầnP2matchedcontrols, không giả controllerheuristic là model đã học.

## Gate tiếp theo

P0kỹthuật còn cần qualificationaudio-loopback và nativeattachmentE2E cuối cùng;
resource đo lại sau thay đổiruntime. TrướcP1 cần bộvideoVi/En thật1–2h với
100DEV+100TEST annotated độc lập, táchvideo và freeze nguồn/license/manifest.
Không biến syntheticfixture hiện tại thành benchmark để nhậnPASS.

P2 chưa chạy: seeds260113/260137/260171, strongestfixed/frozen/readout-only/
learned/shuffled matchedbudget;≥5pp với lower95%CI>0. Hidden ablation phải mất
gain và vượt readout-only.20stages meanforget≤3pp,worstgroup≤5pp. P3soak8h chưa
chạy. Ngưỡng không đổi sau kết quả. Chưa có LoRA/scale/modeltrainingmới.

Tests: 645 distinctcases đã qua trong các lượt split/targeted, không skipnative.
Lượt full cuối có6APIpass và638pass/1fail: failure là catalogcount120 sau khi
thêm mụcHYBRID-V1 thứ121. Đã sửa test để kiểm120mục cũ còn nguyên +1mục hybrid,
đồng thời giữ kiểmsourceG1 cũ không đổi;40affectedchecks qua sau khi stagefiles.
Không thay quality/forgetting gate để sửa failure này. Các sốtest
lặp không cộng thành evidence khoa học. Xem receipt bàn giao cuối để biết source
version/validation cuối, không suy rằng mọi commit sau được đo bởi receipt cũ.

Provenance: các lượt P0 sớm giữ raw receipts và source baseline/dirty metadata,
nhưng không có full source-byte bundle cho mọi chỉnh sửa giữa lượt. Không tuyên
bố bit-exact reproduction của các prototypefailed đó. Final code/models và
receipt bàn giao được khóa bằng Git SHA + asset/source hashes.
