# Học liên kết câu hỏi với câu trả lời đã quan sát

10/11/2026. **Đã hoàn tất12trials; STOP_OBSERVED_ASSOCIATION_V1. Không mainimport.**
[Protocol](../../experiments/g1-observed-association-v1.md).

Audit28 gợi ý scalarreward×score quá nhiễu; các familyđó đãSTOP. Hướng này thử
cơ chế khác: dấu vết presynaptic của câu hỏi được ghép với hoạt động ALIF khi
đọc câu trả lời đã cung cấp. Trọng số input→hidden được cập nhật bằng tích
pretrace×postactivity-trừ-mean. Slidingactivitymean hỗ trợpotentiation/depression,
không là importance nhân quả hoặc kiến thức đãconsolidate.

Đây là **label-conditioned hetero-associative Hebbian learning**, không phải
unsupervisedlearning hoặc tự biết đáp ánđúng. Observedanswer là privileged
traininginput và phải công bố. Khác target-rate family đãSTOP: không gán bảng
neuron/classrates, không rate-error/surrogatederivative/classifierfeedback hoặc
reward×likelihoodeligibility. Chỉ hoạt động ALIF thực trong pha quan sát và
tracecục bộ tạoupdate; không W_outbackward/BPTT/Transformer/LLMproxy.

Phải tách ba pha để tránh rò rỉ: đọc câu hỏi→quan sát answer để học hidden→
resetworkingstate và trainlocalreadout với previouswords. Khi đánhgiá/probe/
generate, **không chạy pha quan sát và không nhận current/goldanswer**. Reset
policy này là prototype nghiên cứu riêng, không sửa vòngneuroncontinuouscủaapp.

Một deterministicALIF48 cùngforwardG1, denseencoder128cốđịnh, readoutrawdelta,
chỉ contextinputsynapses học; recurrent/word/visionweights không học.12trials=
3freshseeds×paired/frozen/crossfamily/erase-trace, cùng1024experiences vàcompute.
So ownmatchedcontrols, không gọi khác biệt vs stochasticScoreWord là gaincủa
một thành phần. Đo hiddenprobe trênprompt-only và câu model tựsinh; không chọn
template/epoch/seed, không nhận trainceiling hoặc ΔW làmthànhcông.

Tối đa2GB/CPU,120s/trial/1800sround; codec riêng, noleak/locality/continuation
tests trước training. FailureSTOP riêng, không tự mở tuninground2 hoặc scale.
Giữ nguyên P1/P2/retention gate và kết quả lịch sử. Khôngimportcandidate vào
privateweights. Các dataAGY/Qwen thật đãcó được tái dùng đúngnguồn, không tạo
thêm dữ liệu chỉ để che lỗi thuậttoán. Visionbìnhthường và lifelongskill còn mở.

Cơ sở covariance/threshold:
[Neuronal Dynamics19.2](https://neuronaldynamics.epfl.ch/online/Ch19.S2.html).
Đây là adaptation kiểm chứng, không tuyên bố phátminh, mô phỏngnão hoặc tái lập
chất lượng/hiệuquảđiện của tài liệu. Chưa bảo đảm cơ chế này cải thiện semantics.

## Kết quả giữ nguyên tiêu chí

[Run](../../runs/20261109T213531Z-g1-observed-association-v1-d3e0e7/report.md).
3freshmodelseeds31013/31019/31033×4conditions,1024experiences mỗi model.
Mỗi condition cùng15tickqueryobservation,21tickanswerobservation cốđịnh,
12tickboundary+3tick/targetword cho readout; sampler/ticks khớp trong cùngseed.
Observedanswer chỉtraining, evaluation/probe sinh từ query và neuralpreviouswords.

| Điều kiện | Train16 tự sinh | Dev31 tự sinh | Prior48 tự sinh | Prompt-only probeDev |
|---|---:|---:|---:|---:|
| Frozenhidden |10,42%|2,15%|6,94%|31,18%|
| Pairedassociation |6,25%|2,15%|2,78%|29,03%|
| Cross-familyobservedanswer |8,33%|2,15%|4,86%|37,63%|
| Erasequerytrace |10,42%|2,15%|6,94%|31,18%|

Paired−frozen devgain0đ, CI95%−6,45..6,45; priorgain−4,17đ, CI95%−6,25..−2,08.
Prompt-onlyprobegain−2,15đ, CI95%−6,45..3,23. Paired−crossfamilydevgain0đ,
CI95%−3,23..6,45. Ba seedbootstrap thăm dò, dev/prior đã từng xem; không fresh
confirmation, không ý nghĩa phổ quát ngoài cấu hình/dữ liệu đãđo.

Mỗi pairedmodel có7168hiddenweighttransactions thật, ΔWcontextL2 trungbình6,14,
nhưng không qua gatefreeutterance hoặcrepresentation. Xóa trace triệt hidden
learning và khớp frozenweights/predictions chính xác, xác nhận đườngsynaptic
mechanism tồn tại; **đó không là bằng chứng cơ chế có ích**. Crossfamilyprobe
cao hơn paired mô tả một lầnreshapefeatures, không chứng minh đã tích lũy
semanticrule hoặc rằng nhãn sai có ích trong thực tế.

Baselinefrozen cũng yếu (teacher-tokenaccuracy≈42,92%dev và10,42%train16exact),
nên không quy mọi lỗi chỉ cho covariance hoặc tuyên bố mọi associativeSNN bất
khả thi. Rawhead/deterministicforward khác control27; không lấy44,09% từcontrol
đó làm learnabilitymatched ởđây. Không nới tiêu chí, normalizehead/tuneLR/epoch
sau khi xem kết quả hoặc nhập model. **STOP lượt v1; chưa tự mở vòng2.**

## Integrity, dữ liệu và giới hạn

[Technicalaudit](../../runs/20261109T213950Z-g1-observed-association-audit-v1-b8fcaf/report.md):
12/12coldprocesscontinuations khớp loss/nextreply/samplerRNG/fullbyteshash.
Frozen/erasedcorearrays khớp; onlycontextinputblock đổi, recurrent/word/control/
image/stablearrays giữ initial.7slots×1024=7168observationopportunities/model,
tick/samplerstream bằng nhau giữa4conditions từngseed; source/learningcode/
protocol/datahash giữ nguyên. Mỗi model117.562byte neural/localarrays, cộng
encoding/runtime được phản ánh trong số đo process. Không hiddengradient.

Lượt chính14,23s, peakprocesstreeRSS155,74MB/commit179,01MB;2CPUworkers, không
iGPU/offload/energyclaim. CorpusAGY117train+64authored thật,31dev/prior48 không
train.12.288experienceupdates là repeatedsamples, không12.288kiếnthức mới.
95Qwenreferences có sẵn chỉđối chiếu/khớpprompt, không dùng làm nhãn hoặcproxy.
Đầu ra toàn bộcases được lưu vàlocalhosttự theo tệp mới nhất, có condition/
labelconditionedtraining/gold-freeinference rõ, không bấm chạy/chọnmodel.

PhiênHermes vẫn một UnifiedCoreALIF, không đổi cơchế/trọng số hoặc kiểu neuron.
Normalchat/vision và lifelongretention chưa đạt. Master giữ các STOP/gate cũ;
tổngthểPIVOT. [Đề xuất30](30_PROPOSED_LEARNING_SIGNAL_BOUNDARY.md) nêu rõ một
ranhgiới tínhiệu khác và đang chờ quyết định người dùng; chưa code/train nhánh đó.
