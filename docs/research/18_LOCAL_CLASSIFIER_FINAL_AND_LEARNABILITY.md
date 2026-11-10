# P1: vòng cuối của local classifier và đối chứng learnability

Ngày2026-11-09. **NO-GO cho nhóm adaptation local-classifier đã thử** sau hai
vòng; dự án SCAR-X tiếp tục PIVOT. Không mở confirmation, tăng quy mô, nhập
weights vào model live hoặc đổi gate để nhận một cải thiện chưa đủ.

## Giả thuyết và đường tín hiệu

V1 dùng readout chính đã normalize nhưng classifier phụ nhận spike-rate thô.
V2 kiểm hai khả năng: conditioning của local loss chưa phù hợp và thời điểm dạy
binding trước query làm tín hiệu nhiễu. Đây là giả thuyết, không là nguyên nhân
đã được chứng minh từ việc đọc code.

[Protocol v2](../../experiments/p1-local-classifier-v2.md) khóa trước:3fresh
development seeds×3tasks×12configurations=108trials,2.048examples/trial. Giữ
ALIF48/K8/R12/D8/O2, cùng forward và topology. Mean/std lấy từ128unlabeled
calibration episodes, frozen trước nhãn. Không lấy statistics từ dev/test.

Classifier phụ vẫn là fixed randomG, independent taskW_out. Với feature đã
normalize và clip5, tín hiệu postsynaptic là `G.T(onehot(y)-p)` nhân đạo hàm
clip/scale. Synapse nhân tín hiệu đó với forward EPSP/eligibility nhưv1. Không
reverse-time tape, autograd, taskW_out feedback hoặc truyền gradient qua các
neuron trong candidate. Phép chiếuG.T vẫn là teacher-error transport có thông
tin không cục bộ; không gọi là học thuần tín hiệu tại mỗi synapse/unsupervised.

LAST4 cập nhật sau ticks8,9,10,11; FINAL4 cập nhật4lần sau tick11, không thêm
neural tick. Cùng12forwardticks,4hiddenupdates và1main-headupdate/episode.
Đối chứng raw/normalized, timing, shuffled teacher, readout-only, scalarSTDP.

Đây là adaptation thủ công ý tưởng local auxiliary loss; không tái lập đầy đủ
[DECOLLE](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00424/full).
Forward ALIF/filter và quy tắc eligibility của SCAR-X khác công trình nguồn.

## Kết quả giữ nguyên gate

[Run gốc](../../runs/20261109T123739Z-p1-local-classifier-v2-fedda9/report.md)
chọn normalized-LAST4,LR0,002 bằng mean development probe gain đã đăng ký.

| Bài | Initial hidden probe | Candidate probe gain | CI95% gain, điểm % | Candidate direct | Readout-only direct | BPTT direct |
|---|---:|---:|---:|---:|---:|---:|
| Order | 92,71% | +2,60đ | −0,78…4,69 | 96,61% | 93,75% | 99,87% |
| XOR | 86,20% | +3,78đ | 2,34…5,47 | 91,93% | 88,67% | 100,00% |
| Strict binding | 58,33% | +0,26đ | −2,73…3,13 | 65,49% | 62,37% | 91,67% |

CI là seed bootstrap chỉ3development seeds, đã tham gia chọn candidate;
**thăm dò, không phải confirmation độc lập**. Không đạt gain5đ trên2tasks,
không mở8freshconfirmation seeds. Direct cao trên order/XOR không thay gate
representation, không chứng minh normalchat/vision hoặc solved lifelong.

## Điều gì có ích, điều gì chưa có ích?

[Audit additive](../../runs/20261109T124335Z-local-classifier-v2-attribution-5e01a0/report.md)
giữ nguyên selection/decision, chỉ phân tích paired seed effects:

- Normalize so với raw, cùng LAST4/LR0,002: probe tăng4,69đ trênorder,
  5,73đ trênXOR nhưng giảm1,04đ trênbinding. So với initial hidden, lợi ích
  thấp hơn; raw auxiliary có thể làm representation tệ đi. Chưa qua joint gate.
- Dồn4updates về tick cuối so với LAST4: probe +0,39đ/−1,95đ/+1,56đ trên
  order/XOR/binding. Không có lợi ích nhất quán; giả thuyết timing không được
  xác nhận. Không mở thêm vòng để tìm lịch cập nhật khác cho cùng family.
- Teacher đúng so với shuffled, selected setting: binding probe thấp hơn2,86đ;
  direct chỉ hơn0,65đ. Chưa có bằng chứng chắc về representation quan hệ học
  từ nhãn đúng. Các p sauHolm3tasks đều chưa đạt với3seeds.
- Hidden weights thay đổi không đủ để gọi là học representation. Số cập nhật,
  loss, ΔW và mọi dự đoán đã được lưu; không dùng bảngloss làm chứng nhận kỹ năng.

## Đối chứng BPTT nói được và không nói được gì?

18isolatedcontrols: cùng3seeds×3tasks, readout-only/fullhiddenBPTT. Forward ALIF,
topology, initialization, input, sample order, calibration và local main-head
delta rule giống candidate. **9/9readout controls có toàn bộ target/prediction
khớp NumPy readout-only**; forward parity kiểm cả trước/sau học.

Fullcontrol dùng taskW_out error qua neuron/time, Adam hiddenLR0,002, normclip1,
updateclip0,025; chỉ nằm trong `evaluation/controls.py`, không phục vụ Hermes.
Control chứng minh kiến trúc nhỏ này có thể học binding tốt hơn trong budget;
không thể quy mọi thất bại cho thiếu neuron hoặc dữ liệu bất khả học.

Tuy nhiên, control khác local về objective, đường credit, optimizer và coverage
mọi hidden synapse so với rotating12rows. Không chỉ ra riêng thành phần nào gây
thất bại, không chứng minh BPTT là cách duy nhất và không là lợi thế equal-time.
Local có4hiddenupdates/episode, BPTT1 nhưng full temporal graph/allrows; head
updates và examples giống nhau. Thời gian/RAM/buffer thực nằm trong raw result.

## Persistence, RAM và trace

9selected checkpoints tiếp tục từ state/RNG/eligibility/calibration trong Python
process mới cho hash giống hệt reference continuation; chỉ là audit clones với
random input, không sửa original weights hay mở nghiên cứu học mới. Giữ codec
v1 bất biến; v2 có format riêng, stats frozen và checksum đầy đủ.

Lượt nghiên cứu mất khoảng186s, peak process-treeRSS750,1MB, committed khoảng
1,25GB, không Qwen đồng thời. Sampling không là hard aggregate cap; Windows
per-process cap2GB đã áp dụng. Model live vẫn một core, không nhập candidate.

Trên localhost đã quan sát job đang sống,40+/108trials,48neuron thật, learning
blocks512examples với loss/ΔW và heldouttensor→prediction. Các output là
classification, không giả hội thoại hoặc Qwen text. Poll chỉ đọc. Test đối
chứng được táchprocess để Torch không làm nhiễu guard RAM của runtime tests;
guard2GB không bị nới để làm suite qua.

## Hướng tiếp theo trong master plan

Đóng family local-classifier, local-target-rate và main-error random-feedback
đã thử; không gọi lại bằng tên khác. P1 còn bài toán tín hiệu/trace/coverage và
representation quan hệ; P2 còn readout stability/importance và plasticity.
Cần protocol riêng cho giả thuyết khác, phân lập teacher signal, trace và
optimizer/updatecoverage trước khi quy nguyên nhân hoặc đề xuất topology mới.

Không có bằng chứng đủ để train10Mtoken/tăng model, dùng replay che P1fail,
hoặc nói đã chuyển trải nghiệm thành kỹ năng. Antigravity corpus và Qwen
reference trước đó vẫn có nguồn, giữ nguyên; thêm dữ liệu không giải quyết
được credit assignment chưa học được. Mục tiêu normalchat/vision chưa hoàn tất.
