# P2: readout bị ghi đè và representation chưa đủ cho nhiều context

Ngày 09/11/2026. **Chưa qua P2.** Không có phương pháp bảo vệ synapse mới được
thử trong vòng này; chưa kết luận importance không hiệu quả. Mục đích là tách
nguyên nhân trước khi dành tài nguyên tối ưu replay/consolidation.

## Thiết kế và giới hạn

[Protocol](../../experiments/p2-paired-score-diagnostic-v1.md),
[run gốc](../../runs/20261109T140620Z-p2-paired-score-diagnostic-v1-c5a357/report.md):
3 seed mới × 6 điều kiện = 18 lượt. Giữ quy tắc paired-score đã khóa ở P1,
ALIF48/K8/R12; D24/O4 theo bài stress 20 context lịch sử. Không chuyển weights
D8/O2 của P1 sang cấu hình không tương thích hoặc tăng quy mô neuron.

Mỗi context có cue đầu vào quan sát được; stream trộn dần hai context liền
kề, không đưa task ID/boundary metadata cho learner hoặc replay. Đây vẫn là
bài có context cue và reset mỗi episode 12 tick, **không phải** dòng trải nghiệm
tự nhiên không đánh dấu hoặc policy never-reset của model trong Hermes.

Mỗi lượt có 3.840 quan sát mới và 1.920 slot ôn lại/lặp mẫu mới, tổng 5.760
updates, 8 rollouts/update. Reservoir 128 episode được cấp cùng dung lượng cho
mọi điều kiện. Calibration chỉ dùng context0 chưa có nhãn, không dùng tương lai.
Test cố định, evaluation stateless/common noise, không rollback trên test.

Offline-joint được học trộn toàn bộ training labels, kể cả tương lai. Đây là
đối chứng learnability lạc quan, không là thuật toán lifelong được phép triển
khai. Refit ridge chỉ là diagnostic head, không nhập vào core hoặc runtime.

## Kết quả trung bình ba seed

| Điều kiện | Acquisition | Final | Quên TB | Quên tệ nhất* | Refit probe |
|---|---:|---:|---:|---:|---:|
| Full, lặp mẫu mới | 62,10% | 30,49% | 33,92đ | 67,45đ | 50,13% |
| Frozen hidden, lặp mới | 56,71% | 26,50% | 32,23đ | 85,68đ | 44,75% |
| Full + reservoir | 57,64% | 34,74% | 29,09đ | 71,88đ | 53,41% |
| Frozen + reservoir | 50,49% | 28,67% | 27,04đ | 84,11đ | 44,75% |
| Offline-joint full | không áp dụng | 30,83% | không áp dụng | không áp dụng | 49,04% |
| Offline-joint frozen | không áp dụng | 26,43% | không áp dụng | không áp dụng | 44,75% |

*Cột tệ nhất là trung bình của worst-context từng seed; không dùng số gộp này
thay gate phải đạt trên **mọi seed**. Mọi stream condition đều fail acquisition/
final65%, mean-forgetting≤3đ và worst≤5đ. Không mở confirmation/scale từ vòng này.

Replay giảm quên full từ33,92 xuống29,09đ nhưng acquisition giảm4,45đ, vượt
margin3đ đã đăng ký. Chỉ bằng nhau số update, chưa equal-time confirmation;
không thể gọi replay thắng tiêu chí continual learning. So full với frozen
trong reservoir, refit probe tăng8,66đ nhưng vẫn chỉ53,41%; chưa đủ chất lượng.

## Vì sao quên?

Khi hidden hoàn toàn không đổi, forgetting vẫn32,23đ nếu lặp mẫu mới và27,04đ
với reservoir: readout bị ghi đè là một nguyên nhân có bằng chứng trực tiếp.
Hoán đổi old/new head và hidden trên cùng dữ liệu/noise cho full+reservoir:
head-only drop trung bình5,04đ mỗi window, hidden-only1,47đ, interaction−1,57đ.
Đây là tương tác theo window, không cộng các số này thành tỷ lệ nguyên nhân
của toàn bộ forgetting hoặc gọi head chịu một phần trăm cố định.

Tuy nhiên, bảo vệ head không tự tạo feature thiếu: cả offline-joint và fresh
refit head trên raw hidden features đều dưới65%. Bằng chứng chỉ áp dụng cho
implementation/budget hiện tại; không là định lý mạng48neuron không thể giải
bài. Vì acquisition cũng yếu, chưa nên tối ưu importance để đóng băng trạng
thái chưa học được kỹ năng có chất lượng.

## Thông tin đầu vào, trạng thái và noise

[Protocol audit hậu kiểm](../../experiments/p2-cue-information-audit-v1.md),
[run](../../runs/20261109T141355Z-p2-cue-information-audit-v1-a427f6/report.md):
6 checkpoint offline-joint, không train core, không đổi RNG/weights/calibration.
Cùng dữ liệu train/test của nguồn; refit head tại prefix2/6/12 và MC8/64.

| Hidden nguồn | MC rollouts | Tick2 | Tick6 | Tick12 |
|---|---:|---:|---:|---:|
| Đã học offline | 8 | 51,56% | 54,44% | 48,44% |
| Đã học offline | 64 | 64,87% | 67,66% | 59,82% |
| Frozen | 8 | 50,00% | 52,94% | 45,13% |
| Frozen | 64 | 65,78% | 67,94% | 56,09% |

MC64 dùng thêm8× inference, không là một cải thiện cùng compute hay cách vận
hành đã triển khai. Khác seed/noise schedule với probe nguồn nên các số MC8
có thể khác nhẹ; audit không thay số nguồn hoặc chọn lại model.

Cue/context raw 24feature chỉ cho ridge khoảng39–42%; outer-product80feature
đạt100% trên cả3seed. Đây là **engineered input-information control**, nhiều
feature hơn core48; không là neural learning, fair-capacity baseline, RAG hoặc
đáp án trả lời thay model. Nó xác nhận thông tin liên kết có trong input.

Kết quả gợi ý noise che thông tin và decodability suy giảm về cuối chuỗi.
Không xác định riêng một cơ chế neuron gây ra, không chứng minh trí nhớ bị
xóa vĩnh viễn. Giảm variance bằng nhiều mẫu chưa đủ giúp endpoint đạt65%.
Thêm replay hay importance không thay việc cần cải thiện representation và
working-state dynamics trong ngân sách thực tế.

## Kiểm chứng, tài nguyên và quyết định

18/18 model+replay continuations khớp; frozen-hidden intervention có hidden-only
effect0. Neural state/RNG không bị evaluation hoặc audit thay đổi. Source run
mất278s, peakRSS221MB/commit245MB; audit51s, RSS64MB. Thời gian trial lớn nhất
33,3s, dưới180s đã khóa. Main/private core không nhận weights hay thuật toán.

132 tests qua, gồm phần đọc số đo trên website và native baseline parity.
Checkpoint, replay/RNG, cursor, stream hash, accuracy matrices, mọi dự đoán và
component interventions giữ trong run. P1 order/XOR pass vẫn là kết quả lịch
sử hợp lệ; không suy ra P2 hoặc khả năng chat/vision từ nó.

**PIVOT ưu tiên nghiên cứu:** giữ acquisition cùng retention, tập trung biểu
diễn cue–context và độ bền trạng thái/noise, đồng thời kiểm readout stability.
Importance chưa được thử nên không kết luận NO-GO cho importance. Mọi thay đổi
dynamics/architecture phải có đề xuất và protocol riêng, tách khỏi code đang
phục vụ; không mở vòng3 hidden-score hoặc tăng model/data để che failure.
Mục tiêu normalchat/vision, broad skill và Experience-to-Skill vẫn chưa đạt.
