# Đo trí nhớ và học liên tục của SCAR-X

Tài liệu kỹ thuật bổ sung cho [master plan duy nhất](../RESEARCH_PLAN.md#9-learning--memory-metrics).
Ngày 2026-11-09. Các phép đo mới dưới đây là **đề xuất protocol**, chưa có kết quả
benchmark G1. Không khởi chạy Phase 2.2, không thay ngưỡng lịch sử.

## Ba thứ cần tách riêng

1. **Giới hạn tiếp nhận:** G1 hiện nhận tối đa 2.048 ký tự mỗi tin nhắn. Hermes
   giữ bộ đệm giao thức 4.096 token, nhưng không gửi lại prompt công cụ hoặc cả
   lịch sử trong chế độ `stateful_chat`. Hai con số này là giới hạn kỹ thuật,
   không phải lượng kiến thức hay trí nhớ đã chứng minh.
2. **Trạng thái neural đang làm việc:** điện thế, adaptation, spike/features và
   traces có thể mang thông tin từ trước, cũng có thể suy giảm hoặc bị nhiễu.
   Tiếp tục chạy tick không chứng minh còn nhớ hoặc đang suy luận hữu ích.
3. **Kiến thức trong synapse và dữ liệu trên SSD:** kỹ năng phải thể hiện qua
   dự đoán trên ví dụ mới, kể cả sau reset trạng thái/restart trong đánh giá và
   tắt truy xuất. Tìm được một bản ghi hoặc replay lại đúng câu không chứng minh
   đã chuyển trải nghiệm thành kỹ năng. Bộ đệm hiện tại 128 prompt là dung lượng
   dữ liệu, không phải 128 ký ức neural được bảo đảm.

## Bảng đo thay cho mục tiêu độ dài context

| Nhóm đo | Định nghĩa / đơn vị | Kiểm tra để tránh kết luận sai |
|---|---|---|
| Chất lượng theo tuổi ký ức `Q(d,n,u)` | Accuracy/điểm kỹ năng sau độ trễ `d`, `n` sự kiện xen vào và `u` lần cập nhật weights | Báo cả thời gian, neural ticks và interference; chỉ định rõ background on/off |
| Tầm nhớ làm việc hữu dụng `H(q)` | Độ trễ lớn nhất trong miền đã thử mà mọi mốc trước đó vẫn đạt ngưỡng `q` đăng ký trước | Weights khóa; không replay/truy xuất; bài quan hệ và thứ tự, có distractor; không chọn một điểm may mắn ở xa |
| Giữ kỹ năng cũ | Accuracy tuyệt đối, mean/worst forgetting và backward transfer qua chuỗi học | Phải đã học được kỹ năng ban đầu; accuracy gần chance không được gọi retention tốt |
| Khả năng học mới | Số trải nghiệm, local updates, CPU giây để đạt ngưỡng chất lượng đã khóa | So với fresh baseline cùng ngân sách; không đạt ngưỡng ghi censored/unreached |
| Chuyển kinh nghiệm thành kỹ năng | Heldout transfer/compositional generalization, forward transfer | Cách diễn đạt/quan hệ/tình huống mới; retrieval off, ablation readout-only/frozen hidden |
| Dung lượng kỹ năng hữu dụng `C(q,I,B)` | Số nhóm kỹ năng vẫn qua ngưỡng `q` sau lịch interference `I`, trong tài nguyên `B`, đồng thời còn học kỹ năng mới | Phải công bố bộ kỹ năng, độ khó, verifier, RAM/CPU/replay; không suy rộng thành số sự kiện có thể nhớ tùy ý |
| Chi phí và độ tin cậy | Peak RSS/commit khi học/restore, latency, restart fidelity, CPU/SSD growth | Cùng quality và compute; số neuron/spike/synapse không thay cho chất lượng |

`q`, lịch delay/interference, tập kỹ năng và verifier phải khóa **trước** lần
chạy. `H(q)` chỉ có giá trị trong miền tác vụ/phân bố đã đánh giá; không quảng
cáo thành “nhớ vô hạn”. Có thể báo đường cong thay vì ép thành một scalar.

Các chỉ số accuracy, backward/forward transfer có cơ sở trong nghiên cứu
[GEM](https://arxiv.org/abs/1706.08840). Dùng cách đo không buộc SCAR-X dùng
thuật toán gradient của GEM. Các định nghĩa `H`/`C` ở đây là đề xuất vận hành
của dự án, không phải một chuẩn trí thông minh mới đã được công nhận.

## Quy trình phân biệt nguồn trí nhớ

- **Working memory:** model clone cố định weights, input rồi delay/distractors;
  không reset trạng thái giữa các mốc của một episode, không truy xuất/replay.
  Đánh giá quan hệ/thứ tự trên mẫu giữ kín, tránh bài recall có hiệu ứng trần.
- **Synaptic skill:** học có nhãn được kiểm tra, sau đó clone/reset trạng thái
  làm việc và restart từ checkpoint; tắt memory search và idle probes. Chấm
  heldout transfer. Reset ở đây chỉ là can thiệp đánh giá trên clone, không đổi
  lifecycle liên tục của runtime sản phẩm.
- **SSD/replay:** bật/tắt riêng, cùng lượng dữ liệu và compute. Báo retrieval
  accuracy/latency riêng với neural skill; so sánh idle-on với idle-off trong
  ngân sách tổng bằng nhau để tránh lợi ích do thêm training.
- **Hidden contribution:** full vs readout-only/frozen hidden, can thiệp hidden
  hoặc readout với dữ liệu và ngân sách ghép cặp; trọng số thay đổi chưa là bằng
  chứng representation có ích.
- **Lifelong:** dự đoán trước khi học trên stream không task boundary; chấm theo
  mốc đã khóa. Báo chất lượng cũ/mới, forgetting, plasticity và transfer cùng lúc.

Giữ mean forgetting **≤3 điểm phần trăm**, worst **≤5 điểm** theo protocol đã
đăng ký. Không dùng phép đo mới để đổi các trial fail thành pass. CI phải phản
ánh seed độc lập và dữ liệu có nhóm/phụ thuộc; CI nhị thức theo mẫu không thay
cho bất định giữa seed. G1 hiện chỉ có hai seed joint-training và chưa chứng minh
working-memory horizon hay dung lượng kỹ năng trên bộ đo mới.

## Giới hạn của suy luận từ topology

Một hệ có trạng thái hữu hạn không có bảo đảm lưu chính xác vô hạn thông tin
độc lập. Định lý về capacity trong
[Dambre et al.](https://www.nature.com/articles/srep00514) có các giả định cụ
thể về hệ động lực, biến trạng thái và phép đọc; không được lấy “48 neuron” rồi
suy ra “48 ký ức” hoặc một mức IQ. Kích thước weights, readout, precision,
plasticity, mã hóa và yêu cầu tác vụ đều có vai trò. Khi chưa đo, ghi **chưa đo**.

## Trạng thái hiển thị

Hiện đo được: số neuron/synapse/ticks/updates, RAM/CPU, bounded buffer và exact
continuation. Chưa đo được: `H(q)`, `C(q,I,B)`, lợi ích reasoning lúc rảnh,
Experience-to-Skill hoặc retention đạt gate. Dashboard không được gán điểm giả
từ số tick, spikes hoặc lượng dữ liệu đã nhập. Chưa triển khai bộ benchmark mới
chỉ vì đã đặt tên chỉ số.
