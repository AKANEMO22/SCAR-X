# Tăng tải kiến thức, retention và mốc Qwen4B

2026-11-09. Đây là diagnostic của **core hiện được triển khai: một quần thể
recurrent48neuron**, không phải đánh giá một kiến trúc neural nhiều tầng đã triển
khai. Live app đang có64neuron đã cấp; phép thử dùng public release48để cô lập
rule và giữ lịch sử người dùng. Không kết luận cho một thiết kế phân tầng khác.

Người dùng đã làm rõ “ba tầng” là ba chế độ fast/deep/idle trong cùng core,
không phải stack nhiều lớp vật lý; xem [tài liệu12](12_ADAPTIVE_COMPUTE_AND_IDLE_COGNITION.md).
Phép thử này chưa đánh giá ACT hoặc autonomous exploration chưa triển khai.

## Protocol và bằng chứng

- [V1](../../experiments/knowledge-load-retention-v1.md): signed arithmetic;
  [run bị dừng](../../runs/20261109T071710Z-knowledge-load-retention-v1-1a4927/report.md)
  do decoder không cho dấu trừ đứng đầu. Không đổi code neural hoặc giấu lỗi.
- [V1r1 đăng ký riêng](../../experiments/knowledge-load-retention-v1-amendment.md):
  đáp án không âm;128curriculum/32biểu thức giữ kín/8bài gây nhiễu. Nhãn tính
  lại độc lập. [Run chính](../../runs/20261109T071930Z-knowledge-load-retention-v1r1-c674ae/report.md).
- [Audit bổ sung](../../experiments/knowledge-load-audit-v1.md) và
  [báo cáo đã sửa metric](../../runs/20261109T072524Z-knowledge-retention-audit-a99682/report.md).
  Raw runs giữ nguyên. Field `old_mean_forgetting_pp` ban đầu thực chất là net
  accuracy drop; audit tính đúng trung bình mất điểm từng family theo điểm cao
  nhất trước đó. Không cho gain ở family khác che loss; ngưỡng3pp/5pp không đổi.
  Stop rule dùng acquisition, nên audit không thay quyết định dừng hoặc weights.

Ba seed17/29/43 là ba thứ tự curriculum từ **một initialization pretrained**,
không phải ba model initialization độc lập. Chưa đủ để suy rộng đến mọi core.
Không mở lại acceptance set lịch sử, không chọn winner để triển khai.

## Kết quả trên model hiện tại

| Cơ chế | Số bài/order | Đúng nguyên cách hỏi | Đúng cách hỏi mới | Quên cũ TB đã audit | Worst family |
|---|---:|---:|---:|---:|---:|
| Frozen |128|0%|0%|0pp|0pp|
| Full local |16|14,58%|0%|25pp|100pp|
| Readout-only |16|20,83%|0%|25pp|100pp|
| Full + replay |16|31,25%|0%|25pp|100pp|
| Guarded |16|0%|0%|8,33pp|50pp|

Các bản được học dừng sau hai bậc8và16cùng không đạt tăng10pp trên cách hỏi
mới, đúng stop rule. Bản frozen chạy tới128không phải đã học128kiến thức.
Guard chỉ bảo vệ vài train probe; mất điểm development vẫn xảy ra. Bảo vệ guard
không phải bảo đảm toàn bộ kỹ năng không quên. Các tỷ lệ trên là mô tả ba order,
không phải ước lượng năng lực đa năng.

Old-skill diagnostic chỉ có2development prompts/family,8families và baseline
25%trước học. Nó chưa đạt quality gate65%ngay từ đầu, nên không phải phép nghiệm
thu Phase2.2 đầy đủ. Chênh lệch50pp/100pp có độ phân giải rất thô; ba order từ
một initialization không thay thế seed-confirmation/cỡ mẫu lớn hơn. Các số đo
cho thấy lỗi trong cấu hình này, không chứng minh mọi SNN hoặc mọi kiến trúc
phân tầng đều thất bại.

Full và readout-only đều0%trên paraphrase: **hidden chưa chứng minh học
representation hữu ích**. Readout-only còn nhớ nguyên cách hỏi nhiều hơn full
trong vòng này; không suy ra hidden sẽ luôn vô ích với mọi quy tắc khác.

## Nhớ lâu đến đâu?

Không có bài mới đạt định nghĩa acquisition hữu ích trên paraphrase, nên primary
retention **không xác định**; không được ghi100%vì không có gì đã học.

Phân tích thăm dò khác giữ các đáp án đúng nguyên cách hỏi tại checkpoint, rồi
thêm8bài mới/256update slots. Tổng theo ba order:

| Cơ chế | Đúng trước nhiễu | Giữ sau nhiễu | Giữ với replay cùng budget |
|---|---:|---:|---:|
| Full |7|0|0|
| Readout |10|0|0|
| Full + replay |15|1|3|

Replay có giúp giữ vài đáp án nhưng chưa đạt retention hay generalization gate.
Đây là probe thăm dò sau kết quả, không thay tiêu chí thành công ban đầu.
10.000quiet ticks và cold process reload giữ kết quả tương ứng, chỉ chứng minh
persistence/dynamics trong phép thử ngắn. **Chưa kiểm chứng nhớ qua ngày/tháng.**

## Qwen được đối chiếu thực tế

[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
GGUF Q4_K_M của lmstudio-community, pinned revision/hash trong protocol. Không
nhập weights vào SCAR-X, không sửa cấu hình Hermes. CPU4threads, context1024,
temperature0, max12outputtokens. [Run đối chứng](../../runs/20261109T072118Z-qwen4b-numeric-reference-bda43a/report.md).

Trên160câu số học không âm, **Qwen160/160**, gồm32biểu thức chưa dùng để dạy
SCAR-X. Phép đối chiếu phụ đưa cùng160chuỗi prompt có yêu cầu chỉ trả số cho
cả hai bên: SCAR-X full/readout/replay trung bình0,2083%tức1lượt đúng/480lượt
qua ba order; guarded/frozen0%. Đây không phải cùng toàn bộ điều kiện pretraining.

Qwen peak resident tree khoảng2,924GB, commit khoảng3,604GB, đã tắt sau đo.
SCAR-X diagnostic peak resident tree khoảng102,7MB. **Không matched-memory**;
Qwen bản lượng tử này cũng không nằm trong trần2GB. Không dùng phép toán nhỏ này
để tuyên bố biết toàn bộ lượng kiến thức của4B hay ngang Qwen đa năng.

## Quyết định theo master plan

**PIVOT_NO_SCALE** cho rule hiện tại. Dữ liệu nhiều hơn không sửa được những
điểm đang thấy: input chưa có temporal encoding đủ mạnh, scalar hidden credit
assignment yếu, readout bị ghi đè và guard bảo vệ không bao phủ kỹ năng.
Thử nghiệm này không phân rã được mọi nguyên nhân hay chứng minh một kiến trúc
phân tầng chưa triển khai sẽ thất bại.

Đi tiếp theo P1representation/credit assignment và P2readout stability của
master; làm rõ phân tầng neuron trước khi chọn thay đổi topology. Tiếp tục dùng
Antigravity cho curriculum có kiểm tra nguồn/nhãn, không coi teacher label là
chân lý. Normal photo QA vẫn cần encoder/đường học ảnh–từ đã được kiểm chứng.
Không mở pretraining lớn, không thay framework hoặc thêm model suy nghĩ riêng.

Trace live: **http://127.0.0.1:8765/** — source input, spikes, voltage, update,
review/rollback; không phải diễn giải ngữ nghĩa nội tâm. Phép thử/các clone và
Qwen đối chứng không thay live core hay dữ liệu người dùng.
