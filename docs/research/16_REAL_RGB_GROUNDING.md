# Ảnh thật: phép thử RGB object grounding nhỏ

Ngày 09/11/2026. Diagnostic thuộc P1 visual representation. Không đổi model đang
phục vụ, không pretrained encoder, LLM proxy, backprop hoặc tăng hidden topology.
Phân loại ảnh nhỏ không thay thế mục tiêu hiểu và trả lời về ảnh bình thường.

## Nguồn dữ liệu và split

[Protocol gốc](../../experiments/rgb-object-grounding-v1.md) tải bản binary CIFAR,
nhưng vượt 360 giây và kết thúc lỗi. File tải dở không được dùng. Theo
[source amendment](../../experiments/rgb-object-grounding-v1-source-amendment.md),
chuyển sang [UofT-CS/CIFAR10](https://huggingface.co/datasets/uoft-cs/cifar10),
khóa revision và SHA-256 của train parquet. Không tải hoặc mở original test,
không dùng pickle hoặc remote dataset code.

Nguồn tham khảo: Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny
Images*, 2009; [trang dataset gốc](https://www.cs.toronto.edu/~kriz/cifar.html).
Card ghi license chưa rõ; ảnh chỉ dùng local research và không có trong ZIP.

Từ 10.000 training rows đầu, chọn 64 train và 16 development mỗi lớp bằng seed
10103: tổng 640/160 ảnh RGB 32×32. Không có image hash trùng giữa train/dev.
Thứ tự có thể khác binary batch gốc, không giả tương đương indices. Nhãn do
nguồn cung cấp có thể nhiễu; chưa xác minh từng ảnh. Bộ này không đại diện ảnh
độ phân giải cao, screenshot, OCR hoặc visual question answering.

## Đường xử lý và kết quả

Cùng ByteBackbone 48 ALIF, K=8, R=12, D=454. InputEvents chuyển RGB 16×16 thành
16 patch 4×4; lấy trung bình 48 neural features qua các patch nội dung. Reset
working state giữa ảnh độc lập chỉ trong diagnostic, không tác động core live.
Hidden đóng băng; local output delta học head 10 lớp. Raw và normalized cùng
2.560 update/seed, ba seed độc lập, không chọn model theo development.

[Run sáu trial](../../runs/20261109T112104Z-rgb-object-grounding-v1-1fae49/report.md):

| Head | Development | Training | Ảnh bị che | Ảnh ghép khác lớp, đã audit |
|---|---:|---:|---:|---:|
| Raw | 18,96% | 20,47% | 10,00% | 10,00% |
| Chuẩn hóa | 20,42% | 33,44% | 10,00% | 10,83% |

Ngưỡng đã đăng ký là accuracy ≥30% và tăng ≥10 điểm so với cả blank và mismatch.
**Chưa đạt.** Neural features mang một ít thông tin ảnh dùng được ở readout,
nhưng chưa đủ chất lượng. Hidden frozen nên đây không chứng minh hidden học
representation mới. Chuyển class ID thành tên bằng host cũng không phải model
đã học sinh caption hoặc trả lời ngôn ngữ.

## Lỗi đối chứng và cách sửa

Raw mismatch dùng roll một vị trí trên development xếp theo lớp: 150/160 lần
giữ cùng class. Vì vậy không được dùng chênh lệch khoảng 1 điểm ban đầu làm
bằng chứng ảnh không có tác động.

[Audit additive](../../runs/20261109T112204Z-rgb-object-grounding-negative-audit-6517c5/report.md)
đăng ký trước khi sửa, giữ nguyên raw run và source. Mapping mới đổi lớp ở mọi
160 ảnh, là một hoán vị không trùng vị trí. Vì readout là hàm pointwise, có thể
hoán vị saved predictions để kiểm lại mà không retrain, chọn split hay sửa
ngưỡng. Kết luận vẫn fail ngưỡng 30%.

## Khoảng cách còn lại

Chưa có word decoder được học trên ảnh, câu hỏi khác nhau về cùng ảnh, OOD
source/classes, đánh giá mù hoặc đối chứng Qwen vision. Qwen3-4B-Instruct hiện có
là model văn bản; không giả gửi ảnh cho nó để lấp chỗ thiếu. So vision cần QwenVL
đúng năng lực và protocol/budget riêng, không proxy vào SCAR-X.

P1 cần visual feature learning có ích; P2 cần giữ acquisition và retention cùng
lúc. Giữ một core, CPU-first, mục tiêu 2 GB và không backward nhánh chính.
**PIVOT cơ chế, chưa tăng quy mô hoặc đưa candidate này vào Hermes.**
