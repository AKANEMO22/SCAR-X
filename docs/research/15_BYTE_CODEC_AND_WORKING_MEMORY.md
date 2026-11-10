# Codec UTF-8, đối chứng tối ưu và bộ nhớ khi sinh câu

Ngày 09/11/2026. Tiếp tục P1/P2 trong master plan, giữ 48 neuron ALIF và topology
cũ. Các thử nghiệm đều cô lập; không thay model hoặc dữ liệu cá nhân trong Hermes.

## 1. Biểu diễn được từ mới chưa có nghĩa là nói được

[Protocol byte](../../experiments/byte-dialogue-prototype-v1.md) dùng 256 giá trị
byte cố định, cộng BOS/EOS. Cursor kiểm cú pháp theo
[RFC 3629](https://www.rfc-editor.org/rfc/rfc3629.html), chặn UTF-8 không hợp lệ
và ký tự bị cắt giữa chừng. Không chèn đáp án hoặc ký tự thay thế vào output.

Cả 24 đáp án development có thể biểu diễn mà không lấy từ vựng từ development;
codec từ trước đó chỉ biểu diễn đủ 9/24 đáp án. Tuy nhiên, UTF-8 hợp lệ là bảo
đảm của host mask, không phải khả năng ngôn ngữ mà neuron đã học.

Đầu vào 454 kênh, đầu ra 268 kênh; số neuron vẫn 48. Chi phí và số bước byte
khác thử nghiệm word, nên không coi hai trial là matched-compute.

[12 trial, 3 seed × 4 condition](../../runs/20261109T100856Z-byte-dialogue-prototype-v1-e4e370/report.md),
4 lượt qua 96 phản hồi train, 24 phản hồi development:

| Cơ chế | Dev exact cả câu | Train exact cả câu | Đúng byte khi teacher forcing |
|---|---:|---:|---:|
| Khóa học | 0% | 0% | 0,48% |
| Readout cục bộ, raw | 0% | 0% | 22,92% |
| Readout chuẩn hóa | 0% | 0% | 12,33% |
| Chuẩn hóa + hidden scalar STDP | 0% | 0% | 13,16% |

Output vẫn vô nghĩa hoặc lặp. Các checkpoint khớp output và score khi nạp trong
12 process mới, nhưng persistence không phải chất lượng. Đã dừng đúng protocol;
không thêm lượt học, nới metric hoặc đưa candidate vào phiên người dùng.

## 2. Đối chứng BPTT độc lập cũng chưa học được trong ngân sách này

[Protocol control](../../experiments/byte-bptt-learnability-control-v1.md) và
[6 trial](../../runs/20261109T101515Z-byte-bptt-learnability-v1-520af1/report.md)
giữ cùng forward ALIF, dữ liệu và kích thước, dùng ba seed mới. Adam-readout chỉ
học head; Adam-BPTT học mọi trọng số với surrogate derivative. Code backward chỉ
ở `evaluation/controls.py`, không được import vào nhánh chính.

| Control | Train byte accuracy | Dev byte accuracy | Dev exact cả câu |
|---|---:|---:|---:|
| Adam-readout | 25,92% | 16,34% | 0% |
| Adam-BPTT | 27,98% | 23,18% | 0% |

Không control nào đạt điều kiện train learnability 65%. Ngân sách bốn lượt học
và core nhỏ giới hạn kết luận: chưa thể quy mọi lỗi cho local learning hoặc kết
luận mọi SNN đều bất khả thi. Optimizer, gradient và cách gộp update khác nhau;
đây không phải so sánh cùng compute. CPU một thread, không CUDA, peak RSS khoảng
737 MB. Artifact chỉ là mảng đối chứng, không có optimizer resume và không được
gọi là checkpoint SCAR-X để vận hành.

## 3. Thông tin đầu vào mất dần trong lúc decoder phát byte

[Audit](../../runs/20261109T110945Z-decoder-state-retention-audit-v1-9225d1/report.md)
dùng 32 cặp last-write mới, ba model mới và ba checkpoint đã học. Hai nhánh bắt
đầu cùng trạng thái, đọc câu khác thứ tự rồi nhận cùng chuỗi previous-byte và
pooled context.

Features còn khác sau encoding. Đến byte 32, khoảng 69–94% cặp trong các model
đã học có features giống hệt; đến byte 127, mọi cặp của cả sáu model đều trùng.
Chênh lệch xác suất output của cặp cũng bằng 0 tại byte 127.

Đây là kết quả trên trajectory đã khóa, không phải mọi prompt. Reset chỉ xảy
ra trong bản sao chẩn đoán. Pooled context không giữ thứ tự nên không phục hồi
được phân biệt đã mất. Tăng ticks không tự sửa vấn đề này.

## 4. Giữ activation của cùng core vẫn chưa tạo lợi ích trả lời

[Vòng 2](../../runs/20261109T111202Z-byte-working-memory-v2-288db8/report.md)
giữ 48 features cuối encoding, tổng 192 byte, trong một response. Readout đọc
96 giá trị fast + held. Có thêm 12.864 synapse readout, được tính rõ; không thêm
neuron hay model. Snapshot chỉ là bộ nhớ làm việc, không phải kiến thức dài hạn.

| Condition | Dev exact | Train exact | Dev byte accuracy |
|---|---:|---:|---:|
| Không snapshot | 0% | 0% | 23,01% |
| Snapshot/raw | 0% | 0% | 19,87% |
| Snapshot/chuẩn hóa | 0% | 0% | 10,98% |

Ablation xóa hoặc đổi snapshot cũng 0 exact; chưa có bằng chứng snapshot giúp
giải tác vụ. Lưu và khôi phục đúng không làm kết quả này thành learned memory.
**Dừng nhóm byte-dialogue sau hai vòng.** Không thử tiếp cùng họ dưới tên mới
hoặc deploy một candidate thất bại.

## Quyết định

Mục tiêu hội thoại và vision bình thường còn thiếu representation, temporal
credit assignment và acquisition/retention. Giữ **PIVOT_NO_SCALE**. Một giả thuyết
cơ chế mới phải có review và protocol riêng theo P1/P2, không chỉ thêm dữ liệu.
Phép thử ảnh thật tiếp theo được báo riêng; không suy từ valid UTF-8 sang khả
năng nhìn hoặc trả lời về ảnh.
