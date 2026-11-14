# Hidden đã đổi representation, nhưng chưa có lợi thế tổng quát đáng tin cậy

**Quyết định giữ PIVOT.** Hai chẩn đoán mới tách được thay đổi hidden khỏi
readout, nhưng chưa chứng minh representation mới giúp học kỹ năng chuyển
sang dữ liệu mới. Không promote protection hoặc thay model Hermes.

Đây là kiểm tra sau [screen20chặng](PREDICATE_RETENTION_DEV_20261114.md), trên
cùng dữ liệu owned predicate, không phải benchmark video/ngôn ngữ hoặc một
confirmation TEST mới. Các ngưỡng acquisition65%, forgetting3/5pp, gain5pp và
lowerCI>0 không thay đổi. Không có bước backward, BPTT, cập nhật VLM hay LoRA.

## Giữ bản ghi, đổi truy vấn

[Protocol trước output](../../experiments/hybrid-predicate-binding-diagnostic-v1.md),
[run c1f23f](../../runs/20261114T044643Z-predicate-binding-diagnostic-v1-c1f23f/result.json),
[auditor dữ liệu](../../runs/20261114T044643Z-predicate-binding-diagnostic-v1-c1f23f/independent-contrast-audit-v1.json).

663 cặp truy vấn có đáp án trái nhau, cùng một candidate record. Có394cặp đổi
chủ sở hữu,200cặp đổi điểm thời gian và69cặp đổi từ khóa.31 trường hợp không
đủ điều kiện được ghi trước inference:6timestamp ở tương lai,25body chứa đủ
bốn từ màu nên không có từ màu vắng mặt để đối chiếu. Không bỏ lỗi dự đoán.
Nhãn từ SQL và predicate Python riêng; encoder không nhận nhãn/comparison flag.

Tỷ lệ **đúng cả hai câu trong cặp**, gộp3seed (các cặp và seed không độc lập):

| Checkpoint | Đổi chủ sở hữu | Đổi thời gian | Đổi từ khóa |
|---|---:|---:|---:|
| Frozen |3,64%|1,67%|1,93%|
| Readout-only |12,01%|2,67%|9,18%|
| Hidden hiện tại |12,44%|4,17%|7,25%|
| Hidden + protection |10,32%|4,17%|9,66%|

Đây không phải accuracy của từng câu; một model luôn trả “khớp” có thể đạt50%
trên hai câu cân bằng nhưng đúng cả cặp0%. Model hiện thường không đổi nhãn
đúng hướng khi quan hệ thay đổi. Không có exact opposite-label input alias
trong663cặp này: có khác biệt input chưa đồng nghĩa model đã biết dùng nó.
Hai encoded aliases của400finalDEV gốc vẫn được giữ và công bố riêng.

Đảo hai block8tick ở112cặp đầu tiên mỗi nhóm được lưu trong raw. Đây là format
perturbation ngoài phân bố huấn luyện; thay đáp án sau đảo block không tự chứng
minh hiểu quan hệ thời gian. Các query thời gian là point queries có chủ đích,
không phải bài tổng quát hóa toàn bộ khoảng thời gian hoặc video.

12checkpoint cố định,18600CUDAforwards,178,985s. Không học hoặc fit thống kê.
Runner kiểm weights/stable/importance/statistics/replay/RNG/counters không đổi;
auditor chấm lại nhãn, hash, distribution/argmax và toàn bộ metric từ artifacts,
không giả làm đã chạy lại inferenceGPU hoặc peerreview bên ngoài.

## Một bộ đọc chung cho các representation

[Protocol](../../experiments/hybrid-hidden-common-probe-diagnostic-v1.md),
[run813109](../../runs/20261114T045315Z-hidden-common-probe-v1-813109/result.json),
[audit tuyến tính độc lập](../../runs/20261114T045315Z-hidden-common-probe-v1-813109/independent-common-probe-audit-v1.json).

Khóa hidden, lấy core.features128 chưa qua readout/f_mean/f_var; cùng320mẫu
train để chuẩn hóa và fit một linear ridge head có cùng công thức/lambda1.
400finalDEV chỉ được chấm sau fit. Đây là readout chẩn đoán offline mới, không
phải learner triển khai hoặc bằng chứng online learning tốt lên.
Ridge truy cập lại toàn320mẫu train; ngân sách khác learner online/replay. Không
lấy điểm fit này làm bằng chứng đã sửa forgetting hoặc có learner mới tốt hơn.

| Representation | Train trung bình | FinalDEV theo3seed | FinalDEV trung bình |
|---|---:|---|---:|
| Frozen / readout-only |83,02%|53,50 /54,50 /53,25%|53,75%|
| Hidden hiện tại |92,08%|54,25 /52,25 /61,75%|56,08%|
| Hidden + protection |85,73%|53,75 /52,50 /51,50%|52,58%|

Hidden−frozen:+2,33pp, paired seed/family CI95% **[-3,08;+8,17]pp**.
Protection−frozen:−1,17pp, CI **[-3,08;+0,33]pp**. Ba seed ít, một seed có gain
8,5pp nhưng một seed giảm2,25pp; không đủ kết luận cải thiện đáng tin cậy.
Không chọn seed tốt hoặc tăng thêm epoch để đổi phán quyết.

Readout-only và frozen cho feature matrices giống byte. Khi chỉ reset w_in/
w_rec của learned checkpoints về ban đầu, giữ nguyên learned readout và online
statistics, feature matrices cũng trở lại đúng byte của frozen. Do đó có thể
khẳng định **hidden updates làm đổi representation**, không quy nhầm thay đổi
này cho readout hoặc normalization. Tuy nhiên lợi thế heldout của representation
mới chưa được xác nhận; khả năng khớp mẫu train tăng mạnh hơn khả năng chuyển
sang nguồn mới. Chưa có shuffled/magnitude-matched control để quy gain cho cấu
trúc tín hiệu học thay vì drift/biến đổi feature ngẫu nhiên.

18representation×720cases=12960neuralforwards,157,281s. Neural weights vẫnFP32,
GPUforward; ridge được giải bằng conjugate gradientsFP64 trênGPU, rồi kiểm độc
lập NumPy closed-form cùng features/labels train. Auditor đọc lại NPZ không
pickle, tự tính lại statistics/weights/scores/predictions/CI; source và parent
checkpoint hashes khớp. Có sensitivity bỏ2encodedaliases trong artifact riêng,
không thay điểm gốc. Không training/rollback/replay vào live.

## Điều này thay đổi hướng xử lý thế nào?

1. Không chỉ chữa forgetting bằng làm synapse khó đổi: protection vẫn chưa giúp
   tổng quát hóa. Một checkpoint cố định với head mới cũng chưa giải tốt các
   quan hệ chưa gặp; lỗi không thể quy riêng cho bị ghi đè readout về sau.
2. Đầu vào query/record có tín hiệu khác nhau nhưng core thường không tận dụng
   được. Cần phân biệt mất thông tin do băm, binding giữa field/chủ thể và
   credit qua thời gian; chưa xác định một nguyên nhân duy nhất.
3. Có thể nói hidden đã học một representation khác và khớp dữ liệu đã thấy
   tốt hơn. Chưa được nói hidden học quy luật hữu ích đáng tin cậy, cải thiện
   LLM+RAG, tự tìm nguyên nhân tool loop hoặc tích lũy kỹ năng lâu dài.
4. Model live giữ nguyên Qwen0.8B native + ALIFCUDA với policy experimental;
   runtime guard dừng lặp là cơ chế phần mềm. Phần VRAM còn lại có ứng viên
   [Whisper-smallFP32 đã cải thiện phiên âm](ASR_CANDIDATE_AND_TOOL_DATA_20261114.md),
   nhưng chưa chứng minh QA video tốt hơn hoặc profile production dài hạn.

Các bảng đang tự hiện ở [trace thật](http://127.0.0.1:8775/?after=7357&tab=trace),
tách score/lab khỏi bộ đếm neural live. Lượt8h554c19 là phép đo riêng còn tiếp
diễn; PID/source/final receipt phải được kiểm lại trước khi kết luận ổn định.

Regression sau thay đổi:251Python/native/CUDA checks,15Node checks và một
fixture reply HTTP/SQLite qua. Đã quan sát trang thật có kết quả/tiến độ lab,
bộ đếm live vẫn5. Các checks này không bổ sung mẫu chất lượng hoặc seed.
