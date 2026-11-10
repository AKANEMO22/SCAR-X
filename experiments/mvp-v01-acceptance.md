# MVP V0.1 — acceptance sản phẩm, không đổi gate nghiên cứu

Đăng ký trước acceptance run. Người dùng đã phê duyệt Product-first/MVP, chấp nhận model yếu. Forgetting≤3pp giữ trong hồ sơ nghiên cứu, không chặn ứng dụng thử nghiệm.

- Local Windows, không cloud/model download/LLM. Engine: LifelongLearner score_ema, PlasticCore ALIF H48/K8/R12/D8/O2; local rule hiện có.
- A: nhóm tín hiệu0/1 so2/3; B: chẵn/lẻ. Input có cue tín hiệu/task, không có target channel. Rule chỉ kiểm nhãn/chấm điểm.
- 128 mẫu đánh giá/task từ seed800001/800002 tách training generator, có biến thiên cường độ/nhiễu. Chỉ là biến thể4 tín hiệu, không phải khái niệm/ngôn ngữ mới.
- Dự đoán mean4 stochastic passes cố định. Prediction/evaluation không đổi weights, learner RNG, importance hoặc replay.
- Teach single cần label0/1, confirmation và verifier builtin-rule-v1; nhãn trái quy tắc bị từ chối. Auto lesson lấy nhãn từ quy tắc, không từ model output.
- Max1024 fresh updates+32 replay/đợt,15s. Một transaction chứa learner/plasticity-v1, replay, RNG và pointer. Fault rollback giữ RAM/DB nhất quán.

## Demo acceptance cố định trước kết quả

Seed17; evaluate trước; dạyA64×4; evaluate; dạyB64×4 có replay; evaluate. Không cần accuracy100% hoặc forgetting3pp để pass MVP. Xác nhận học thật: A gain≥20pp, readout weights đổi; hidden drift báo riêng, không gọi là representation success. Restore initial phải đưa full state về như cũ; restart learned session giữ scores/hash/RNG.

## Failure/verification checks

Wrong label, missing confirmation, unknown task, NaN/out-of-range, oversized batch/body bị từ chối không mutate. DB failure và interrupted process không được thay committed generation hoặc lưu trải nghiệm lẻ. Checksum lỗi phải phát hiện, không reset model im lặng. UI xem/export/restore checkpoint và raw SQLite experiences.

Write API chỉ loopback+token+origin+JSON. RAM/CPU đo thật; Windows process-commit cap2GB khi Job assignment thành công, RSS/shared pages monitor riêng. Không thay policy OS/pagefile toàn máy. Chạy pytest gồm native parity, API smoke và browser desktop/mobile. QA data-dir riêng; người dùng mặc định local/mvp. Không sửa kết quả Phase2/2.1.

## Build iteration 2 — demo nhiều đợt, ghi trước acceptance tiếp theo

Smoke đầu đạt A50%→64,0625% sau256update: có học nhưng **không đạt yêu cầu gain20pp của demo đầu**. Không ghi run đó thành pass. Rule/encoder/topology giữ nguyên. Vòng xây sản phẩm này đổi hướng dẫn/demo thành hai đợtA64×4 rồi hai đợtB64×4; đây là thay budget demo, không thay threshold20pp hoặc ngưỡng nghiên cứu. UI giữ từng điểm trung gian và người dùng có thể dạy thêm thủ công. Phép đo mới là acceptance theo tình huống đã xem, không phải xác nhận nghiên cứu trên test hoàn toàn chưa từng kiểm. Báo rõ B có thể học yếu và A có thể suy giảm.
