# 0012 — Bảng quan sát tự theo dõi công việc thực tế

Ngày2026-11-09. Người dùng làm rõ: không chọn model hay bấm chạy; khi Codex thực
hiện công việc trên SCAR-X thì kết quả phải tự xuất hiện trên localhost.

Thay phần operator controls trong UI củaADR0011 bằng passive follow. Giữ
historical measurements và protocol; API authenticated vẫn phục vụ runner.
GET chỉ đọc, không tự khởi động inference/training/teacher/job.

Inference log được tạo từ kết quả `Generation.chat` và lưu cùng transaction
runtime. Learning log từ return loss thật của `train_pair`, weight differences
và guard verdict thực, không đổi update rule/threshold/order. Rollback giữ0ΔW
trên model, lưu attempted metrics bên ngoài neural state. Log observability
không được dùng làm nhãn hoặc tự nhập vào synapses.

Chỉ model đang dùng có scope “Lõi đang dùng”; registered runners phải khai rõ
scope, checkpoint và input. Qwen saved measurements ghi archived, timestamp và
truncation; chưa đo thì thiếu. Target không đi vào inference; synthetic sample
không tự là chân lý. Không dịch trace thành độc thoại “nó nghĩ gì”.

Acceptance24 actual serving inferences + một verified integer lesson dùng
controller/guard hiện có, không nghiên cứu thêm stopped families hoặc mở rộng
model.17 updates được giữ không chứng minh hidden representation/retention.
Existing93 rejection không có metric mới vẫn giữ nguyên lịch sử.
