# ADR0009 — Học xuyên hội thoại, idle review và chủ động hỏi

2026-11-09. Người dùng yêu cầu đối chiếu Markdown: không biến sản phẩm thành
những lần huấn luyện lại riêng rẽ; học trong hội thoại, lúc rảnh xem lại, khi
chưa rõ phải chủ động hỏi người dùng trong Hermes.

## Quyết định sản phẩm

- Một learner/core/checkpoint tồn tại xuyên lượt và restart. Các epoch/seed của
  báo cáo trước là đối chứng nghiên cứu, không phải lifecycle đích. Micro-update
  có giới hạn là một lát tính toán, không khởi tạo model hoặc xóa kiến thức.
- Quan sát input và draft neural, lưu nguồn/session. Câu chưa có nhãn không tự
  thành supervised target. Controller hỏi khi ngoài miền đã biết, output không
  ổn định hoặc lệch với nhãn được người dùng cung cấp.
- Hermes dùng clarification UI sẵn có và một watcher opt-in. Phản hồi gắn đúng
  questionID/sessionID; bỏ qua/timeout không là nhãn. Tối đa một pending question
  mỗi phiên; không hỏi lại một mục đã bỏ qua chỉ vì đang idle.
- Lời sửa rõ ràng và đáp án cho câu hỏi trở thành desired-response lesson có
  provenance. Toán nhỏ kiểm lại từ phép toán hiển thị; văn bản do người dùng
  cung cấp vẫn không mặc định là factual truth. Không tin cờ `verified`.
- Giữa lượt có local micro-update; lúc rảnh review và ôn nhãn đã chấp nhận. Guard
  giữ các probe phát triển đang đúng, reject/rollback khi suy giảm. Thay đổi của
  hai draft chưa xác minh chỉ tạo câu hỏi, không dùng draft làm nhãn đúng.
- Guard không chứng minh solved forgetting: có thể chặn cả update hữu ích. Báo
  cả số chấp nhận/từ chối, acquisition và review. G1 chưa có importance hữu ích,
  representation/credit assignment và consolidation-to-skill còn yếu.
- State câu hỏi, lessons/progress/cursor/counters lưu cùng neural/RNG/topology.
  Core/session serializer v2 đọc checkpoint G1 cũ. Reader cũ phải từ chối v2,
  tránh bỏ qua learning-controller/capacity/encoder state. Snapshot≤3MB.

## Đã kiểm, chưa chứng minh

Unit tests kiểm false feedback, session isolation, pending restart, skip/cooldown,
weight updates, guard rejection, DB failure toàn-state và trạng thái RNG/neural
khi tiếp tục. Clone guard có thể reset diagnostic state; live core không reset.
Controller hiện so sánh đầu ra/probe, không giải mã ý nghĩa của “neuron sâu”.

Engineering stream giữ một learner, không epoch/task reset; dùng human-answer
fixture và8bài toán Antigravity đã máy kiểm. `1+1` vẫn chưa học được bền vững
trong guardbudget: phản hồi được ghi, có update thật và rejection, nhưng không
gọi là đã giỏi toán. Hệ có thể hỏi để lấy trợ giúp; nhiều nhãn đúng hơn không
tự sửa credit assignment. Gate research mean≤3/worst≤5pp vẫn không đổi.

## Dataset bootstrap

Antigravity CLI1.5.3 được người dùng cho phép tạo data.240ví dụ sinh,217qua kiểm
tra cấu trúc/leakage;178train/39dev.53bài toán độc lập kiểm được (48ởtrain).
23mục quarantine:1trùng confirmation lịch sử,22chưa parse chắc.164rubricrecords
chưa độc lập xác minh, không tự nạp chúng làm factual targets.

Dataset chuẩn bị nền tảng/curriculum; không thay thế online learning. Subset
verified chỉ dùng trong clone nghiệm thu, chưa thay weights phiên cá nhân bằng
một checkpoint huấn luyện lại. Người dùng có nút tạm dừng học/hỏi ở website.
