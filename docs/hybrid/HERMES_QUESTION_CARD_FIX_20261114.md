# Sửa bố cục Hỏi để học trong Hermes — 14/11/2026

Ảnh người dùng cho thấy câu hỏi, input và bốn nút nằm dính trên một dòng.
Nguyên nhân: LearningQuestion ở composer nằm ngoài root .sxh, trong khi CSS cũ
chỉ style input/button trong root đó; widget có mỗi border/padding inline.

Đã thêm CSS riêng .sxq, không sửa CSS của chat Hermes:

- Header nhỏ, nội dung câu hỏi và textarea full width ở các dòng riêng.
- Footer có khoảng cách, tự xuống dòng khi hẹp; chỉ hai action chính Bỏ qua /
  Gửi trả lời.
- Rating hữu ích/chưa hữu ích dùng icon có tooltip, aria-label, aria-pressed.
  Chọn rating chỉ đổi draft; gửi mới có feedback. Trả lời thường và Bỏ qua
  không tự tạo helpful label, kể cả trước đó đã chọn rating.
- Ctrl/Cmd+Enter gửi, Enter thường xuống dòng. Gửi rỗng bị chặn; draft được
  giữ nếu API lỗi và được reset khi pending question đổi ID.

11 Node checks qua, gồm composer/video/trace và rating semantics. Installer có
backup20261114T023352434712Z và hot reload; đã quan sát trực tiếp đúng Hermes:
header/câu hỏi/textarea/footer tách rõ, hai action và hai icon hiện đúng.
Không gửi câu trả lời hay bỏ qua pending question thật để test.

Không restart model/gateway, không đổi synapse/feedback API. Lượt endurance8h
vẫn giữ source/hash; thay đổi chỉ ở plugin renderer và tests. Ponytail không có
tool/context được cung cấp trong phiên, không giả định đã nạp.
