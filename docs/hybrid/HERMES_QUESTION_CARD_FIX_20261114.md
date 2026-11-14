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

## Sửa nút gửi sau phản hồi của người dùng

Card ban đầu dùng form lồng vào composer form của Hermes. Nút submit phụ thuộc
browser form dispatch, có thể không gọi endpoint feedback; source host xác nhận
slot top nằm trong form chat. Không nhận request question_answered ở log lúc
kiểm; question đã được chủ máy Bỏ qua trước khi kiểm UI, không giả làm đã nhận
nội dung người dùng đã gõ. Thay bằng region/div và button typebutton có onClick
trực tiếp; chặn propagation của shortcut textarea để không gửi nhầm chat.

Ref guard chặn doubleclick đồng thời. Card giữ draft và lỗi inline khi HTTPfail,
chỉ xóa khi máy chủ accepted/skipped rõ ràng. Receipt “Đã gửi câu trả lời cho
SCAR-X” còn hiện sau refetch, có nút đóng; đóng receipt không làm hiện lại câu
hỏi đã nhận nếu status cache còn cũ.

Regression dùng chính JS handler của plugin, HTTPloopback thật với HybridEngine
và SQLite trong tempdir: lần1controlled503 giữ draft/lỗi, lần2accepted, doubleclick
không gửi trùng, không tạo helpful ngầm hoặc neuralupdate. Test đỏ trên phiên
b809d1b trước sửa, xanh sau sửa.11Nodechecks +1HTTPintegration qua.
Installerbackup20261114T024632706683Z/hotreload, kiểm installedbytes khớpsource,
Hermes backendready. Không tạo/gửi phản hồi thử vào memorylive để kiểm.

## Đồng bộ đường gửi trong Debug và chat — 09:24 UTC

Tab Debug còn dùng bộ xử lý riêng: không kiểm tra accepted/skipped, không có
receipt, lỗi ở đầu trang thay vì cạnh câu trả lời. Đã thay bằng cùng
LearningQuestion của composer; Debug vẫn dùng được khi model đang chọn khác.
Nút gửi trực tiếp, guard double-click, giới hạn chờ 20 giây và giữ draft khi
lỗi áp dụng chung. Mỗi nội dung/ID/rating có client_id ổn định để thử gửi lại
sau mất xác nhận nhận được receipt đã lưu, tránh học/ghi nhận trùng.

Run `20261114T092354Z-hermes-question-send-v2-a3bd7a` giữ source JS thực sự
đã cài, hash, ba log HTTP và receipt. Ba ca chat/Debug/mất receipt sau commit
đều qua trên HybridEngine + SQLite cô lập. Phần cache receipt trong harness
là transport fixture theo hợp đồng gateway, không là model response hay dữ
liệu huấn luyện. Mỗi ca chỉ có một question_answered, 0 cập nhật neural.
7 composer checks và 14 trace checks cũng qua.

Computer Use xác nhận trên Hermes thật: nhập chữ làm nút Gửi trả lời bật,
xóa bản nháp kiểm tra làm nút tắt. Không bấm gửi dữ liệu thử vào câu hỏi thật.
Installer chỉ thay plugin.js, backup `20261114T092124122472Z`; không restart
gateway/model, live updates vẫn 5 và câu hỏi thật còn pending. Log cũ có
Unknown answer/question identity; chưa khôi phục được nội dung người dùng từng
gõ hoặc tái hiện chính xác lần bấm lỗi trên UI live. Đây là sửa/kiểm đường gửi,
không tuyên bố đã nhận câu trả lời đã mất hoặc synapse học thêm.
