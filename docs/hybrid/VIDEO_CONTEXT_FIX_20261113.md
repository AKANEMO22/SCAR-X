# Sửa tham chiếu video trong chat13/11

Nguyên nhân: `attachVideos` đọc`video:<session>` (hoặc một khóa`video:draft`
chung) rồi nối literalMarkdown`[Video: ...](scarx://stream/...)` vào`draft.text`.
Nút chọn nguồn trong Debug từng cho phép giữ fixture làm nguồn cho lượt sau.
Backend lại scanURI trong toàn history, nên chỉ xóa đoạn chèn ở frontend chưa
đủ ngăn stale fixture trở lại.

Sửa trong nhánhHybrid, không main/SNN/weights:

- Plainturn trả đúng draftobject cũ, không đọc stickyselection; legacycurrent/
  draftkeys được bỏ khi reload và trước submit. Debug không gắn video vào chat.
- Freshnativeupload tạo contextticket bằng metadata trên endpoint`/api/observe`
  có sẵn, không cần restart Hermes hoặc thêm service/proxy. Text gốc
  chỉ dùng để tạoSHA256, không chèn markup. Renderer gửi streamID, digest,
  sessionaliases và ticketID bằng kênhRESTprivate, ngoài messagebody.
- Chat đã có session chỉ nhận ticket đúng session và digest. Draft đầu chưa có
  runtimeID được consume một ticket còn hạn khớp digest; nhiều ticket giống
  text bị từ chối, không tự chọn video. Ticket300s/tốiđa16; binding12h gắnsession
  lưu riêng trongSQLitev2. Không có binding`local`/`draft` dùng chung.
- Backend không suy routing từ`scarx://` trong user/history nữa. Existing links
  giữ như nội dung lịch sử, không cấp quyền dùng nguồn. Screenbindings chỉ còn
  hiệu lực khi nguồn thực sựstarting/sharing; stoppedsource không tựtheo lượt.
- Modelrequestcopy vẫn có thể kèm bằng chứng video được chọn. Transcript và
  nội dung người dùng đã lưu không bị sửa. Không xóa/cải biên audit cũ.

Regression: chạy middlewareJS thật quaNodeVM, gồm staleengineeringfixture,
text nguyên byte, nativevideo+ảnh, firstdraftkhôngglobalkey, lỗi uploadkhông
mấtdraft. Python kiểmmetadata/session/TTL/ambiguousdraft/restartbinding và
legacyURI/stoppedscreen không lấy nguồn. Không train hoặc tạobenchmarkmới.

Giới hạn: ticketfirstdraft chưa có runtimeID dùng exacttextmatch ngắn hạn;
ambiguouspending failclosed, thiếu identity không dùngglobalcontext. Khi
gatewayrestart trước submit làm mất pendingticket, cần gửi lại attachment.
Hội thoại cũ vẫn hiển thị đoạn đã lưu; bản sửa chỉ ngăn phát sinh mới và routing
sai. Đây là lỗi tích hợp, không phải hành vi neuron hoặc model tự thêm video.

Đã cài vàoHermes và gateway được nạp lại, không restartHermes/core hay đổi
modelweights. Receipt`20261113T133713Z-hybrid-video-context-fix-899c36` xác nhận
qua canonicalHermesRPC: tin do agent gõ trong phiên riêng được persist đúng
nguyên văn, không cólegacyfixtureURI.46Pythonaffected/native/catalogchecks và
4Nodecomposerchecks qua; lượt test riêng đã dừng, không đểmodel chạytoolloop.
