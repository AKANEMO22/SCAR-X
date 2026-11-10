# ADR0010 — Tab Hỏi để học và ba chế độ hoạt động

2026-11-09. Người dùng yêu cầu tab Hermes riêng cho SCAR-X chủ động hỏi và
người dùng nhắn/dạy. Fast/deep/idle là ba chế độ cùng core, chưa phải ba lớp vật lý.

- Dùng disk plugin desktop/backend namespace hiện có; route `/scarx-learning`,
  nav “Hỏi để học”, indicator số câu chờ. Không thêm framework/model; thông báo
  không tự đổi route hoặc chiếm focus.
- Câu hỏi từ chat/idle cùng feed. Reply gắn đúng questionID/source sessionID;
  có bỏ qua, nhắn trước và dạy cặp tình huống–đáp án. Nguồn controller khác neural.
- `extra_body.scarx_question_surface: learning_tab` tắt watcher inline cho
  provider này. Main chat chỉ routing notice khi chưa trả lời được, không giả
  neural skill. Transcript cũ giữ nguyên.
- Plugin dùng auth Hermes, chỉ kết nối SCAR-X loopback đã cấu hình. Mutation
  token ở Python backend; không gửi renderer/log, không follow redirect.
- SQLite correspondence giới hạn512tin, feed200gần nhất + mọi câu hỏi open.
  ClientID dedup trong lịch sử còn giữ. Label/learning/RNG/core snapshot như cũ.
- Transcript là audit, không là neural knowledge/retrieval. Neural rollback
  giữ log; lesson state hiển thị theo snapshot hiện tại. Full-folder backup gồm
  log; neural-export ZIP không chứa toàn bộ correspondence history.

Deep Thinking/Adaptive Computation Time và tự khám phá có verifier vẫn Proposed;
idle review đã có thành phần. Xem tài liệu12/master P0–P5. Không học draft chưa
xác minh, tự gọi teacher/cloud vô hạn hoặc đổi forgetting gate.

Tests kiểm wrong label, nguồn reply, idempotence/restart/bound history, draft
retry và real plugin discovery/auth → SCAR-X HTTP. Native UI proof ghi riêng.
