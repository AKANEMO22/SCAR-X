# Hermes với backend SCAR-X có trạng thái

SCAR-X giữ kiến trúc, weights, cách học và ngân sách RAM riêng. Hermes cung cấp
giao diện, lưu transcript và vận chuyển input/output. Không đổi SCAR-X để giả
thành LLM 64K, không proxy sang model khác.

## Chế độ `stateful_chat`

Named provider khai báo `interaction_mode: stateful_chat`,
`max_input_characters: 2048`. Adapter Hermes ràng buộc theo cả endpoint và model;
những provider không chọn chế độ này giữ hành vi agent và guard 64K cũ.

- Không nạp tools, memory prompt, context files hoặc background review của agent.
- Không fallback, MoA hay Persistent Search Controller của Hermes.
- Titler của named custom loopback giữ tiêu đề lấy từ tin nhắn, không gửi prompt
  đặt tiêu đề vào core neural để tranh lượt hoặc làm nhiễu trạng thái.
- Gửi đúng user payload mới nhất, giữ ảnh content parts nếu có. Không gắn system
  scaffold, language-policy reminder, task-state graph hay transcript vào wire.
- Lịch sử ở UI/SessionDB vẫn append-only; nó không là prompt neural đã học.
- Bộ đệm 4096 là giới hạn host/giao thức. Trí nhớ/retention/học mới phải đo riêng
  theo [master](../../docs/RESEARCH_PLAN.md#9-learning--memory-metrics).
- Trace suy nghĩ tự sinh của Hermes không là trace neuron SCAR-X. Measured neural
  trace vẫn ở website local; learned reasoning/tool actions chưa được khai báo.

## Cài trên máy đã có Hermes tương thích

```powershell
python scripts/install_hermes_stateful.py
python scripts/install_generation1_provider.py
```

Installer kiểm anchor chính xác, backup source gốc vào
`local/hermes-stateful-backups/<timestamp>/`, ghi receipt/hash và
`stateful-chat.patch` đầy đủ để review. Giữ các thay đổi không liên quan trong
Hermes; không reset Git hoặc ghi đè helper do người khác sửa. Khi phiên bản
Hermes khác không khớp anchor, dừng để review, không ép patch.

**Khởi động lại backend/Hermes** sau sửa Python source, rồi mở một phiên chat
mới. Reload riêng renderer không đủ nếu backend còn giữ module cũ trong RAM.
Gói SCAR-X không tự cài patch Hermes khi chỉ chạy `run_app.py`.

## Kiểm chứng đã có

### Tab Hỏi để học hiện hành

Chạy thêm `python scripts/install_hermes_learning_tab.py`: hai cửa plugin sẵn có
`desktop-plugins/scarx-learning/plugin.js` và `plugins/scarx-learning/dashboard/`.
Backend được bật trong `plugins.enabled`; backup ở
`local/hermes-learning-plugin-backups`. Không cần rebuild renderer; restart
Hermes để nạp Python API, desktop JS có hot reload.

Provider mẫu dùng `extra_body.scarx_question_surface: learning_tab`, không bật
watcher inline. Người dùng có thể reply/bỏ qua, nhắn trước hoặc dạy một cặp đáp
án; các transcript cũ giữ nguyên. Auth Hermes và token mutation SCAR-X đều giữ;
token không xuống renderer. Log512tin khác neural memory; restore không xóa
lịch sử và không mặc định human label là chân lý.
[ADR0010](../../docs/decisions/0010-learning-tab-and-cognitive-modes.md).

### Cập nhật học xuyên hội thoại

Provider có thể opt-in `proactive_questions:true`. Một watcher nhận câu hỏi của
learning controller từ SCAR-X, hiện native clarify và gửi đáp án về đúng session.
Đây là câu hỏi điều phối, không phải output reasoning từ neuron. SCAR-X mới là
nơi chạy guarded micro-update/idle review; background review LLM của Hermes vẫn
tắt. Không tự học nhãn từ bản sinh chưa xác minh.

Standalone question có terminal event `clarify.resolved`, không chờ `tool.complete`
của một LLM. Metadata đi kèm giúp xử lý trường hợp client đã clear pending sau
acknowledgement; requestID ngăn một answer cũ đóng câu hỏi mới. Parent compute-host
mirror cũng clear khi resolved. Installer lưu bản gốc/hash trước từng cập nhật.

Footer hiện **Nạp X ký tự · N neuron**, popover phân biệt active/allocated/max,
input UTF8/images/vector. Đây là lượng nạp từng request, không là capacity kỹ năng.
6 Python integration tests và5 UI tests cho watcher/lifecycle/load đã qua.
Xem [báo cáo hiện hành](../../docs/research/10_DIALOGUE_DATA_AND_ONLINE_LEARNING.md).

### Bằng chứng trước cập nhật học online

[Nonstream](../../artifacts/hermes-stateful-check.json) và
[stream](../../artifacts/hermes-stateful-stream-check.json): real `AIAgent`,
real HTTP đến SCAR-X, 4096 host buffer, zero tools, raw `hi`/`chào bạn`, hai user
turn vẫn còn trong SQLite. Không mock model, không đổi weights hoặc huấn luyện.
Đây là kiểm tra tích hợp, không phải bằng chứng general chat/retention.

Hai invariant tests nằm ở `tests/agent/test_stateful_chat.py` trong Hermes;
dùng runner `scripts/run_tests.sh`. Cùng meta agent-init và title generator có
48 Python tests qua;10 UI tests cho panel/statusbar và chế độ cũ đã qua.
Đã build renderer và cài trên Hermes local, giữ nguyên Electron main/preload.
Backup renderer ở `local/hermes-renderer-backups/`; old hashed assets giữ nguyên.

Hermes desktop đã restart, hai lượt chat trong cùng phiên được quan sát; không
còn lỗi64K, `hi → Hello.`. Lượt `chào bạn → Bạn. chào bạn.` cho thấy chất lượng
core vẫn yếu, không được sửa bằng template. Footer **Backend state** và popover
ghi input2048ký tự, memory horizon/retained skills/learning capacity chưa đo.
[Ảnh kiểm chứng](../../artifacts/hermes-stateful-ui.jpg). Lịch sử hai lượt còn
trong UI. Lần cài trên máy khác vẫn cần restart backend và build/deploy renderer;
không coi một `/v1/models` thành công là chứng minh UI chat chạy được.

Để rollback: dừng Hermes, dùng các receipt theo thứ tự ngược, chỉ thay file khi
hash hiện tại khớp `after`; khôi phục bản `before` được backup. File helper mới
có `before:null` có thể chuyển vào thư mục backup. Không dùng Git reset trong
checkout đang có thay đổi khác của người dùng.
