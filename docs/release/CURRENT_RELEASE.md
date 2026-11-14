# SCAR-X Hybrid — RC5 cùng các bản sửa đang QA

**Cập nhật mới nhất:** [Antigravity QA và triage](ANTIGRAVITY_FULL_QA_20261114.md).
238 Python/27 Node/3 HTTP checks qua trên bản sao; Antigravity review source đã
xong, nhưng tự chạy test bị sandbox chặn. Codex xác nhận lỗi P1: video ở draft
chưa có session có thể bị chat khác cùng text lấy nhầm. Chưa approve production;
không xem RC5 ZIP/receipt lịch sử bên dưới là bản khớp mọi sửa candidate hiện tại.

Lỗi marker trong ảnh đã sửa và kiểm qua Hermes thật. Screen history sau stop/
restart cũng đã sửa và kiểm trên cửa sổ thử. Xem
[phạm vi và giới hạn](SCREEN_HISTORY_AND_GREETING_20261114.md). Draft ownership
và snapshot dùng chung path còn cần xử lý trước khi chốt gói tiếp theo.

**Dùng trong chat Hermes bình thường**, model **Scarx Hybrid**, ID
`scarx-hybrid`. Một gateway `127.0.0.1:8775`, một worker Qwen3.5-0.8B nguyên
BF16/F32, Whisper-small FP32 và ALIF H128 học cục bộ trên GPU. Không LoRA,
quantization, pruning hoặc thay cơ chế học.

Gói: `artifacts/releases/SCAR-X-Hybrid-1.0.0-rc.5+83c5fd2bc936.zip`,
801.428 byte, 51 payload files. SHA256:
`01136cb5cb2cfd8c5327d706fd44bbaee2c3fc67d3761d185166547f553f1f8d`.
Git SHA nguồn gói: `83c5fd2bc936458de2bb3585a66261fb818f267a`.
Sidecar `.installed-verification.json` xác nhận payload khớp code đang cài,
asset/Hermes/dependency pins khớp, không restart hoặc gọi model khi kiểm cuối.
Main/R13 giữ nguyên; không merge hoặc push. Các gói/receipt cũ được giữ riêng.

## Dùng ngay

- Chọn **Scarx Hybrid** trong chat Hermes như model trước đây.
- Ảnh/video dùng menu đính kèm sẵn có, hỏi tiếp trong cùng chat. Video được
  lấy mẫu hình và phiên âm từng đoạn; không đảm bảo thấy mọi sự kiện.
- Muốn xem video màn hình: yêu cầu trong chat và chỉ rõ cửa sổ cần xem. Không
  mặc định theo dõi toàn màn hình hoặc microphone.
- Trace: `http://127.0.0.1:8775/?tab=trace`. Debug chỉ quan sát.
- Đây là gói cập nhật code cho deployment Windows/Hermes đã cài trên máy này,
  không chứa trọng số, venv, hội thoại hay checkpoint riêng; chưa là bộ cài máy mới.

## Đã sửa và kiểm

| Luồng | Bằng chứng và phạm vi |
|---|---|
| Chat/video Hermes | Lời chào trả lời bình thường; clip engineering 21 giây được đính kèm qua Files, trả đúng màu đỏ/mốc 5 giây. Mã transcript đi qua metadata, không chèn link vào lời người dùng. |
| GPU/persistence | VLM/vision, ASR, ALIF trên GPU; 3 startup parity cases và 3 ASR FP32/CUDA calls thật qua. Các mảng/RNG/replay giữ qua restart, live updates vẫn 5; không nhập lab weights. |
| Search | Dispatcher Hermes + SearXNG native trả được tài liệu chính thức; không API trả phí/Docker. |
| Tool loop | Observer đọc đúng `request.body.tools`; actual dispatcher cho thấy 0→1 probe đọc lại README ở đúng project sau 3 lỗi. Guard replay chặn lần giống hệt thứ tư. Đây là runtime repair, chưa là neural learning. |
| Chat dài/workspace | Query FTS không còn làm tin nhắn 7.028 ký tự bị HTTP 400. Project trung lập được đọc bằng runtime prelude; VLM trả đúng mục đích, receipt và port. |
| Compact | 5 lượt tiếp theo ×7.115 ký tự kích hoạt 3 auto compact; manual compact 8→6 message. Owner display giữ 7 đoạn user nguyên văn, raw có 4 handoff rows nhưng display có 0. Hỏi sau manual trả đúng code/port, không gọi tool. |

**171 Python checks qua**, gồm streaming/non-streaming, observer và CPU/native
parity; bộ Node 27 checks của giao diện/plugin giữ làm bằng chứng tương ứng
với bytes không đổi. Native startup có kiểm thực, không suy chất lượng từ unit test.

Chat dài chia query truy xuất thành phần ≤6.000 ký tự và pool ≤32 ứng viên;
**tin nhắn gửi VLM giữ nguyên**. Trần xử lý 65.536 ký tự không phải context 65K
token. Context vẫn **16.384 token**, gồm system/history/tools; không tăng model
hoặc nới cửa nghiên cứu để đạt các kết quả này.

## Bằng chứng và bảo trì

- [Chat/video/GPU](evidence/RC3_OPERATIONAL_SMOKE_20261114.json),
  [observer](OBSERVER_ENVELOPE_REPAIR_20261114.md),
  [chat dài và compact](LONG_INPUT_AND_COMPACTION_20261114.md).
- `scripts/activate_hybrid_rc.py`: backup/CAS và kiểm learned state;
  `scripts/verify_installed_hybrid_rc.py`: đối chiếu artifact với deployment.
- Hai acceptance scripts dùng canonical gateway trên project tự tạo;
  `scripts/audit_hermes_context_acceptance.py` đọc lại receipt, không gọi model.
- Raw prompt/checkpoint/log nằm ở vùng private. Không ghi đè receipt thất bại,
  xóa journal hay rollback toàn database để làm phép kiểm trông thành công.

## Giới hạn còn lại

- Model 0.8B còn yếu; clip/project đã biết không chứng minh năng lực đa năng,
  chất lượng video chưa thấy hoặc ngang model online.
- SNN chưa chứng minh vượt cùng LLM + RAG, retention chưa đạt ngưỡng đăng ký,
  chưa tự học chắc chắn nguyên nhân tool loop hoặc kỹ năng sửa lỗi lâu dài.
- Guard/retrieval/host tool routing/compact không được ghi thành hidden-neuron
  improvement. LoRA tự động và learner mới vẫn tắt/backlog.
- Save/load/rollback state đã kiểm cô lập và giữ state live qua restart; không
  thử rewind cuộc trò chuyện riêng của người dùng. Code rollback không ghi đè DB.
- Đã kiểm canonical owner display projection, chưa chụp lại toàn UI Desktop
  hoặc nghiệm thu mọi luồng capture màn hình/project/tool. Không gọi full UI PASS.
- Lượt small8h `32cd13` đã hủy theo yêu cầu sau 11.268,281 giây: giữ 38 chu kỳ,
  7 checkpoint, không worker sót. `USER_CANCELLED_NOT_EIGHT_HOUR_PASS`, không chạy lại.

Đây là **RC dùng thử có giới hạn**, không phải production hoặc lifelong PASS.
