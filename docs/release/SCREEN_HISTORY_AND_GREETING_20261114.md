# Màn hình đã xem và marker trong lời chào

## Hai lỗi đã tái hiện bằng runtime thật

**Mất liên kết sau dừng màn hình.** Capture window chạy được nhưng watch-request
không truyền session ID cho stream. Liên kết chỉ ở RAM và bị bỏ khi dừng, nên
câu hỏi “cửa sổ vừa xem” không chọn lại nguồn, dù frame và mô tả vẫn nằm trên SSD.

Đã truyền đúng session ID và lưu `screen_context_v1`/`media_focus_v1`. Phiên chat
có thể hỏi rõ về màn hình đã xem sau stop/restart; không cần tiếp tục ghi hình.
Khi người dùng chọn video upload mới, nguồn đó được ưu tiên. Câu hỏi không liên
quan không nhận ảnh cũ; history/tool output không tự cấp phép capture mới. Nếu
lưu binding thất bại, worker vừa mở được dừng. Đây là liên kết/truy xuất bằng
chứng ở runtime, không phải kiến thức đã chuyển vào synapse.

**Lời chào lặp marker Hermes.** Người dùng gửi `chào bạn`, model trả
`[OUT-OF-BAND USER MESSAGE — Chào bạn]`. DB xác nhận chuỗi do VLM sinh, không
phải renderer tự thêm. System prompt Hermes chứa ví dụ `STEER_CHANNEL_NOTE`.
Với đúng prompt và weights cũ, lỗi tái lập; bỏ riêng briefing steering không
dùng ở lượt xã giao thì cùng model trả lời bình thường.

Đã sửa helper host chỉ cho lời chào thuần túy: bỏ schema tool dư và briefing
steering **khớp nguyên văn của Hermes** trong role system. Không sửa văn bản
người dùng, history hay output để che lỗi, không hardcode câu chào. Tác vụ thật,
tin nhắn steer thật và câu hỏi giải thích marker giữ protocol/công cụ cũ.

## Kiểm chứng màn hình

- Chỉ mở cửa sổ FFplay do script tạo, chạy clip engineering đã biết, tắt audio.
  Bắt đầu qua tool `scarx_video` trong dispatcher Hermes đang cài, sau request
  chat có session ID; request chưa cấp quyền bị từ chối.
- Bản trước: capture thành công nhưng câu hỏi sau stop không có selected stream.
  Bản sửa: 13 frames ghi, 2 frames được VLM mô tả, capture frontier 12,438s và
  vision frontier 8,360s. Các thời gian tính từ lúc share, không phải player time.
- Stop sạch, rồi hỏi lại hình bên trái lúc đầu: model trả “màu đỏ” từ frame đầu.
  Sau gateway restart vẫn chọn đúng nguồn và trả cùng màu; không recapture.
- Một câu không liên quan không nạp ảnh cũ. Không còn capture worker/player thử.
  Không chụp cửa sổ riêng tư, không dùng speaker loopback hoặc microphone.
- Learned weights, traces, importance, RNG/replay và updates=5 giữ nguyên.
  Snapshot trước shutdown và final save có 16 bước forward phát sinh, làm đổi
  voltage/adaptation/spikes/features. Không gọi là full byte-identical snapshot.

[Audit màn hình](evidence/SCREEN_HISTORY_20261114.json) đọc actual receipts,
journal và metadata; không gọi model hoặc chụp thêm trong audit.

## Kiểm chứng lời chào

- Baseline native với prompt Hermes: lặp đúng marker trong ảnh. Bản bỏ ví dụ
  dư: “Chào bạn! Tôi sẵn sàng hỗ trợ. Bạn cần gì?”. Không đổi weights.
- Sau cài helper/reload riêng gateway Hermes, ba lượt canonical mới (`chào bạn`,
  `xin chào`, `hello`) đều không sinh marker. Lượt `chào bạn` trả lời bình thường
  bằng tiếng Việt. `xin chào` còn trả tiếng Anh, `hello` còn hỏi mục tiêu quá cứng;
  ghi rõ chất lượng xã giao/ngôn ngữ của model nhỏ chưa hoàn hảo.
- Khi phát lại history đã có câu sai, retry native trả “Chào bạn!”, không cần
  xóa hội thoại hoặc sửa câu trả lời cũ trong DB.
- 22 checks host-route/greeting qua; actual tool tasks vẫn giữ steering note.
  Các kiểm tra source/persistence và bộ QA độc lập được ghi ở biên bản riêng.

Raw prompt/log và checkpoint ở `local/`, không gửi cho Antigravity. Antigravity
chỉ nhận source/tests công khai và kết quả fixture không có dữ liệu riêng.

## Giới hạn

Đã kiểm gateway/tool dispatcher/capture thật, không phải toàn bộ thao tác UI
Desktop. Clip ngắn đã biết không là video benchmark hoặc chứng minh tổng quát.
Không kiểm audio loopback live trong lượt này. Trí nhớ SSD không đồng nghĩa
hidden synapses học kỹ năng; SNN advantage, retention và learned error causality
vẫn chưa đạt chứng minh. Test 8 giờ giữ trạng thái đã hủy theo yêu cầu.
