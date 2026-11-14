# Sửa hai finding đã xác nhận — video request ownership và snapshot

**Đã sửa và kiểm luồng thật; chưa tuyên bố production hoặc neural PASS.**
Giữ ALIF/local learning, VLM/ASR nguyên precision, một model ID và một gateway.
Không train, tạo dataset, đổi threshold hoặc chạy lại test 8 giờ.

## 1. Video ở draft mới có chủ sở hữu rõ ràng

Trước sửa, một ticket upload chưa có session có thể bị bất kỳ chat nào có
cùng text digest lấy trước. Đã bỏ đường nhận video bằng text matching.

Đường mới sử dụng identity thật:

```text
Composer tạo submission_id và giữ nó khi retry
    → Hermes tạo/khôi phục đúng session, kiểm drift
    → callback onSessionPrepared của chính lần gửi
    → upload/queue metadata với session + submission_id
    → prompt.submit commit cùng submission_id
    → canonical executor bind row.request_id bằng ContextVar
    → model request mang X-SCARX-Submission và X-SCARX-Session
    → gateway chỉ consume ticket trùng cả hai identity
```

Callback generic nằm đúng seam đã resolve session trước admission, không đọc
global focus sau await. Nếu prepare lỗi hoặc người dùng đổi chat, không submit;
đường restore của Hermes giữ bản nháp/attachment và retry ID. Metadata không
chèn vào lời người dùng hoặc ảnh hưởng thông điệp tool history.

`VideoContext` từ chối session trống/draft hoặc thiếu submission, giữ tối đa
16 pending ticket có TTL. Pending và binding theo lượt gửi được lưu SQLite;
consume/focus/receipt cùng transaction. Một request đã bind không thể đổi sang
video khác; retry vẫn nhận đúng video của chính request đó dù focus đã đổi.
Các binding v2 đã tồn tại được giữ để không phá lịch sử, nhưng không tạo mới
unscoped ticket v2. Không mở rộng claim thành phân quyền nhiều người dùng:
đây vẫn là app/gateway local cùng chủ máy.

## 2. Snapshot không ghi đè câu hỏi khác

Screen trước đây luôn ghi `question-frame.jpg`; video extraction cũng dùng
cùng tên trong mỗi stream. Đã thay bằng `question-frame-<SHA256 JPEG>.jpg`.
Hai ảnh khác có đường dẫn khác; ảnh giống nhau tái dùng bytes immutable.
Publication dùng temporary file đã fsync, trên Windows rename không ghi đè;
nếu writer khác đã công bố cùng hash thì đối chiếu bytes. Nhờ vậy một reader
giữ ảnh mở không làm writer overwrite gây sharing violation.

Video FFmpeg có temporary output riêng rồi publish content-addressed. Quota
archive tính cả cache mới; cleanup chỉ xóa file do runtime sở hữu, không xóa
file gốc của người dùng. Tests chạy đồng thời 32 requests và kiểm ảnh thứ nhất
không đổi sau snapshot nguồn thứ hai; TTL/quota cũng được kiểm.

## 3. Lỗi mới tìm thấy khi chạy UI và bản sửa

Lần gửi video đầu tiên qua UI đã bind đúng request nhưng gọi model trước khi
có frame. VLM đi tìm web từ text observations rỗng và bị guard dừng. Giữ trace
thất bại; không gọi phần đó là thành công hoặc xóa lịch sử để chấm lại.

Đã thêm chờ quan sát local đầu tiên trước admission, tối đa 35 giây; timeout
trả lỗi có nghĩa và giữ draft. Native call trả lời media local không nhận tools,
để observations của video riêng không thành automatic web/remote tool query.
Đây là policy điều phối, không phải neural policy đã học tự tránh lỗi.

## Bằng chứng

- **256 Python checks qua**, gồm CUDA/CPU parity, persistence, rollback, đồng
  thời snapshot, session/submission ownership, restart và transaction failure.
  Warning CuPy CUDA_PATH không tự detect vẫn được ghi; pinned DLL loader và
  CUDA numerical tests qua. Không có skip để che case lỗi.
- **29 Node checks qua**; thêm hai draft cùng text nhưng khác admission, stable
  upload ID sau renderer reload, đúng callback trước bind và giữ draft khi lỗi.
- **12 Hermes Desktop tests qua**, dùng Vitest thật: chờ metadata sau tạo chat,
  không admit khi lỗi/đổi phiên, retry cùng ID, first-send/queue/composer contracts.
  Renderer build với `--typecheck` thành công. Không thay Electron main/preload.
- **UI Hermes thật**, chat mới gửi video qua Files: canonical DB có request ID
  đúng binding, duy nhất một user text giữ nguyên; frame local có trước bind,
  VLM nhận đúng nguồn, không có tool call ở lượt đã sửa. Model trả màu đỏ bên
  trái/xanh bên phải. Nhưng trả tiếng Anh và chưa dẫn timestamp: còn lỗi chất
  lượng, không gọi là hoàn thành mọi yêu cầu của câu hỏi.
- [Audit máy đọc được](evidence/MEDIA_ADMISSION_REPAIRS_20261114.json) đọc lại
  actual events, canonical admissions/transcript, renderer hashes và checkpoint
  receipt. Audit không gọi model hoặc capture thêm.

JUnit/log/candidate/before-source/checkpoint ở `local/`. Public regression
fixture của Hermes được giữ tại `integrations/hermes/media_admission_submit.test.tsx`;
source patch trong `release/hybrid/HERMES_MEDIA_ADMISSION.patch`. Các mốc lỗi
và receipt cũ giữ nguyên. Không sửa main/R13, merge hoặc push.

## Triển khai và phạm vi gói

Đã nạp runtime/plugin/generic metadata seam và renderer đã build trong deployment
hiện tại, với backup source/renderer và saved state. Toàn mảng snapshot, RNG,
replay/metadata và 5 live updates giữ qua activation. Không nhập lab weights.

RC6 là overlay cho **deployment đã cài bản Hermes tương ứng**. Asset pins bao
gồm source gateway và renderer hiện tại; installer từ chối app khác fingerprint.
Source patch đi kèm để bảo trì, không phải auto-updater của mọi phiên Hermes.
Không thêm gateway/service mới hoặc mã model khác.

Tính tổng quát ngôn ngữ/vision, lợi thế SNN so với cùng LLM + RAG, credit
assignment/retention và tự học nguyên nhân lỗi vẫn chưa được chứng minh.

## Antigravity follow-up

Đã giao riêng review source hai bản sửa bằng cùng 3.8 Flash high/effort high,
chỉ source/tests công khai. Lượt này timeout 240 giây, không có kết luận cuối;
không tính là approval. Review 6 finding trước vẫn giữ nguyên; closure của hai
finding ở đây dựa vào repro hồi quy và UI/admission/journal thực tế đã audit.
