# Antigravity release QA — kiểm tra và phản biện độc lập

**Kết luận: chưa approve production.** Bộ kiểm tra chức năng qua, nhưng có lỗi
gắn video của draft mới cần sửa bằng quyền sở hữu request/attachment rõ ràng.
Không xem test xanh hoặc nhận xét của agent là bằng chứng không còn lỗi.

## Đã thực hiện

Antigravity CLI nhận bản sao 256 file code/tests công khai, dùng
`gemini-3.8-flash-high`, effort `high`. Init xác nhận đúng model. Không gửi
hội thoại, video riêng, token, checkpoint, model weights hoặc dataset. Bản sao
không đổi source qua hash trước/sau. Main, R13 và test 8 giờ không bị can thiệp.

CLI thử lệnh test nhưng sandbox từ chối ghi thư mục kết quả. Lượt đầu bị dừng
vì lặp chẩn đoán quyền; lượt tiếp có phạm vi directory rõ vẫn bị từ chối và
chuyển sang review source. Không thay ACL, tắt sandbox hoặc vượt kiểm tra quyền.
**Không ghi rằng Antigravity đã tự chạy xong các test.** Nó hoàn thành review
đọc source, trả 6 finding, status SUCCESS sau khoảng 330 giây.

Codex chạy đúng driver trên cùng bản sao và kiểm JUnit/log:

| Phần kiểm | Kết quả có bằng chứng |
|---|---|
| 17 file test release Python | **238 passed**, 0 failed/skipped, gồm 4 CUDA parity/persistence cases |
| Node UI/trace/composer | **27 passed** |
| HTTP feedback thật trên engine cô lập | **3 passed**: chat, debug, mất receipt sau commit |
| Trạng thái live | Tests không sửa model hoặc dữ liệu người dùng; chỉ fixture ALIF CPU/CUDA có compute/cập nhật cục bộ |

Lần đầu bản sao thiếu hai DLL cuBLAS: 234 pass/4 lỗi môi trường, giữ nguyên log.
Sau thêm đúng dependency public, cả 238 qua; actual installed-runtime CUDA
receipt riêng cũng 4 pass. Không thay code/test/threshold để làm xanh. Có warning
CuPy không tự tìm CUDA_PATH, nhưng loader nạp pinned DLL và GPU parity đã qua.
Đây là **toàn bộ phạm vi test release đã chọn**, không phải mọi test nghiên cứu
trong repository hoặc chấm chất lượng chat/video/tự học.

## Triage sáu finding

| Finding của Antigravity | Đối chiếu Codex | Quyết định |
|---|---|---|
| Bộ nhận lời chào quá hẹp, có thể giữ ví dụ steering | Các cách như “chào bạn nhé”, “hi there” chưa match. Lỗi đúng ảnh đã tái hiện bằng native model và sửa ở host; chưa chứng minh mọi biến thể bị echo | Chấp nhận một phần, P2; mở rộng cần test, không coi tất cả là lỗi đã đo |
| Mất truy xuất màn hình với cách hỏi tự nhiên | “What was visible on the screen?” và “what you saw on the window” chưa match. Nhưng ví dụ “Tóm tắt nội dung màn hình vừa rồi” mà agent nêu **đã match** | Chấp nhận một phần, P2; bác ví dụ sai |
| Video draft chưa có session có thể bị chat khác lấy theo cùng text | Dùng Store/VideoContext thật trên fixture: một pending upload không có session ID bị session B tiêu thụ, session A sau đó không còn nguồn | **Xác nhận P1, chưa giải quyết. Chặn approve production** |
| Audio lỗi dừng cả capture; snapshot dùng chung file; lock share toàn hệ | Snapshot hai frame khác nhau trả cùng path và ghi đè đã tái hiện bằng PIL/filesystem. Audio hiện fail-stop đúng source. Một share tại một thời điểm là policy tài nguyên đã chọn | Chấp nhận snapshot P2 và ghi hạn chế audio; global lock không tự là bug |
| Incident bỏ sót lỗi đổi args và “im lặng vĩnh viễn” | Detector hiện cố ý đo cùng tool/args/result; args khác có thể là chẩn đoán hợp lệ. Scope theo từng turn, không phải một khóa vĩnh viễn cả phiên. Hermes còn hard stop riêng | Ghi giới hạn coverage; không thay detector/reward theo khuyến nghị chưa kiểm chứng |
| Packager cần Git nên ZIP không standalone | Git dùng để build từ checkout; installer ZIP không chạy Git. Fresh-machine/source-export build chưa trong phạm vi RC | Không là release blocker theo phạm vi đã công bố; không thêm SHA fallback tùy tiện |

Các repro không dùng video/hội thoại riêng. Report gốc, event log, source manifest,
JUnit và thử nghiệm đối chiếu nằm ở `local/Antigravity-release-QA/`; raw agent
output được giữ, không ghi đè. [Biên bản máy đọc được](evidence/ANTIGRAVITY_FULL_QA_20261114.json)
ghi hashes và phân biệt executor/test/review.

## Lỗi marker người dùng vừa báo

System prompt có ví dụ `STEER_CHANNEL_NOTE`; VLM0.8B chép thành
`[OUT-OF-BAND USER MESSAGE — Chào bạn]`. Cùng weights/prompt tái hiện được;
bỏ đúng ví dụ host không dùng cho một lượt xã giao thì hết lỗi. Helper đã cài,
reload riêng gateway Hermes. Canonical Hermes thật trả lời “chào bạn” bình thường;
thử lại với history có câu sai cũng không cần xóa DB.

Không hardcode câu trả lời hoặc lọc output để che lỗi. Không thay ALIF, weights,
quy tắc học hoặc context 16K. Lời chào khác còn có thể sai ngôn ngữ/thiếu tự nhiên;
không tuyên bố chất lượng chat đã hoàn hảo.

## Việc bắt buộc tiếp theo

1. Thay first-draft text matching bằng binding có draft/admission identity thực,
   truyền qua metadata đúng request. Kiểm hai chat cùng câu hỏi, retry, đổi tab,
   restart và upload cùng lúc. Không chèn token kỹ thuật vào lời người dùng.
2. Tách snapshot theo nội dung/source hoặc tạo immutable snapshot để hai lượt
   hỏi không ghi đè nguồn của nhau. Kiểm concurrency và quota archive.
3. Nâng coverage cách hỏi lịch sử màn hình/lời xã giao theo repro; giữ phân biệt
   truy xuất lịch sử và quyền capture mới. Xác định chính sách degrade audio.

Các bản sửa screen/greeting đang cài là candidate có phạm vi. Không phát hành
nhãn production hoặc đổi kết luận SNN-over-RAG/retention chỉ vì suite qua.
Lượt test 8 giờ vẫn đã hủy, không mở lại nghiên cứu/dataset/model mới.
