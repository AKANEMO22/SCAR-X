# SCAR-X Hybrid 1.0.0-rc.14

Bản cập nhật RC cho SCAR-X Hybrid **đã cài trên máy Windows/Hermes hiện tại**.
Đây là gói code nhỏ, tái sử dụng model/runtime đã có; không phải bộ cài độc lập
cho máy mới. Không chứa venv, trọng số model, hội thoại hoặc checkpoint cá nhân.

RC14 yêu cầu lời tính toán rõ ràng cho dấu gạch ngang ASCII; chuỗi trần như
3-1/03-01 có thể là tỷ số/ngày nên không tự cấp nhãn phép trừ. Proof IDv3.
Review Antigravity high được kiểm độc lập, không phải approval chất lượng model.

RC13 chỉ cho verifier số học tự xác minh các câu tính toán hoàn chỉnh nằm
trong grammar hỗ trợ. Biểu thức nằm trong câu hỏi màu, code, chuỗi trích dẫn
hoặc nhiều yêu cầu không tự thành nhãn máy. Word problem/cách diễn đạt ngoài
phạm vi vẫn dùng chat/review bình thường; chưa được xác minh, không coi là sai.
ALIF, công thức học, encoder và checkpoint không đổi.

RC12 đọc được envelope `untrusted_tool_result` thực của Hermes khi lưu nguồn;
giữ nguyên envelope trong prompt model và luôn để snippet ở trạng thái observed.
Kiểm tra có thể replay từ receipt cũ mà không tìm web/chạy model lại.

RC11 lưu nguồn web đúng lượt tool và câu hỏi gốc cho việc xem lại/feedback.
Không trộn FTS trí nhớ cá nhân vào câu hỏi web công khai; nguồn snippet luôn
`observed`, không tự thành verified. Diễn giải viết tắt chỉ là metadata runtime,
không sửa dòng chat hoặc tuyên bố neuron đã học. Có thể còn trả lời sai/thiếu;
nguồn đúng và self-review có ngữ cảnh không tự chứng minh có kỹ năng bền vững.

RC10 chọn web_search thật qua Hermes cho câu hỏi thông tin mới và giải quyết
lượt tiếp “tìm kiếm đi” theo chủ đề người dùng trước đó. Scope này dùng SearXNG,
không tự chuyển sang dịch vụ extraction. Yêu cầu workspace giữ công cụ local;
lượt khác khôi phục toolset bình thường. Đây là routing runtime, không neuron
đã học lựa chọn nguồn hoặc một đáp án đã được kiểm chứng.

RC9 lưu feedback, checkpoint và biên nhận trong một transaction SQLite. Retry
cùng mã và nội dung trả cùng kết quả; đổi nội dung phải dùng mã gửi mới. Lỗi
SQLite trả lỗi có thể thử lại và phục hồi trạng thái trước phản hồi. Receipt
ở journal ngoài neural checkpoint, không bị xóa khi rollback synapse. Không
đổi learning rule hoặc tự xác minh nội dung lời người dùng.

RC8 tách lỗi decode/ASR không phải lỗi tài nguyên khỏi luồng hình màn hình.
Sau ba lỗi audio liên tiếp, tạm dừng ASR của lượt chia sẻ đó; hình tiếp tục,
coverage ghi rõ đoạn thiếu. Dừng rồi chia sẻ lại để thử audio từ đầu. Không
tự chuyển CPU hoặc dùng microphone thay loopback. OOM/breach vẫn dừng pipeline.

Đường nâng cấp RC14 pin đúng payload RC13 và từ chối source khác trước khi sửa.
Source gate của test đã hủy nhận đúng bản trước/bản kế tiếp, không yêu cầu source
đóng băng vĩnh viễn ở baseline cũ. Test đang chạy vẫn chặn cập nhật; trạng thái
hủy vẫn không thành PASS. Backup/rollback chỉ phục hồi code, giữ trải nghiệm mới.

Từ RC6, gói cần Hermes đã có đường metadata admission tương ứng được build/cài.
Installer đối chiếu fingerprints và từ chối app cũ; không tự thay source hoặc
renderer Hermes khác phiên bản. `HERMES_MEDIA_ADMISSION.patch` đi kèm để bảo trì/
review source; không tự chạy patch tùy tiện vào bản Hermes mới.

## Dùng thử

1. Giải nén vào một thư mục riêng. Giữ file ZIP và SHA256 để đối chiếu.
2. Chạy `Verify-Package.ps1`, sau đó `Install-Hermes.ps1` để cập nhật.
   `Install-Hermes.ps1 -Check` chỉ kiểm tra, không sửa source hay dừng dịch vụ.
3. Mở chat Hermes bình thường, tiếp tục chọn model **Scarx Hybrid** hiện có.
   Model ID `scarx-hybrid`, một gateway tại localhost:8775. Tab Debug chỉ quan sát.

Installer từ chối nếu worker bảo vệ còn chạy hoặc không có custody hợp lệ.
Theo yêu cầu mới của bạn, lượt 8 giờ đã dừng để ưu tiên sản phẩm; receipt và
audit ghi rõ USER_CANCELLED, không phải endurance PASS. Không xóa dữ liệu cũ.
Gói RC vẫn cần nghiệm thu vận hành, không chứng minh độ bền 8 giờ.

Model đang dùng: Qwen3.5-0.8B nguyên BF16/F32, Whisper-small FP32, ALIF H128 FP32.
Các tính toán neural trên GPU hiện có; I/O, SQLite và điều phối trên CPU.
Không quantization/pruning/hạ precision hoặc tự bật LoRA.

## Khôi phục

`Rollback-SCARX.ps1` phục hồi code trước cập nhật khi file vẫn khớp SHA đã ghi.
Không ghi đè SQLite, xóa journal hoặc hoàn tác trải nghiệm người dùng phát sinh
sau cập nhật. Checkpoint đầy đủ được sao lưu riêng ở thư mục private
`local/hybrid-v1/release-backups/`; không nằm trong gói.

`Start-SCARX.ps1` / `Stop-SCARX.ps1` điều khiển đúng gateway đã đăng ký, có kiểm
PID/create-time/cmdpath; không dừng Hermes hay các tiến trình khác.
Nếu installer báo source/dependency/pin Hermes không tương thích, dừng và giữ
thông báo, không sửa manifest hoặc xóa dữ liệu người dùng để ép chạy.

## Phạm vi RC

Hai sửa chính: truy xuất toàn câu hỏi trong giới hạn 6.000 ký tự và loại
action verify khi checker không hỗ trợ. Chúng là sửa runtime, không phải bằng
chứng SNN đã học kỹ năng mới. Cửa nghiên cứu/retention không được thay.

Hermes dùng đúng mã transcript cho video đính kèm. Với lời chào đơn giản,
Hermes không gửi schema công cụ dư hoặc ví dụ steering không dùng tới; tác vụ
thật và ảnh vẫn giữ công cụ/protocol. Không sửa câu người dùng hoặc hardcode đáp án.

Cửa sổ đã xem được gắn với đúng phiên chat. Sau stop/restart có thể hỏi rõ
“cửa sổ vừa xem”; hệ thống dùng lại bằng chứng còn giữ mà không capture lại.
Câu không liên quan không nhận ảnh cũ; upload mới có focus riêng. Đây là
truy xuất nguồn, chưa là SNN đã học nội dung vào synapse.

Video của draft mới chỉ được gắn sau khi Hermes xác định đúng session và mã
lượt gửi. Không dùng câu hỏi giống nhau để đoán chủ sở hữu. Retry dùng lại
identity cũ, binding/pending được lưu atomic; restart không làm chat khác lấy
nhầm video. Trước khi admit, chờ quan sát local đầu tiên tối đa 35 giây; timeout
giữ bản nháp để gửi lại. Lượt trả lời từ media local không được tự gọi công cụ
web bằng các observations riêng. Công cụ vẫn dùng bình thường ở tác vụ khác.

Snapshot câu hỏi theo hash nội dung và công bố immutable, nên hai nguồn ảnh
khác không dùng chung đường dẫn ghi được; ảnh giống nhau tái sử dụng cache.
Cache tính vào quota archive. Không tăng dung lượng neural để sửa hai lỗi này.

Lượt trả lời media không có tools được bỏ riêng ví dụ steering host không dùng,
giữ mọi hướng dẫn khác. Ảnh/ngữ cảnh đưa trước câu hỏi để tránh model chỉ caption
ảnh. Mốc nguồn được runtime ghi thêm từ metadata đã cung cấp; log giữ riêng
native_answer và runtime_reference_footer. Đây không phải nhãn model trả lời đúng.

Observer đọc đúng envelope `request.body.tools` của Hermes để không bỏ qua
quyền đọc file khi kiểm tool loop. Tối đa một probe đọc lại nguồn trong project
cho mỗi turn; kết quả chưa được xem là nhãn đúng hoặc kỹ năng neural đã học.

Tin nhắn chat dài không còn bị từ chối chỉ vì query truy xuất vượt 6.000 ký tự.
Gateway chia query FTS thành phần nhỏ, giới hạn pool 32 ứng viên; nội dung gửi
model giữ nguyên. Trần xử lý văn bản 65.536 ký tự không phải context 65K token;
context model vẫn 16.384 token và còn phụ thuộc system prompt/history/công cụ.

Xem `KNOWN_LIMITATIONS.md`, `MANIFEST.json`, `ASSETS.json` và `DEPENDENCIES.json`.
SHA/version/codeoverlay có trong manifest; không xuất token hay cấu hình bí mật.
