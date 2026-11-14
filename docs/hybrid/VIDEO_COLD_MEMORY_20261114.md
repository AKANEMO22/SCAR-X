# Nhớ nguồn sau restart khác với hiểu đúng video

**Nguồn và trạng thái đã được giữ qua restart; lỗi trả lời vẫn còn.** Trong
lượt kiểm mới, hai tiến trình riêng nạp cùng trí nhớ video tạo cùng prompt và
cùng câu trả lời, kể cả câu sai. Không có bằng chứng việc restart làm mất dữ
liệu trong các ca này; cũng chưa có bằng chứng SNN đã biến video thành kỹ năng.

[Protocol trước inference](../../experiments/hybrid-video-cold-memory-diagnostic-v1.md),
[run7f5e62](../../runs/20261114T070321Z-video-cold-memory-v1-7f5e62/result.json),
[audit dữ liệu](../../runs/20261114T070321Z-video-cold-memory-v1-7f5e62/independent-data-audit-v1.json).

## Đã kiểm thế nào?

Tái dùng hai kho trí nhớ owned của [lượt video trước a98122](ASR_CANDIDATE_AND_TOOL_DATA_20261114.md):
Whisperbase/small đã xử lý cùng clip21giây có speechFLEURS tiếng Việt và hình
học do script tạo. Không quay màn hình/nghe dữ liệu riêng tư, không recapture
hoặc sửa transcript. Đây là clip đã biết, không phải mẫu TEST mới hay1–2giờ
video đa dạng theo gateP1.

Mỗi kho có hai selector: scorerALIF chưa học/khóa weights trong lúc đo, và
RAG giữ thứ tự candidate của FTS/temporal filter. Baseline này chỉ là
candidate-order control, không thay cho strongestRAG trong benchmark lớn.
SNNmode trong checkpoint vẫn là learned, nhưng learning_enabledfalse và toàn
lượt0update; không đánh tráo thành kết quả của một SNN đã được huấn luyện.

Bốn cell, mỗi cell tiến trìnhA hỏi3câu rồi thoát; tiến trìnhB mới nạp data và
checkpoint vừa lưu rồi hỏi lại. Tổng8PID khác nhau,24nativeHTTPcalls thực với
24responseID khác nhau. Có6requestbody duy nhất do lặp phase/selector và hình
ảnh dùng chung. Không dùng responsecache của harness để giả lần gọi thứ hai.
NativeVLM39372 giữ nóng, nên **không đo coldstart của toànVLM**.

ALIFforward trênCUDA; không `engine.start`, idle worker, feedback, replay,
optimizer hoặc học hidden. Giữ nguyên HybridEngine.ask, prompt và output128.
RAGcontrol override relevance thành0 để giữ candidate order; annotation riêng
ghi điều này vì raw retrievalstring của engine vẫn chứa mô tảSNN chung.

## Persistence được chứng minh trong phạm vi nào?

- Payload checkpoint cuốiA khớp byte với checkpoint đầuB ở cả4cell.
- Bảng evidence, stream metadata, source media và hash learnedstate (weights,
  stable, importance, normalization, RNG/replay/counters) giữ nguyên.
- Cả12cặp câu hỏi trước/sau chọn đúng cùng evidence/thứ tự và prompt byte.
- Câu trả lời trước/sau cũng giống hệt trong12cặp. Đây là số đo của lượt này,
  không bảo đảm mọi lần nativegeneration luôn deterministic.
- Hai kho gốc a98122 không thay đổi; chỉ privateclones có lịch sử hỏi/đáp và
  checkpoint mới. Model live, database hội thoại live và lượt8h không bị đổi.

Auditor đọc checkpointNPZ khôngpickle, tự tính SHA/strictscore, đối chiếu source
SQLite, sourcefiles, request/rawresponse và process phases. Đây là cùng
orchestrator, chưa có review ngoài; việc A thoát trước B được bảo đảm bởi
parentwait/exitcode đã qua trong source và raw completedphase records.

## Điểm trả lời và ví dụ lỗi

Theo hợp đồng gốc trả số/một từ, cả4cell đều **1/3 trước và1/3 sau**. Không
nới rubric sau khi thấy output. Câu hỏi màu cuối trả “Yellow” đúng. Câu hỏi
thứ hạng đầu trả cả câu thay vì chỉ số, nên strictfail dù có số15; yêu cầu dẫn
nguồn của system và định dạng ngắn của câu hỏi có thể xung đột, vẫn giữ nguyên.

Ví dụ nguyên văn trên khoWhispersmall, trước restart:

| Câu hỏi về AdamHall, đáp án mẫu9 | Output thực |
|---|---|
| SNNscorer | `Adam Hall xếp thứ 15.` |
| RAGcandidateorder | `Adam Hall xếp thứ 9 trong ánh sếp hàng Super G.` |

Cảhai vẫn strictfail vì không chỉ trả số. Hai đoạn bằng chứng là cùng một tập,
chỉ khác thứ tự: một đoạn mang số15, đoạn kia số9. Transcript gọi tên người
thứ hai là “Aramhorn”, không phải exactanchorAdamHall. Không tự coi chúng là
cùng người hoặc dùng output có số9 để tạo nhãn học đúng. Điều này phù hợp với
[diagnostic binding/order trước](VIDEO_ENTITY_BINDING_DIAGNOSTIC_20261114.md):
thứ tự bằng chứng và tên bị nghe sai còn ảnh hưởng đáp án. Chưa chứng minh
representationSNN giúp chọn/hiểu nguồn tốt hơn.

Không báo CI hay24mẫu chất lượng độc lập: chỉ1video/3câu hỏi đã biết, hai
ASRvariants và các lần lặp. Giữ kết luận matchedRAG/SNNDEV trước đây là PIVOT.

## Giới hạn và bước tiếp

Whole lượt21,562s, nativeVLM dùng chung; không có model reload hoặc training.
252Python/native/CUDA regression và18Nodechecks qua, gồm canonicalSQLite rows/
UTF8backup. Các checks không tăng số video/câu hỏi hoặc bằng chứng học. Lượt đầu
[d5867b](../../runs/20261114T070146Z-video-cold-memory-v1-d5867b/result.json)
dừng vì child đọc manifestUTF8 bằng mặc địnhCP1252 trênWindows,0nativecalls.
Giữ lỗi gốc; lượt sau dùng UTF8 rõ ràng và chuẩn hóa row factory trước tính hash.
Tổng campaign24calls, không mở lại để làm điểm đẹp hơn.
Auditor hỗ trợ `--output local/<tên-mới>.json` để đối chiếu lại dữ liệu mà không
gọi model hoặc ghi đè kết quả cũ.

Cần giải quyết perception/tên riêng, binding và khả năng đọc bằng chứng của
VLM/SNN; không chỉ lưu thêm lịch sử. Cold persistence không phải retention của
một kỹ năng neural đã học. P1 video đa dạng, tự chẩn đoán tool và lifelong
retention vẫn chưa được xác nhận. Rulehistory mới vẫn chờ phạm vi duyệt,
không triển khai hoặc train trong khi làm lượt kiểm độc lập này.

Output thật và đối chiếu trước/sau hiện ở [localhosttrace](http://127.0.0.1:8775/?tab=trace).
Các số này thuộc diagnostic, không cộng vào liveALIF hoặc biến thành reward.
