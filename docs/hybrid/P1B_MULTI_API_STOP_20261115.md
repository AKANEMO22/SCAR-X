# P1B-v2: nhiều API vẫn thiếu action usefulness; dừng family sáu action

**NO-GO cho việc train selector trên family sáu action hiện tại.** Không tuyên
bố SNNkhôngthểhọc nói chung; giữ ALIF/nativeprecision và các ngưỡng đã đăng ký.
Sau hai vòng DEV đã đăng ký, không tạo thêm bank hoặc tăng epochs để tìm thắng.
Thí nghiệm adapter-retention riêng không là bằng chứng SNN advantage.

## New hypothesis, source và actual calls

V1AddAlarm bank đã đóng với TESToracle bằngRAG. [Protocol v2](../../experiments/hybrid-p1b-multi-api-error-v2.md)
đăng ký trước outputs giả thuyết ba API có thông tin schema/state khác nhau,
có thể giúp selector chọn action hữu ích. AddReminder,CompleteReminder,
DeleteAlarm dùng nguyên source ToolTalk e05f4ce6, không gửi reminder thật hoặc
thao tác tài khoản bên ngoài. Exact state checker không dùng semantic similarity.

Đã tạo72teacher corrections thật,36TRAIN/12DEV/24TEST. Các setup object IDs
được RNG riêng cho từng case tạo bằng API thật; public task biết ID/giá trị yêu
cầu, expected checker state không đưa vào policy. Split wording/setup RNG tách,
API/error schemas vẫn chung.0 NNtraining trong data phase.

[0ddc0e](../../runs/20261115T081550Z-p1b-multi-api-v2-0ddc0e/result.json)
dừng0nativecalls trước inference do checker sai enum: mô tả API ghi completed,
nhưng implementation CompleteReminder setstatus"complete". Lỗi của harness
Codex, source/failure giữ nguyên. Hai pure regressions đối chiếu ASTsource và
no-other-record mutation qua. Chỉ sửa source-status checker, không thay
score/criteria/prompt/label dựa modeloutput; zero-call continuation khóa hash
failure, không có retry của một kết quả inference/learning.

[Run c07292](../../runs/20261115T081658Z-p1b-multi-api-v2-c07292/result.json)
thực84native inputs =12DEV×6actions+RAG; actual correction replies đều thực thi
trên isolated source snapshot. Rule, mọi constant-action và RAG retrieval cùng
nguồn/native0.8B/base, không dùng LoRA vừa train để đổi điều kiện. Không native
TEST hoặc SNN forward/update được chạy sau usefulness STOP.

## Đã đo điều gì?

|Control/action|Đúng task effect trong12DEV|
|---|---:|
|show_schema|2|
|inspect_status|6|
|retry|3|
|clarify /defer /finish|0 /0 /0|
|RAG verified TRAIN experience|5|
|Best six-action outcome, biết trước cái nào đúng|6|

Constant inspect_status đã bằng oracle6/12: marginal benefit tối đa0pp, dưới
gate5pp. Thêm RAG như một action thứ7 trên **các outcomes đã có** tăng union
đến7/12=58,33%, vẫn dưới acquisition65%. Vì vậy cũng không cần chạy neural để
chứng minh giới hạn đó; không tự đổi readout/architecture hoặc nới gate.

Theo API: AddReminder oracle1/4,CompleteReminder2/4,DeleteAlarm3/4. Một số
actual answers vẫn lặp task_text, giữ due_date"not a date" hoặc giữ sai type
list dù tool báo lỗi. Selector chỉ chọn một thông điệp phục hồi chưa làm cho
VLM xử lý đúng được; credit/local learning không thể tạo một action thành công
ngoài tập outcomes hiện có. Đây là nút thắt **capability/action contract**.

[Independent audit240b2a](../../runs/20261115T082203Z-multi-api-data-audit-v2-240b2a/result.json)
replay264 source-tool calls, verify72 teacher corrections và84 saved replies,
0 new native inference/training, khớp mọi effect. Auditor dựng lại source
setup/error prefix để so statehash (snapshot JSON sortkeys đổi insertion-order,
không phải thay trạng thái). First auditor assertion về serialization order
không được dùng làm lý do sửa model scores; independent state equality vẫn kiểm.

## Quyết định tiếp theo

-Giữ sáu-action familySTOP, chưa hidden/readout/20stage SNN retention mới.
  Không lấy model không-update/khôngquên làm retentionPASS; historical failures
  giữ nguyên. SNN contribution chưa được chứng minh.
-LoRA native portability có hiệu quả trên taskrepairhẹp, nhưng không biến
  action-table cũ thành SNN success. Muốn nới capability phải có source/version/
  registered hypothesis mới, không sửa TEST hoặc import final outputs vào train.
-Hiện tiếp tục **một adapter dùng chung** học APIerrors đã được source-verify
  trong [retention20 study](../../experiments/hybrid-shared-lora-retention20-v1.md),
  đo interference qua20stages. Đây là LoRAbackprop, không ALIF; giữ3/5pp gates.
  Không tự promote adapter hoặc sửa Hermes/product core.
-Cần phân biệt khả năng giữ kỹ năng qua các chặng học với trí nhớ video/event
  và general reasoning.72controlledAPI cases không trả lời mọi mục tiêu đó.
