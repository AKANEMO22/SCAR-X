# Guard khả năng verifier — qua bản sao, chưa nạp live

Đã chuẩn bị sửa lỗi controller cung cấp actionverify cho mục tiêu mà checker
hiện không hỗ trợ. Lỗi này đã xuất hiện1lần trong base8h. Candidate chỉ cho
phép verify khi `parse_integer_expression` hiện có chấp nhận biểu thức và
goal chưa thửverify. Không thay ALIF, encoder, công thức học, source reward,
checkpointformat hoặc trọng số. **Đây là runtimeguard, không là learnedrepair.**

[Protocol](../../experiments/hybrid-verifier-capability-guard-p0-v1.md),
[recipe exactEngineSHA](../../integrations/hermes/verifier_capability_guard_patch.py),
[raw8e205a](../../runs/20261114T105211Z-verifier-capability-p0-v1-8e205a/result.json),
[data/ASTaudit](../../runs/20261114T105211Z-verifier-capability-p0-v1-8e205a/independent-data-audit-v1.json).

## Thay đổi đã kiểm

Actual `HybridEngine.allowed` của baseline và candidate được gọi trên context
controller/SQLite riêng, không khởi tạo model hoặc sinh câu trả lời giả.32câu
nguồn ×6điều kiện =192case, khóa trước đo:20câu cộng gốc, ca unsupported gốc,
canonicalEnglish/Vi/fullwidth, toán ngoài phạm vi, phủ định và câu khôngtoán.

-28mask được bỏverify khi checker không hỗ trợ; không có delta nào khác.
-150mask của các câu parser-accepted giữ nguyên, **gồm6context của câu ý định
  mơ hồ bên dưới**, không đồng nghĩa150câu toán được hiểu đúng.
-Giữ localreview trước hành động, không verifylần2, một câu hỏi chờ, explicit
  publictopic và searchrate. Không mở thêm quyền/tool hoặc đổi logicsearch.
-AST của toàn engine ngoài methodallowed giống hệt; sourceactive không bị đổi.
-10unitchecks trên codeguard thật qua.0neuralforward/learn/VLM/webcalls;
  tổngCPU/SQLite~0,031s, không là speedbenchmark hoặc qualitymetric.

Sourcechecker vẫn chỉ nhận một số biểu thức hai số nguyên1–2chữ số với cộng/
trừ và mẫu câu cố định. Guard không thêm tri thức mới hoặc đọc ý định bằngSNN.

## Hạn chế cần giữ rõ

`What color is 12+3 written in?` vẫn được parser chấp nhận vì có biểu thức
12+3, dù câu hỏi có thể yêu cầu màu. Sáu context này được đăng ký như giới hạn
đã biết và giữ trong rawcase; không được gọi thành semanticPASS. Việc tính15
đúng không chứng minh15 giải quyết đúng mục tiêu người dùng.

Thay đổi chỉ loại checker **không thể** xử lý theo parser hiện tại; chưa giải
quyết falsepositive ý định, sourcebinding, factualverification hoặc các tool
khác. Negativeutilityfeedback vẫn chưa được nối. Không biến reportederror/
verificationflag thành nhãn đúng, hoặc dùng guard này để chứng minhSNN học
nguyên nhân lỗi. [Targetaudit và hiddenreset](WHAT_THE_SNN_LEARNS_20261114.md)
giữ nguyên mọi kết luận/giới hạn.

## Tích hợp tiếp

Lượt smallmixed32cd13 đang chạy với source36file khóa. Không thayengine/store
hoặc restart nativeworker để cài guard bây giờ. Sau actualterminal và dataaudit,
có thể review/cài cùng [fullquerySQLite](FULL_QUERY_RETRIEVAL_P0_20261114.md):
exactSHA backup, fullprivatecheckpoint/configbackup, maintenancelease và source
CAS, rồi targetedcontroller/SQLite/state/native checks. Không lấy kết quả8h của
source cũ để chứng nhận bản sau khi thay. Không tự nhập checkpointlab vàoHermes.

**GO cho guard ở bước tích hợp kế tiếp; PIVOT nghiên cứu neural giữ nguyên.**
Main/protectedR13 giữ nguyên; không localrule mới, scaling, LoRA hoặc language
training. Hai candidate là các sửa runtime có phạm vi rõ, chưa là hệ production
đã đủ chất lượng hoặc khả năng tự học kỹ năng lâu dài.
