# Đo thực từng cách phục hồi trước khi dạy SNN —14/11/2026

**Đã hoàn tất bảng kết quả thực thi, chưa huấn luyện policy.**24ca sở hữu riêng,
144nhánh trên snapshot clone,266unique nativecalls bằng đúng Qwen0.8B/CUDA
đang chạy;281,125s. Mọi input/schema/expected effect khóa trước inference.
Không phát sinh alarm thật, thay quyền/tài khoản, model thứhai hay liveweight.

[Protocol](../../experiments/hybrid-recovery-outcome-bank-dev-v1.md),
[raw bảng/state/source](../../runs/20261114T025511Z-recovery-outcome-bank-dev-v1-196043/result.json),
[auditor re-execute không gọi model](../../runs/20261114T025511Z-recovery-outcome-bank-dev-v1-196043/independent-outcome-audit-v1.json).
Audit replay actualcalls từ cùng initialstate/RNG, kiểm rawnativeinputSHA và
afterstatehash, tính effect/status độc lập; không có integrity problem.

## Đã đo gì?

Sáu tình huống×4ca (2Vi/2En): task đã làm đúng; sai định dạng giờ; thiếu argument
bắt buộc; chưa đăng nhập; người dùng chưa cho giờ; và đã tạo nhầm giờ trước đó.
ToolTalk mô phỏng AddAlarm/DeleteAlarm trong database RAM sở hữu riêng.

Mỗi ca thử cùng6hành động research: show_schema, inspect_status, retry, clarify,
defer, finish. Three terminalactions không được thực thi thêm tool. Các nhánh
khác có cùng upperbudget3nativecalls/128tokens. Safetyguard/actionorigin được
ghi rõ, không coi exhaustiveorchestrator là SNN tự chọn.

| Tình huống | Số ca có ít nhất một nhánh qua effect/status | Quan sát |
|---|---:|---|
| Đã tạo đúng alarm |4/4|finish báo trạng thái hỗ trợ bởi effect|
| Sai HH:MM thay HH:MM:SS |4/4|schema/status/retry giúp sửa ở một sốwording|
| Thiếu argument |4/4|sửa được trên một sốnhánh|
| Chưa login |0/4|lặp call bịtừchối hoặc wrongstructuredstatus|
| Chưa cho giờ |0/4|lặp argument rỗng hoặc hỏi nhưng statuscompleted|
| Đã tạo nhầm giờ |0/4|thêm alarm nhưng không sửa sạch state, hoặc báo đãxong|

Trên24ca, successfulbranchcounts: show_schema7, inspect_status6, retry7,
clarify0, defer0, finish4. Đây là số nhánh đo effect/status, không accuracy của
policy, vì chưa có policy chọn giữa các nhánh hoặc học từ chúng.12ca đầu có
outcomes khác nhau;12ca sau không được biến thành nhãn âm cho mọi cơ chế chung.

96nhánh có executablegoal contract (đúngmộtalarm/ID/time, otherstateunchanged,
finalstatuscompleted),48nhánh dùng structuredsafe-escalation kèm phrasecheck
yếu. Cả natural-language message không được verifier hiểu tổng quát; không
dùng fieldverified_resolution làm bằng chứng mọi câu trong message đềuđúng.

## Giới hạn mới phát hiện ở harness

Response schema liệt kê statusneed_info/need_auth/working, nhưng prompt chưa
giải thích đầy đủ ý nghĩa của chúng; nó giải thích rõ finish/statuscompleted.
Model thường trả ask/statuscompleted, kể cả message nói cần thôngtin. Rawrun
vẫn hợp lệ theo contract đã khóa, nhưng chưa đủ quy mọi failedstatus là thiếu
khả năng nhận ra lỗi. JSONgrammar là ràng buộc cú pháp, không mặc định là tài
liệu hướng dẫn đầy đủ cho model.

Không nới scoring hoặc ghi đè original. [Đề xuất diagnostic statuscontract](../../experiments/hybrid-recovery-status-contract-diagnostic-v1.md)
khóa contrast rõ ràng trước lượt mới. Nếu explicitvocabulary giúp, đó là
harness/control improvement; không tự gán gain cho synapse. Phải áp chung cho
RAG/frozen/readout/hidden trước một newbank/learningDEV.

## Điều đã chốt

- Chưa train bank này: thiếu split chưa thấy, các noactionworks cases và status
  contract cần được rà. Không cố train một nhãn “sửa đúng” khi thực tếchưađúng.
- Source/RNG/actualeffectverification đã có, giúp phân biệt lỗi quyền/format/
  completion; nhưng neuralpolicy mới cần học, kiểmhiddenablation và retention.
- Không giấu strongrule/RAG hoặc dùng26xchạy cùng mộtinput làm independentseed.
- Current0.8B/ASRbase/ALIF vẫnCUDA và live5updates;8h554c19 vẫn nguyênsource.
  Bảng research không đổ vào bộ đếm model bạn đang dùng.
- Kết quả, câu trả lời và actionverifier thật tự hiện ởlocalhost trong card
  “Đo phục hồi lỗi”; không cần chọn experiment hay bấmchạy.

235Python checks+12Node checks và realHTTPreply fixture qua. Nút trả lờiHermes
đã bỏ nestedform/gửi trực tiếp, có error/receipt/doubleclickguard; không gửi nhãn
thử vào live.8hchưa hoàn tất, không kết luận production/lifelongPASS.
