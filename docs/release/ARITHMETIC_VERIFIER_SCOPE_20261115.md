# Không tạo nhãn học số học từ một biểu thức tình cờ

## Lỗi đã tái hiện và phạm vi sửa

[Repro RC12](evidence/RC12_ARITHMETIC_TASK_REPRO_20261115.json), không gọi model:
`What color is 12+3 written in?` / `15` được arithmetic reviewer gán đúng;
`Hãy đọc chuỗi 12+3, đừng tính.` / `16` và câu hỏi code chứa `12+3` / `14` bị
gán sai số học. Cả ba không yêu cầu trả kết quả phép tính. Từ đó một review có
thể cấp nhãn cho action verify khi không giải quyết đúng yêu cầu.

P0 capability guard14/11 đã ghi đúng giới hạn này; guard đó thực tế đã nằm
trong payload hiện tại. Không chạy lại192 mask cũ. Lỗi còn lại là phạm vi
checker, không phải thiếu một lần học hoặc chưa bật GPU.

RC13 thêm `scarx.hybrid.arithmetic`: chỉ nhận toàn bộ câu theo grammar hẹp của
hai số nguyên0–99, cộng/trừ, câu tính trực tiếp Việt/Anh; giữ NFKC và lời chào
đơn giản. Biểu thức trích dẫn, câu hỏi màu/code, phủ định, nhiều yêu cầu, toán
ngoài phạm vi và word problem chưa hỗ trợ đều **không được máy tự xác minh**.
Chúng vẫn đi qua chat/review thông thường, không bị gán là sai.

Engine và reviewer dùng cùng checker mới, nên action mask, thực thi verify,
review và kiểm phản hồi số học không lệch nhau. ID proof đổi thành
`integer-arithmetic-explicit-task-v2`; các proof/snapshot lịch sử không sửa.
Toàn bộ methods của HybridEngine giữ nguyên AST; chỉ đổi import của verifier.
Original verifier của nhánh SCAR-X gốc, ALIF, encoder, phương trình cập nhật,
precision, checkpoint và replay không thay đổi. Không train thêm để sửa lỗi này.

## Literature-first và quyết định kế thừa

Đã đọc code [Reflexion update_memory](https://github.com/noahshinn/reflexion/blob/main/alfworld_runs/generate_reflections.py)
và [ExpeL ReflectAgent](https://github.com/LeapLabTHU/ExpeL/blob/main/agent/reflect.py).
Các implementation có success/environment outcome và reflection thành phần
riêng; văn bản tự nhận xét không tự là kiểm chứng độc lập. **REUSE cách phân
tách outcome với reflection; không copy code/API/framework, không tái lập
benchmark của họ.** ToolTalk đã được intake/schema/scorer audit trước; không
chạy lại chỉ để xác nhận cần outcome đúng phạm vi.

Khoảng trống SCAR-X ở lượt này là một lỗi cấp nhãn có counterexample cụ thể.
Sửa grammar/checker và regression đủ để kiểm đường đó; không cần learner mới,
seed sweep hoặc so thắng RAG. User đã cho phép sửa sản phẩm; không suy ra quyền
mở lại history P0, DEV/TEST hoặc nới gate từ việc siết checker.

## Kiểm chứng trước đóng gói

-82 targeted arithmetic/review/cognition/feedback/web checks qua, gồm regression
  của ba counterexample và toàn bộ20 câu số học đã ghi trong base8h cũ.
-28 package checks qua. Full332 RC11 là baseline cũ, không phải suite chạy lại.
-[Data/source audit](evidence/ARITHMETIC_TASK_SCOPE_AUDIT_20261115.json) đối chiếu
  repro đã lưu và ZIP RC12, không forward/train; original verifier/learner/core
  byte-identical, toàn methods engine AST-identical.
-[Exact-source qualification](evidence/ARITHMETIC_TASK_SCOPE_SOURCE_20261115.json)
  khóa recipe/source/JUnit cho gói mới. Không tự coi metadata là proof chất lượng.

Activation/state và actual path acceptance được ghi sau khi thực hiện. Đây là
sửa tính đúng đắn của nhãn, không chứng minh SNN đã hiểu nguyên nhân tool loop,
thắng cùng LLM+RAG hoặc vượt ngưỡng forgetting.
