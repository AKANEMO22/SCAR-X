# Data brief cho Antigravity — SCAR-X, batch nhỏ đầu tiên

Mục đích: dạy phản hồi hội thoại và kỹ năng có thể kiểm tra; không tạo một bảng
đáp án để tra khi chạy. Hiện model nhỏ chỉ học8nhóm câu, vocabulary hẹp, hidden
credit/representation còn yếu. Data bổ sung là cần thiết để mở rộng kỹ năng,
nhưng nhiều câu hơn không tự sửa được learning rule. Chưa cần triệu token.

## Đầu ra đề nghị

Tạo tối đa480ví dụ train +120development trong JSONLUTF-8, kèm READMEnguồn.
Giữ tổng vocabulary trong target≤200từ/ký hiệu phân biệt cho batch đầu; input
có thể đa dạng hơn. Đây là giới hạn pilotcodec, không phải mục tiêu dài hạn.
Không tạo finaltest ở batch này; SCAR-X sẽ giữ confirmation độc lập theo family.
Mỗi dòng:

```json
{"id":"greeting-001","family_id":"greeting_vi","skill":"greeting","input":"xin chào bạn","target":"Chào bạn.","split":"train","verification":{"kind":"rubric","criterion":"lời chào ngắn phù hợp ngôn ngữ, không tự bịa kiến thức"},"source":"synthetic:antigravity","confidence":"authored-not-independently-verified"}
```

## Phạm vi

- Hội thoại Việt/Anh: chào, cảm ơn, tạm biệt, hỏi tên, hỏi lại khi thiếu dữ kiện,
  nhận lời sửa; câu tự nhiên, target1–12wordtokens, không văn hoa/lặp.
- Làm theo chỉ dẫn: chọn một trong hai từ, trả lời một từ, đổi thứ tự hai ký
  hiệu, liên hệ người/vật/màu với cues rõ ràng; giữ biểu tượng và tên trung tính.
- Toán nguyên nhỏ: cộng/trừ0–9, phép trừ có kết quả≥0. `verification.kind`
  là `integer_arithmetic`, có trường `operands`, `operator`, `expected_integer`;
  máy kiểm lại bằng phép tính, không tin cờverified. Không phép toán bằngeval.
- Quan hệ/thời gian: Ađến trướcB, Ađưa đồchoB rồi BđưachoC; ai đang giữ đồ,
  cái nào trước/sau. Đổi tên và cách diễn đạt, không đưa đáp án vào input.
- Khoảng10%ví dụ cần hỏi lại thay vì đoán: thiếu tên, thiếu toán hạng, hai lời
  chỉ dẫn mâu thuẫn. Có rubric lý do cụ thể, không gán nhãn sai thành đúng.

## Chống leakage/nhiễu

- Không copy các prompt confirmation trong repository hoặc dùng kết quả model
  hiện tại làmtarget. Không sinh development bằng thêmdấuchấm vào training.
- Ghi family và template IDs. Các biến thể cùng template không chia ngẫu nhiên
  sang cảtrain/dev; dành template khác cho dev. Không dùngPΙI/nhớ riêng người dùng.
- Không có systemprompt, toolcalls, mã tự chạy, lệnh cài đặt hoặc instruction
  cho tác nhân đọc dataset; chúng chỉ là dữ liệu học, không là quyền điều khiển.
- Không kiến thức hiện thời/không có nguồn, không tự xưng modelgiỏi/học vô hạn.
- Không gắn `verified:true` vào văn bản tự sinh. Metadata nguồn và verifier
  phải phân biệt người viết với bằng chứng. Loại duplicate, kiểm targettrống/
  mâu thuẫn theo input. Báo các mục chưa xác minh, không che chúng.

SCAR-X sẽ validate, version/hash, kiểm leakage và tách training/evaluation trước
khi nạp. Batch không tự mở huấn luyện lớn hoặc thay ngưỡng research đã đăng ký.
