# Đối chiếu các cập nhật mới của DeepSeek

2026-11-10. **Đang kiểm tra độc lập.**
[Protocol kỹ thuật](../../experiments/deepseek-product-review-v1.md).

Phạm vi: các commit sau `8337d60` đến `27b6c10` và script interference v2 mới.
Giữ các sửa guard/quan sát có ích; không đổi learning rule, topology hoặc
trọng số của phiên người dùng đã restore. Lõi vẫn ALIF, Qwen chỉ đối chứng.

Các điểm cần kiểm chứng hoặc sửa: phép ghép nhóm trọng số trong intervention;
fallback đo before trên live khi clone lỗi; chụp SQLite/WAL khi nguồn đang chạy;
phân biệt acquisition với ACCEPTED; false positive của metric dùng substring;
phạm vi state reset và kết luận về hidden/readout. Kết quả lịch sử giữ nguyên,
nhận định rollback cũ được đánh dấu đã bị kiểm tra cô lập mới thay thế.

Chỉ chạy can thiệp học trên bản sao để chẩn đoán, không dạy/restore phiên thật.
Nguồn checkpoint không là seed độc lập; norm/LR không phải thước đo đóng góp.
Chưa có kết quả ở báo cáo này, không dùng review để đổi gate P1/P2 hoặc scale.
