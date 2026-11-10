# ADR0003 — Audit phép đo Phase2.1

- Bài binding ban đầu có shortcut: khi hai giá trị giống nhau, không cần query; khi khác nhau, đoán ngẫu nhiên. Mức75% có thể đạt bằng bag-of-values. Giữ nguyên run/protocol; kết quả binding cũ không chứng minh binding. Thêm challenge riêng với values đối nhau,8seed mới, frozenrule và threshold cũ; không chọn lại hyperparameter.
- Bài temporal permutation dùng một permutation cố định không chắc xóa mọi thông tin thời gian. Negative lesion drop không phải bằng chứng mô hình không cần temporal state; reset-state là một kiểm tra bổ sung. Không kết luận representation thành công khi primary endpoints không đạt.
- Forgotten max-to-final bao gồm cả sự cải thiện sau lần học đầu rồi giảm trở lại. Chênh lệch với diagonal-to-final không thể quy hết cho noise hoặc metricbias. Báo cả hai.
- Các run Phase2.1 đầu có chuỗi mô tả data trong manifest kế thừa `synthetic-tasks-v1`; nhãn này quá chung. Generator thực tế là tasks21.py hoặc lifelong21.py, version xác định chính xác bằng source.zip/SHA-256 và protocol đã snapshot. Không sửa manifest lịch sử. Run mới thêm command/threadenvironment và snapshot scripts để tái lập tốt hơn.
- Wall-time-capped run phụ thuộc scheduling; đã lưu số update mỗi stage. Kết quả equal-update tái lập xác định; kết quả wall-time phải tái chạy bằng recordedupdate schedule khi kiểm tính đúng. Không gán chênh lệch nhỏ wall-time cho thuật toán nếu có biến động lịch chạy.
- Snapshot dùng source tại lúc bắt đầu run; những file stream/control thêm sau đó không thuộc kết quả đã chạy trước. Mỗi run mới có snapshot riêng.
