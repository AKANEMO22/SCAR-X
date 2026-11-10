# Model artifacts

Sản phẩm hiện hành: `SCAR-X-Generation-1.zip`, checksum cùng tên và
`generation1-release.json` trỏ một checkpoint chung. `generation1-engineering.json`
và `generation1-package-check.json` ghi phép kiểm triển khai. Release chọn seed
bằng development; chất lượng ngôn ngữ còn yếu, không đạt generalization gate.

ZIP v0.1/v0.2 được lưu offline trong `retired/`; các pilot weights còn lại chỉ là
fixture nghiên cứu và regression lịch sử, không được server G1 nạp thành model.

`association-seed17/` chứa một checkpoint bài liên kết nhỏ từ run xác nhận, không phải chatbot hoặc model ngôn ngữ. Có manifest ghi nguồn và SHA-256. Giữ các checkpoint có kết quả đánh giá tại đây; file tạm cá nhân để trong `local/`.

Không dùng tên model artifact để tuyên bố chất lượng vượt những gì run đã đo.
