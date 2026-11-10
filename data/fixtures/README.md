# Dữ liệu kiểm thử

Các chuỗi số được sinh xác định trong `src/scarx/evaluation/tasks.py`; seed nằm trong protocol. Không cần tải dữ liệu ngoài.

`scripts/make_learning_demo.py` xuất `examples.json`, `guard.json` và checkpoint ngẫu nhiên vào `local/demo/` để thử CLI. Đây là liên kết tín hiệu nhân tạo, không phải dữ liệu chứng minh năng lực tiếng Việt.

Đối với dữ liệu ngôn ngữ tương lai: ghi nguồn, giấy phép, hash, chuẩn hóa và phân chia train/development/test trước khi huấn luyện. Không tự thu thập dữ liệu riêng tư hoặc tải corpus lớn khi gate chưa qua.
