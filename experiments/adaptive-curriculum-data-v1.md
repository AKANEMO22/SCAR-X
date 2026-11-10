# Dữ liệu development cho chat/vision — chuẩn bị trước learner mới

2026-11-10. Được phép tạo dữ liệu qua Antigravity CLI theo chỉ đạo người dùng.
**Không cho phép triển khai U, huấn luyện, chấm final test hoặc thay main.**
Theo master P1/P2/P3 và đề xuất readout-usefulness-proposal-v2, không mở lại
family đã STOP bằng dữ liệu mới. Không gọi 128prompts là normalchat/vision.

1. Một CLI call, tối đa270giây; nếu lỗi giữ log, không tự gọi lặp vô hạn.
2. 128wording:96chat =6intent×2language×8 và32vision =4kind×2language×4.
   AGY chỉ tạo wording, không đáp án/nhãn chân lý, không chạy tool khác finish.
3. Chỉ gửi contract công khai; CWD tạm bên ngoài repo, không gửi chat/ảnh/DB/
   credentials/weights riêng. Raw CLI output ở local/, candidate công khai ở data/.
4. Validate schema/count/ID/balance; exact normalized duplicates trong candidate
   hoặc trùng public wording cũ bị quarantine. Không sinh bù sau khi nhìn score.
5. Rà semantic từng hàng; vision target phải từ renderer/annotation deterministic
   đã review, chat target là style do dự án đặt riêng. Không dùng Qwen làm nhãn.
6. Chỉ sau review mới có reviewed.json. Tất cả là development đã được xem;
   final test cho nghiên cứu tương lai phải có protocol/split độc lập trước score.
7. Không train/generate SCAR-X hoặc tải Qwen trong bước tạo/kiểm dữ liệu. Reference
   Qwen đã có cho các corpus cũ vẫn giữ nguyên; so fresh data cần run riêng.
8. Log thật trên work trace; thiếu/invalid thì ghi chưa sẵn sàng, không mock output.

## Amendment kỹ thuật trước dữ liệu/quality score

Run `20261110T131337Z-adaptive-curriculum-data-v1-3c0f41` đã terminal sau1.06s,
không candidate, monitor dừng vì commit1.207GB >1GB (RSS534MB). Launcher thiếu
pin numericalthreads, parent process riêng peakcommit669MB. Sửa env trước NumPy
imports như launcher sản phẩm. Cho phép đúng một retry sau lỗi tài nguyên này,
giữ270s/1GB, không thay contract/schema/quality threshold. Nếu lại lỗi thì dừng
và báo, không gọi tiếp hoặc tăng budget từ kết quả. Giữ receipt/log run thất bại.
