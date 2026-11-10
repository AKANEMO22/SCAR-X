# 03 — Phản biện và điều kiện bác bỏ

| # | Phản biện | Mức | Cách kiểm chứng / kết luận hiện tại |
|---|---|---|---|
| 1 | Readout có thể làm toàn bộ phần học hữu ích | Nghiêm trọng | Full vs readout-only: cùng 100% recall; chưa có lợi ích hidden plasticity |
| 2 | Reward scalar khó phân công trách nhiệm cho hàng nghìn synapse | Nghiêm trọng | Bài yêu cầu feature mới; so cùng topology với BPTT, không chỉ bài dễ |
| 3 | STDP không bảo đảm tối ưu ngôn ngữ | Nghiêm trọng | Held-out next-token loss và instruction tasks; chưa mở gate |
| 4 | Plasticity phá kiến thức cũ | Nghiêm trọng | 20-context stream: forgetting vượt ngưỡng; rule hiện tại không đạt |
| 5 | Replay hoặc rollback có thể che việc không học | Cao | Đo accepted-update rate và kỹ năng mới cùng retention; benchmark raw không rollback |
| 6 | Đóng băng weights sẽ có forgetting 0 nhưng không học | Cao | Luôn xét gain vs frozen cùng retention; không lấy retention đơn lẻ |
| 7 | Mạng quá nhỏ có thể chỉ phản ánh thiếu capacity | Cao | Thất bại bác bỏ cấu hình/rule đang thử, không bác bỏ mọi SNN |
| 8 | Trạng thái hữu hạn không mã hóa chính xác vô hạn lịch sử | Giới hạn | B bit có tối đa 2^B states; raw SSD có thể giữ dữ liệu nhưng retrieval có thể sai |
| 9 | Literal lookup không phải hiểu 1M context | Cao | 1M byte smoke chỉ tìm identifier; semantic/multihop/global synthesis chưa đo |
| 10 | Spike không tự động giảm tổng RAM | Cao | Tính stable weights, trace, checkpoint copies, runtime; measure commit/RSS |
| 11 | Ít FLOPs không đảm bảo nhanh CPU | Cao | C++/ctypes hiện chậm hơn NumPy; không tuyên bố tăng tốc |
| 12 | Spike hiếm có thể làm mất tín hiệu ngữ nghĩa | Cao | Đo firing distribution và held-out ability ở protocol sau, tránh chỉ tối ưu sparsity |
| 13 | Quantization có thể xóa cập nhật nhỏ | Cao | So local learning trước/sau quantize; chưa bật quantization |
| 14 | Dữ kiện sai hoặc feedback nhiễu làm hỏng weights | Cao | Nguồn và verified flag là đầu vào, không chứng nhận; cần thử noise 0/10/30% trước dùng thật |
| 15 | Bộ benchmark nhỏ quá dễ, có hiệu ứng trần | Nghiêm trọng | 100% ở nhiều condition; không kết luận hidden rule không thể hữu ích ở mọi bài |
| 16 | 3 seed cho CI yếu | Cao | Báo CI thăm dò; vòng xác nhận lặp cùng seed không tăng số seed độc lập |
| 17 | Rò rỉ test làm mọi kết quả vô nghĩa | Cao | Seed tách; test không replay, không guard; không tune trên kết quả đã xem |
| 18 | Ngang thương mại bị suy diễn từ số mẫu nhỏ | Nghiêm trọng | Chưa có matched comparison, không tuyên bố parity |
| 19 | 2 GB RSS có thể giấu commit hoặc GPU | Cao | Đã báo cả commit, child reload; iGPU chưa dùng, chưa hard-cap hoặc đo paging |
| 20 | Giống sinh học được dùng làm bằng chứng thông minh | Cao | ALIF và local traces là mô hình giản lược; không tương đương não người |

Khuyến nghị: giữ framework đo và persistence; thay bài phát triển để đòi hỏi representation learning trước khi thử quy tắc tiếp theo. Không dùng cùng test để tìm hyperparameter thắng. Chưa huấn luyện lớn.
