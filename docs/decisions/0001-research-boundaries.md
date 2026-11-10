# ADR 0001 — Các ranh giới để kết quả không bị hiểu sai

Ngày: 2026-11-08. Trạng thái: áp dụng.

1. Dự án nằm trong `scar-x/` vì thư mục cha đang chứa công việc font. Dùng src layout; domain neural/learning/memory/runtime/evaluation/interfaces.
2. Phiên này không tìm thấy plugin ponytail qua inventory và plugin search. Không bịa quy tắc của plugin; lưu yêu cầu ưu tiên plugin trong AGENTS.md.
3. Bản đầu dùng ALIF, recurrence thưa theo fan-in, luân phiên khối hàng được học. BPTT chỉ xuất hiện trong control, không phải sản phẩm chính.
4. `full` = readout-local delta + reward-modulated STDP. `readout` = cùng forward nhưng chỉ cập nhật đầu ra. Nếu hai bên cùng tốt, chưa chứng minh STDP tăng năng lực.
5. Benchmark đo quên dùng cập nhật không rollback để thấy khả năng giữ kiến thức của thuật toán. API sản phẩm có development guard và rollback; guard có thể từ chối học chứ không chữa được forgetting.
6. Byte tokenizer chỉ là plumbing dễ kiểm thử và không mất UTF-8 đầu vào. 1M byte token không tương đương 1M subword token. Output ngẫu nhiên có thể không phải UTF-8 hợp lệ.
7. Native C++ là kernel qua ctypes, có SIMD trên máy x64 hiện có. Đây chưa phải runtime C++ độc lập. Giữ NumPy mặc định khi ctypes chậm hơn ở mạng nhỏ.
8. Không triển khai iGPU/quantization/model lớn trước khi có bằng chứng học cục bộ đáng mở rộng. Không thay sang LLM pretrained để đổi tên thành thành công SCAR-X.
9. RAM là số đo process-tree và high-water mark; sampling không phải giới hạn cứng. Không tự tắt pagefile hệ điều hành. Chưa chứng nhận toàn hệ thống dưới 2 GB trên mọi workload.
10. Run bất biến, lưu manifest/hash và từ vòng xác nhận có source.zip. Chưa đủ chứng cứ thì giữ trạng thái chưa kiểm chứng.
