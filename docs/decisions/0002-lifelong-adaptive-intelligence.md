# ADR0002 — Lifelong Adaptive Intelligence

2026-11-08, theo yêu cầu người dùng trong Phase2.1.

- Bỏ1M context và token length khỏi mục tiêu nghiên cứu. Báo cáo Phase2 cũ giữ nguyên như lịch sử, không còn là tiêu chí nghiệm thu.
- Mục tiêu mới: học kỹ năng qua thời gian, retention, adaptation, duy trì plasticity và phân bổ memory hữu hạn.
- Giữ neuron phát xung, synapse học cục bộ, không backward toàn mạng trong nhánh chính; CPU với RAM<=2GB, không tăng48neuron hiện tại.
- Kế hoạch gốcPhase2.1 đã ghi trước chạy, bổ sung lifelong amendment cũng trước lần chạy đầu. Giữ ngưỡng3pp mean/5pp worst forgetting và kiểm soát acquisition.
- Context-labeled stress test chỉ dùng chẩn đoán shared-head và replay. Stream mới không cung cấp task ID/boundary cholearner; periodic consolidation theo update clock.
- Selective learning từ quality metadata là thêm thông tin quan sát được, không phải khả năng tự biết chân lý. So thêm random rejection để tránh gán mọi lợi ích cho plasticity.
- SI-readout là control từ synaptic intelligence áp dụng ở output-local; whole-network SI/MESU/BNN metaplasticity/Continual Backprop chưa được tái lập nên không được tuyên bố vượt chúng.
- Nghiên cứu finite stream không chứng minh học vô hạn. Capacity, chất lượng representation và credit assignment là giới hạn độc lập.
