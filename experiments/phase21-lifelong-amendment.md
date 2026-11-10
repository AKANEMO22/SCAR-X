# Phase 2.1 bổ sung trước lần chạy đầu — Lifelong Adaptive Intelligence

Ngày2026-11-08, theo cập nhật của người dùng trong khi chưa chạy Phase2.1. Không sửa threshold trong phase21-v1. Bỏ mục tiêu1M context/token length; các số đo cũ vẫn là hồ sơ lịch sử.

Thêm stream không task ID/không boundary được đưa cho learner:1280 experience, mixture XOR/order/binding thay đổi trơn và quay lại,20 cửa sổ đánh giá chỉ evaluator biết. Consolidation sau64 cập nhật, không theo ranh giới task. Reservoir128 mẫu, replay một mẫu sau mỗi hai experience; capacity và presentation count cố định.

8 confirmation seeds giữ nguyên. So7 condition: readout+EMA; score+EMA; score+metaplasticity; score+meta có importance decay; selective learning dựa reliability metadata; random rejection cùng xác suất trung bình; periodic readout-SI. SI chỉ dùng gradient tại readout tuyến tính, hidden cố định, không backprop toàn mạng. Đây là adaptation của published SI vào cùng SNN, không tuyên bố tái lập paper đầy đủ. Review MESU/BNN metaplasticity/Continual Backprop làm so sánh lý thuyết; chưa được suy ra thắng phương pháp chưa chạy.

20% nhãn training bị đảo; nguồn reliability có sensitivity0.7 với nhiễu, false rejection0.1 trên nhãn sạch (tổng rejection kỳ vọng0.22). Metadata không phải oracle: giữ một số noise và loại một số sạch. Condition selective được thêm nguồn thông tin chất lượng; lợi ích phải so với random-drop0.22 và không được gán hết cho neural rule. Khi bỏ qua sample, dùng replay đã có cho update slot; report accepted/rejected/noise retained và số update. Test luôn sạch, hidden không nhận clean label.

Metaplasticity-decay giảm importance0.995 mỗi32 update, để kiểm đánh đổi stability/plasticity. Fast weights cập nhật mỗi experience; stable anchors EMA sau64update. Không sinh neuron/tăngcapacity. Kiểm probe học mới cuối stream bằng64 sample binding disjoint; so improvement với chính model trước probe và fresh model cùng64 sample. Đo mức biến đổi weights, feature rank và đơn vị hoạt động ở đầu/cuối.

Primary stream endpoints: clean accuracy của3 families qua20 windows; forgetting max-to-final<=3pp trung bình và<=5pp worst cho mỗi seed, final accuracy>=65%. Test không dùng chọn condition hoặc rollback. Báo từng condition, seed-bootstrap và Holm cho các contrast primary score-meta vs score-EMA, decayed-meta vs meta, selective vs random rejection; outcome không đạt representation/retention vẫn PIVOT. Không coi không-học/đóng-băng là đạt retention.

Budget: cùng48neurons/8fan-in/12plastic rows, memory buffers bằng nhau; cùng1280 presented và replay slots. Báo real CPU time và actual updates; không gọi đây là exact time-matched nếu các condition khác accepted updates. Replay/consolidation factorial trong phase21-v1 vẫn có thêm xác nhận common wall-time budget. Mọi kết luận lợi thế soexistingmethod chỉ trong cùngbenchmark/budget đã đo.

Backprop comparison (module control riêng): surrogate-BPTT trên3 representation tasks với8seed,512train/256test; cùng sparse topology và ALIF deterministic, khác stochastic forward củalocalbranch phải công bố. CPU only; không dùng kết quả để tune localbranch. Không cần triển khaiC++/iGPU.

Tài nguyên dự kiến tổng10–45 phútCPU, mỗirun dừng3h hoặcRAM>2GB; không chi phí cloud. Nghiên cứu thời gian hữu hạn không chứng minh học vô hạn ở capacity hữu hạn.
