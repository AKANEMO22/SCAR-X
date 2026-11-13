# ADR0015 — Hybrid nguyên precision và học chủ động

Ngày 13/11/2026, người dùng phê duyệt thực hiện plan Hybrid và nối Hermes Desktop.

- Một assistant qua một gateway loopback; Qwen3.5-0.8B frozen để nhìn/ngôn ngữ,
  Whisper base FP32 và một ALIF H128 để nghiên cứu local policy/relevance learning.
- Original SCAR-X serving/neural code, historical runs và active peer branches
  không đổi. Ngoại lệ LLM-serving chỉ áp dụng `scarx.hybrid` và entrypoint riêng.
- 3.000.000.000 byte toàn process tree; cộng observer và Hermes khi nghiệm thu
  tích hợp. Báo RSS và private commit riêng. Windows Job aggregate commit và
  sampled RSS không được gọi là hard physical-RAM guarantee.
- Không quantization/pruning/lower precision, không LoRA. BF16/F32 native tensors
  và conversion/readback được ghi bằng source hashes và audit riêng.
- Model-generated uncertainty chỉ tạo learning goal, không tạo factual label.
  Trọng số chỉ học từ human feedback có identity hoặc machine verifier.
- Một H128 có hai masked readout heads (relevance2 + actions6); feature64 fixed
  signed hashing. Đây là adapter mới cần kiểm chứng, không kế thừa scores H128D8.
  Importance được lưu ở trạng thái dự phòng, chưa chứng minh metaplasticity.
- Chính sách neural chọn hành động trong tập quyền runtime cho phép. Clock,
  priority, privacy, rate/RAM limits thuộc runtime, không thuộc system prompt.
- User vắng: vẫn review/tra cứu trong ngân sách, không thông báo; khi trở lại
  gom câu hỏi vào tab Hỏi để học. Một pending; ≤3 chủ đề/giờ, ≤12 search/giờ,
  ≤2 vòng tìm hiểu/goal, CPU idle trung bình≤10%, user foreground ưu tiên.
- Web chỉ public topics được người dùng cho phép; model/web không tự gắn quyền
  public cho dữ liệu riêng tư. Bản đầu MediaWiki vi/en và linked public sources.
- SSD-owned raw archive≤10GB/7days; không xóa original user source. Metadata và
  audit được giữ; clip bị xóa có tombstone/source-unavailable.
- Checkpoint bao gồm neural/eligibility/optimizer/RNG/replay/controller/goals/
  pending questions/cursors; rollback không xóa append-only evidence/audit.
- Giao diện và observer chỉ đưa dữ liệu thực; phân biệt heuristic/neural_policy/
  orchestrator. Không giả văn bản suy nghĩ nội tâm.

P0→P3 và scientific thresholds xem `experiments/hybrid-v1.md`. P0 failures phải
được sửa/đo lại trước P1/P2/P3; không mở scaling hoặc ngầm thay precision.

Ponytail: không có tool/rules plugin trong inventory của phiên triển khai;
đã ghi unavailability trong observer, không tuyên bố đã load plugin.
