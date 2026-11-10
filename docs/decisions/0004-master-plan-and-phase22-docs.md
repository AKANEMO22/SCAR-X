# ADR0004 — Master plan và phạm vi Phase 2.2 đề xuất

Ngày: **2026-11-09**. Trạng thái: quyết định tài liệu đã được người dùng phê duyệt; triển khai chưa được phê duyệt.

## Bối cảnh

Sau audit và kết quả Phase2.1, người dùng phê duyệt cập nhật docs, chỉ định `docs/RESEARCH_PLAN.md` là master plan duy nhất. Yêu cầu bổ sung nhấn mạnh SCAR-X là nghiên cứu kiến trúc neural, không thu hẹp thành hệ memory/RAG cho LLM. Không sửa code hoặc chạy experiment từ phê duyệt này.

## Quyết định

- Roadmap/vision/gates hiện hành tập trung tại [master plan](../RESEARCH_PLAN.md). README/MAP/STATUS là điều hướng/tóm tắt; tài liệu architecture/metrics/design là tham chiếu; protocol đã chạy vẫn bất biến.
- Giữ SNN với LIF/ALIF và local synaptic learning, CPU-first, budget2GB bao gồm học/consolidation/rollback, không tăng model khi gate chưa đạt.
- Representation, credit assignment và readout stability là ưu tiên khoa học trước việc tối ưu replay/consolidation. Importance phải được đánh giá theo ích lợi và plasticity, không chỉ hoạt động lớn.
- Experience-to-Skill là giả thuyết. Tách neural core tạo representation/kỹ năng, memory giữ trải nghiệm và controller chọn/xác minh/điều phối. Retrieval không là bằng chứng synapse đã học.
- P0 đề xuất gồm compatibility `plasticity-v1`, snapshot learner/replay/RNG nhất quán, verifier không dựa vào verified flag, budget toàn session. Đây là prerequisites trước automatic consolidation, chưa được sửa trong code.
- Roadmap đề xuất P0→P1/P2→P3→P4→P5; finite stream20/50/100 không chứng minh lifelong vô hạn. Mọi triển khai cần duyệt kế hoạch/protocol trước.
- 1M context/parameter count không là mục tiêu; số liệu và protocol lịch sử giữ nguyên. NO-GO cho tăng quy mô ở rule hiện tại vẫn hiệu lực.
- Đề xuất Git local tại thư mục hiện tại, chưa init/commit/push/di chuyển tệp. Data policy và scheduler cần duyệt riêng khi triển khai tương ứng.

## Hệ quả và vấn đề mở

Tài liệu cần phân biệt code đã có, kết quả đã đo, lỗi tích hợp chưa sửa và đề xuất. Không ghi Phase2.2 là đã đăng ký/chạy. Không đổi thresholds lịch sử; endpoints mới chỉ có hiệu lực sau đăng ký trước experiment mới.

Các ADR0001–0003 không bị sửa. Tài liệu mới giải quyết mơ hồ về nguồn kế hoạch, không thay kết quả hoặc tuyên bố đã có semantic/procedural memory. Scope phần việc này kết thúc ở docs và kiểm tra tính nhất quán/liên kết của docs.
