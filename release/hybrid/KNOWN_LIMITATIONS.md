# Giới hạn của RC

- Lượt small mixed 8 giờ đã dừng theo yêu cầu người dùng sau khoảng 3 giờ 8 phút.
  Log, 38 chu kỳ và 7 checkpoint được giữ; không đạt hoặc thay thế cửa 8 giờ.
  Việc cho phép cài RC theo yêu cầu mới không phải giảm tiêu chí nghiên cứu.

- Chất lượng chat/vision của VLM 0.8B còn yếu; có thể hiểu sai, trả lời tự tin
  khi thiếu căn cứ hoặc dùng công cụ chưa hợp lý. Không dùng cho quyết định
  cần độ chính xác cao mà không kiểm tra nguồn.
- Chưa chứng minh hidden SNN vượt cùng LLM + RAG, học kỹ năng lâu dài hay giữ kiến
  thức trong ngưỡng forgetting đã đăng ký. Số cập nhật không bằng số kỹ năng.
- LoRA tự động, learner mới, deep thinking/capacity thích ứng tổng quát chưa đạt
  kiểm chứng và không được bật trong đợt release này.
- Video dùng frame lấy mẫu và ASR theo đoạn, không đảm bảo thấy mọi sự kiện.
  URL/description không đồng nghĩa đã xem toàn video. Không tuyên bố hiểu
  video dài dựa vào test lặp một clip ngắn. Xem màn hình cần quyền/cửa sổ được chọn;
  không mặc định theo dõi toàn máy hoặc microphone.
- Verifier số học chỉ hỗ trợ mẫu hẹp; parser không chứng minh đã hiểu đúng
  ý định. Guard về khả năng công cụ không phải SNN tự học nhận ra mọi lỗi.
- Context thực vẫn 16K token. Bộ nhớ SSD và nhật ký phân trang không tạo context
  vô hạn hoặc đảm bảo nhớ mọi chi tiết qua synapse.
- SearXNG chạy native Windows, vẫn cần Internet/engine tìm kiếm bên ngoài.
  Có thể timeout, bị giới hạn hoặc không có kết quả; không giả nguồn đúng.
- Gói này chỉ cập nhật deployment Hermes/runtime đã cài và có fingerprints phù
  hợp. Không là fresh-machine standalone, không chứa model/venv hoặc thay app
  HermesDesktop. UpdateHermes/runtime khác có thể cần kiểm tra tích hợp lại.
- GPU mode không đặt trần RAM hệ thống theo phê duyệt; VRAM vật lý vẫn có giới hạn.
  CPU mode 4,5 GB chưa có cùng độ phủ nghiệm thu với RC GPU này.
- Overlay là CAS từng file có backup/maintenance lease, chưa phải hệ update chống
  mất điện hoàn chỉnh. Nếu bị dừng giữa giao dịch, giữ backup và kiểm/rollback;
  tuyệt đối không phục hồi toàn DB để xóa journal/trải nghiệm mới.

Đây là **Release Candidate về vận hành**, không là production/lifelong PASS,
không ngang AI thương mại và không thay kết quả nghiên cứu thất bại trước đó.
