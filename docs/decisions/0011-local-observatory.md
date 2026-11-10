# ADR0011 — Quan sát cấp phát và thí nghiệm trên localhost

Ngày 09/11/2026. Người dùng yêu cầu nhìn thấy neuron tự giãn theo task và mọi
lượt thử hiện trên trang để theo dõi công việc đang làm.

## Quyết định

- Mở rộng website localhost hiện có; không framework/host mới hoặc model ID
  thứ hai. Ba khu vực: cấp phát theo task, thí nghiệm, lõi đang trò chuyện.
- Chuỗi task quan sát có thể chạy trên core live hoặc checkpoint clone, ghi
  active/allocated/dormant/growth, số tick và trace thật. Trên live, lưu checkpoint
  trước lượt, giữ cấp phát mới, nhường foreground và cho dừng. Không ghi các
  task này thành human experience, lesson hoặc câu hỏi cần người dùng trả lời.
- Heuristic capacity hiện hành không bị đổi thành “đã học độ khó”. Nhiều neuron
  không đồng nghĩa quality; không gọi 0,9 giây chờ hiển thị là neural thinking.
- Kết quả cấp phát cuối lưu riêng trong data-dir, bounded 1,5 MB. Chỉ nhận bản
  terminal hợp lệ khi phục hồi; cache hỏng không thay neural checkpoint.
- `new_run` đăng ký telemetry; `write_json` ghi completion/progress; runner có
  thể công bố mẫu neural. Kiểm PID + create-time; chỉ process chủ sở hữu ghi
  journal. Không tự spawn/restart nghiên cứu từ API đọc hoặc khi journal cũ.
- API read-only báo cáo chỉ đọc `runs/<validated-run-id>/report.md` có giới hạn;
  mutation capacity giữ Host/Origin/token hiện có. Không thêm shell endpoint,
  mở bind mạng, nạp external code hoặc công bố dữ liệu lên cloud.
- Nội dung thí nghiệm tách khỏi state model live. File journal/trace không là
  training label, memory retrieval hoặc bằng chứng học representation thành công.

## Kiểm chứng

Tests kiểm cùng object core, growth thật, dormant compute, old weights không đổi,
không lesson/question giả, cache qua restart, mutation auth, report traversal,
bounded journal và PID reuse. UI phải được quan sát trực tiếp, không coi JSON
hoặc test backend là đủ chứng minh trang hiện đúng.

Xem [hướng dẫn quan sát](../OBSERVING.md). Quyết định không thay research gate,
public release weights, cơ chế học hoặc stop rules của các ứng viên đã thất bại.
