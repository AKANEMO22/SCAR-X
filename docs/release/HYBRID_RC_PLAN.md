# SCAR-X Hybrid — kế hoạch phát hành RC

Ngày 14/11/2026: ưu tiên hoàn tất một bản dùng thử local trong chat Hermes bình
thường. Dừng mở dataset, benchmark nghiên cứu, model hoặc learner mới; giữ các
nghiên cứu chưa xong trong backlog. Không sửa main, R13 hoặc kết quả lịch sử.

**Cập nhật yêu cầu mới:** người dùng yêu cầu bỏ test 8 giờ để ưu tiên sản phẩm.
Đã hủy sạch `32cd13`, giữ result gốc và cancellation audit; không gọi là PASS.
Điều kiện cài RC chuyển sang custody hủy theo yêu cầu + kiểm vận hành cần thiết.
Các ngưỡng nghiên cứu/retention và kết quả lịch sử không đổi.

## Checklist hữu hạn

| Cửa | Điều kiện | Trạng thái |
|---|---|---|
| 1. Khép lượt bị hủy | Actual terminal, giữ result, không residual worker, source khớp trước activation | Đã hủy sau 11.268,281 giây; 38 chu kỳ/7 checkpoint, không PASS |
| 2. Gói cập nhật | Allowlist, SHA, closure import, loại dữ liệu riêng, installer/rollback | RC1 đã kiểm; RC2 cập nhật custody theo yêu cầu, 31 safety/P0 qua |
| 3. Áp dụng | Sau cửa 1, gộp hai bản sửa P0, backup checkpoint, CAS, maintenance lease | Chưa áp dụng lên phiên đang chạy |
| 4. Nghiệm thu | Chat/ACK, workspace, tool có giới hạn và dừng được, compaction, search, ảnh/video, GPU, persistence | Cần nghiệm thu sau cập nhật; không gọi là production PASS |

## Cấu hình và phạm vi

- Giữ model ID `scarx-hybrid`, một gateway tại `http://127.0.0.1:8775`, chat
  Hermes bình thường; Debug chỉ quan sát.
- Giữ Qwen3.5-0.8B nguyên BF16/F32, Whisper-small FP32 và ALIF H128 học cục bộ
  trên GPU. I/O, SQLite và điều phối chạy trên CPU. Không bật LoRA tự động.
- Gói đầu tiên chỉ cập nhật deployment Windows/Hermes **đã cài và được kiểm**.
  Tái sử dụng model/runtime hiện có; không phải bộ cài độc lập cho máy mới.
- Có install, verify, start, stop, rollback và manifest phiên bản/Git SHA.
  Không chứa model, venv, cache, SQLite, token, checkpoint, hội thoại hoặc
  dataset/log nghiên cứu. Không dùng `package_generation1.py` cho Hybrid.
- Chỉ hai sửa runtime đã qua P0: truy xuất toàn câu hỏi trong giới hạn 6.000
  ký tự (`7a9ee0`) và bỏ action verify khi checker không hỗ trợ (`8e205a`).
  Đây không phải thay quy tắc học hoặc bằng chứng hidden synapse học kỹ năng.

## Triển khai và khôi phục

Installer kiểm registration, PID/create-time/cmdpath, source trước/sau và các
pin model/runtime/Hermes. Khi thí nghiệm còn chạy, từ chối trước stop/write.
Sau terminal và audit hủy hoặc audit đầy đủ, lấy maintenance lease, dừng đúng owner, sao lưu checkpoint
đầy đủ rồi áp dụng code theo CAS. Backup ở vùng private ngoài gói.

Rollback chỉ phục hồi code khớp SHA; không ghi đè SQLite, rewind checkpoint
active hoặc xóa trải nghiệm mới. Nếu nguồn có sửa tiếp, từ chối để bảo toàn sửa
đổi đó. Cập nhật từng file chưa chống mất điện hoàn chỉnh; startup lỗi phải báo
rõ và dùng backup khôi phục, không ép kill hay xóa dữ liệu để chạy.

## Backlog và giới hạn kết luận

Metadata intake `d14797`, nghiên cứu credit/representation/retention, history
ablation chưa duyệt và LoRA vẫn ở backlog. Không mở rộng trong lượt release.
Hidden SNN chưa chứng minh vượt cùng LLM + RAG; forgetting chưa đạt ngưỡng.
Giữ nguyên tiêu chí nghiên cứu và kết quả không đạt.

Antigravity được dùng với `gemini-3.8-flash-high`, effort `high`, chỉ nhận code
và mô tả công khai. [Biên bản review](ANTIGRAVITY_RELEASE_REVIEW_20261114.md)
phân biệt ý kiến tham khảo với kiểm tra độc lập.

[Artifact và bằng chứng RC đã chuẩn bị](HYBRID_RC_CANDIDATE_20261114.md).
