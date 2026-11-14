# SCAR-X Hybrid — bản RC đang cài trên máy

**Đã cài và chạy trong chat Hermes bình thường**, model ID `scarx-hybrid`.
Không cần chuyển sang tab Debug để trò chuyện. Gateway local: `127.0.0.1:8775`.
Một worker Qwen3.5-0.8B nguyên BF16/F32; Whisper-small FP32 và ALIF H128 trên
GPU. Không LoRA, quantization, pruning hoặc thay cơ chế học.

Gói hiện tại: `artifacts/releases/SCAR-X-Hybrid-1.0.0-rc.3+ce24c185bda2.zip`,
796.057 byte, 50 payload files. SHA256:
`3f03da4dfc08f5d604258e3dfef524ce50c1221793e794b218d5a845549f1a94`.
Git SHA nguồn gói: `ce24c185bda2834d1a79361ab95aaedc150c0eb5`.
Sidecar `.installed-verification.json` đã xác nhận payload khớp bản đang cài,
64 asset pins/1.159 file Hermes/dependency đều khớp; không restart hoặc gọi model
trong phép kiểm cuối. Main vẫn ở SHA cũ, không merge hoặc push.

## Dùng ngay

- Chọn **Scarx Hybrid** trong chat Hermes như model trước đây.
- Ảnh/video: dùng menu đính kèm sẵn có; hỏi tiếp trong cùng cuộc trò chuyện.
  Video được lấy mẫu hình và phiên âm từng đoạn, không thấy mọi sự kiện.
- Muốn xem video trên màn hình: yêu cầu trong chat và xác định cửa sổ cần xem.
  Không mặc định theo dõi toàn màn hình hoặc microphone.
- Trace tại `http://127.0.0.1:8775/?tab=trace`; tab Debug chỉ quan sát.
- Gói cập nhật là code overlay cho deployment đã cài trên máy này, không chứa
  trọng số/venv/dữ liệu riêng và chưa là bộ cài cho máy mới.

## Đã sửa và kiểm thực tế

1. **Video không được gắn vào câu hỏi:** Hermes gửi đúng mã transcript qua
   `X-SCARX-Session`, giữ ngữ cảnh video ở metadata; không chèn link kỹ thuật
   vào lời người dùng. Clip thử đã nạp qua menu Files trong Hermes.
2. **Từ chối cả lời chào:** đã tái hiện với đúng system prompt + 8 schema công
   cụ. Host chỉ bỏ schema không cần thiết ở lời chào thuần túy, vẫn giữ nguyên
   system prompt/history. Tác vụ thật, ảnh và yêu cầu dùng tool giữ công cụ.
   Không hardcode câu trả lời hoặc đổi trọng số để che lỗi.
3. **Truy xuất mất cuối câu hỏi:** full-query P0 đã cài, giới hạn 6.000 ký tự.
4. **Chọn checker không hỗ trợ:** capability guard P0 đã cài. Parser vẫn hẹp;
   đây không phải neural policy đã tự hiểu nguyên nhân mọi lỗi.

Hermes thật đã trả lời “Bạn có muốn tôi giúp gì không?” cho lời chào và xác
định hình đầu clip màu đỏ tại 5 giây. Clip engineering 21 giây đã biết, không
là tập chất lượng giữ kín. Ba ASR calls thật xác nhận FP32/CUDA0, frontiers
thu nhận/ASR/vision đều 21 giây. Web search qua dispatcher Hermes + SearXNG
native trả về tài liệu chính thức, không lỗi.

Đã qua **140 kiểm tra Python có phạm vi**, 27 kiểm tra Node; loại rõ native CPU
parity case trong lượt Python này. Startup native có 3 parity cases thật qua.
Snapshot trước/sau activation xác nhận các mảng, RNG/replay và 5 cập nhật live
được giữ. Không nhập state của lab vào model đang phục vụ.

[Bằng chứng read-only](evidence/RC3_OPERATIONAL_SMOKE_20261114.json) tách việc
đã đo khỏi khả năng chưa chứng minh. Source audit tại
`scripts/audit_hybrid_rc_operational.py`; raw prompt/checkpoint/log ở vùng private.
Verifier artifact-to-install: `scripts/verify_installed_hybrid_rc.py`.

## Test 8 giờ đã hủy

Theo yêu cầu mới, đã dừng sạch lượt `32cd13` sau 11.268,281 giây, giữ 38 chu kỳ,
7 checkpoint và result gốc. Không worker private sót, không restart lại test.
Cancellation audit ghi **USER_CANCELLED_NOT_EIGHT_HOUR_PASS**. Không gọi đây là
đạt độ bền 8 giờ hoặc thay ngưỡng nghiên cứu.

## Những gì chưa đạt

- Chat/vision 0.8B còn yếu; kiểm một clip/lời chào không chứng minh chất lượng
  đa năng hoặc video chưa thấy. Không có cam kết ngang model online.
- SNN chưa chứng minh vượt cùng LLM + RAG, chưa đạt retention đã đăng ký, chưa
  tự học chắc chắn nguyên nhân tool loop và kỹ năng sửa lỗi áp dụng lâu dài.
- Guard, retrieval, chọn tool ở host là sửa kỹ thuật; không ghi thành neural
  improvement. LoRA tự động và learner mới vẫn tắt/backlog.
- Save/load/rollback state có kiểm tra cô lập và state live đã giữ qua restart;
  không thử rewind cuộc trò chuyện riêng đang mở. Code rollback không ghi đè DB.
- Chưa nghiệm thu lại toàn bộ UI compaction phiên dài, thao tác capture màn
  hình và mọi workflow project/tool trên RC cuối. Các kiểm tra cơ chế đã có,
  nhưng không thay bằng tuyên bố full UI PASS.

Đây là **RC dùng thử có giới hạn**, không phải production hoặc lifelong PASS.
Các việc còn lại là sửa lỗi sử dụng có thể tái hiện và nghiệm thu workflow thực
tế; không mở thêm nghiên cứu/dataset/test 8 giờ trong đợt bàn giao này.
