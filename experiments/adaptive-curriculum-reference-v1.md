# Development reference: actual SCAR-X copy / Qwen3-4B

2026-11-10, khóa trước output. Theo master product acceptance/independent Qwen;
không learner U, không training, không scale hoặc thay main. Đây là development
đã review, không final generalization/retention test.

- Nguồn `artifacts/adaptive-curriculum-reviewed.json`, kiểm sha trước chạy.
- Cố định 24case: mỗi6chatfamily ×2language lấy id01,id02, không chọn theo output.
- SCAR-X: SQLite onlinebackup nguồn hiện tại; model copy, canonical state reset
  mỗi câu, không retrieval/learning. Giữ source vocabulary,encoding,topology.
  Báo riêng số đo này với live continuous state; không giả là hành vi live.
- Qwen/Qwen3-4B-Instruct-2507 Q4_K_M đã có, kiểm file size/SHA256. CPU2threads,
  ctx1024, temp0, seed919, max128outputtokens. Khác ngân sách training/RAM/decoder;
  không suy ra tương đương hay chấm identity Qwen theo literal SCAR-X.
- Qwen request chỉ singlepublicuserprompt, không target/family/SCAR-Xoutput.
  Chỉ là đối chiếu đầu ra, không dùng làm nhãn hay proxy; giữ truncation/errors.
- Qwen/reference process-tree cap4.5GB, không là chứng nhận SCAR-X dưới2GB.
  SCAR-X copy chạy trước reference; không nạp 9B, không đụng Hermes config.
  Không chạy nếu đã có llama-server khác. Chỉ dừng owned subprocess mình mở.
- Startup90s, tổng420s, mỗirequest30s. No retry quality/selection. Hash source
  weights trước/sau đo để phát hiện mutation, không restore phiên thật.
- Raw24đầu ra và trace timestamp lên localhost. Không sinh một điểm intelligence
  từ substring hoặc tự chấm Qwen là đúng. Manual rubric cho từngintent báo riêng
  nếu thực hiện sau, không tự thành nhãn học hoặc đổi ngưỡng P1/P2.
- Vision32wording chưa có ảnh mới, không gửi Qwen4Btext-only như visionmodel.
