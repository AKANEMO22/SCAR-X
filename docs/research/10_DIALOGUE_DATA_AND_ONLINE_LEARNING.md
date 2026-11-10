# Dữ liệu hội thoại, ảnh và vòng học liên tục

Ngày cập nhật: 2026-11-09. Đây là báo cáo triển khai, không thay protocol nghiên
cứu hoặc master [RESEARCH_PLAN](../RESEARCH_PLAN.md).

## Những gì đã chạy

Đã gọi Antigravity CLI1.5.3 trên máy, model `gemini-3.8-flash-high`, bằng tài khoản
đã đăng nhập. Chỉ gửi brief tạo dữ liệu; không gửi transcript cá nhân hoặc nhập
weights của model teacher. Raw CLI output lưu riêng trong `local/`, không đóng gói.

- [Candidate đầu](../../data/candidates/antigravity-20261109T022847Z/validation.json):
  240 ví dụ thô;217 giữ lại,23 cách ly;178train/39development.53 đáp án số học
  kiểm bằng parser và phép tính độc lập.164 nhãn rubric không được gọi là verified.
- [Candidate bổ sung](../../data/candidates/antigravity-supplement-20261109T063155Z/validation.json):
  40 hội thoại, mỗi hội thoại3 cặp lượt,141 target vocabulary tokens.32 hội thoại
  tiếng Việt thuộc train;8 tiếng Anh thuộc development, đồng thời giữ riêng
  scenario. **Confound ngôn ngữ và scenario:** chưa dùng bộ này suy ra bilingual
  generalization; không sửa split để chọn kết quả đẹp hơn.
- Hai ảnh thật công khai từ scikit-learn, kèm attribution CC-BY2.0,16 visual QA
  và2 caption. CLI log có hai `read_file` hoàn tất đúng file và hash. Codex đã
  xem ảnh: vật thể/màu sắc/quan hệ rộng khớp, nhưng hướng ánh sáng, ánh sáng tự
  nhiên và loại thuyền xa không được xác minh. Giữ nhãn candidate, không auto-learn.

Hai ảnh không tạo thành benchmark thị giác. Core hiện nhận64 pixel chữ số và
readout10 lớp; chưa có encoder ảnh tự nhiên, spatial representation hoặc đường
học liên kết ảnh–từ. Không thể biến captions thành khả năng nhìn chỉ bằng import.
Không công bố SCAR-X xem được ảnh bình thường hoặc thao tác desktop.

## Học trong hội thoại và lúc rảnh

Một `Generation/UnifiedCore` sống xuyên lượt. Controller giữ buffer hữu hạn gồm
input, neural draft, nhãn và nguồn; tạo câu hỏi khi thiếu hướng dẫn. Trả lời rõ
ràng tạo lesson, vài local microstep chạy giữa lượt, rồi idle review so với trải
nghiệm cũ. Draft chưa kiểm chứng chỉ dùng phát hiện khác biệt, không là nhãn.
Không restart learner theo chat, không reset live neural state hoặc huấn luyện
lại toàn mạng sau từng cuộc hội thoại. Epoch chỉ dùng ở các thí nghiệm bootstrap
lịch sử, không là đơn vị vận hành lifelong.

Guard so với probe phát triển từng đúng, rollback fullstate nếu mất một probe;
tối đa32 thử/lesson. Cơ chế này có thể bảo vệ những câu cũ bằng cách **không học
được câu mới**. Nhãn human là nguồn hướng dẫn, không tự chứng minh factual truth.
Arithmetic verifier chỉ kiểm những mẫu parser hiểu được, không hiểu mọi ngôn ngữ.

[Acceptance mới](../../runs/20261109T064249Z-continuous-dialogue-acceptance-ce1b2c/report.md)
dùng một đáp án human fixture và8 ví dụ arithmetic đã kiểm, trong learner cô lập:
15 microstep accepted,116 rejected,9 lesson, giữ cùng object, snapshot exact và
continuation neural/RNG khớp. `1+1` vẫn sinh “Bạn có..”. Chưa có acquisition,
transfer hoặc retention thành công; không dùng recall training point làm chứng
minh representation. Corpus và weights clone không thay phiên người dùng.

## Hermes và cách tái lập

Hermes gửi latest user + sessionID/requestID đến một core. Câu hỏi idle controller
dùng native clarify; terminal event `clarify.resolved` mang requestID và metadata
để kết thúc ngay cả khi respond acknowledgement đã xóa pending store trước đó.
Pending mirror ở compute host cũng được giải phóng, tránh hiện lại câu đã trả lời.
Không cần một vòng gọi LLM/tools tiếp theo để đóng thẻ. Câu hỏi controller được
ghi rõ nguồn, không mô tả là neural reasoning đã học.

GUI đã kiểm trực tiếp: trả lời6 cho câu QA3+3, thẻ chuyển thành kết quả đã trả
lời; gửi tiếp `hi` nhận `Hello.`. Footer hiển thị2 ký tự/64 neuron. Đáp án6 là
fixture QA đã xác định, không phải nội dung do người dùng tự nhập. Đây là bằng
chứng hết kẹt UI, không phải bằng chứng model đã học được phép cộng.

92 SCAR-X tests và4 CPU/native parity tests qua;6 Python Hermes integration và
5 UI tests mục tiêu qua; real AIAgent nonstream/SSE đều qua. Kiểm rộng thêm
`test_compute_host_phase1.py` có8 qua/1 fail ở concurrent log append (31 thay
vì32 dòng). Đó là đường logger không được patch trong đợt này; chưa giải quyết
lỗi đó và không tuyên bố toàn bộ suite Hermes đã qua.

```powershell
python scripts/validate_antigravity_dataset.py
python scripts/validate_antigravity_supplement.py
python scripts/verify_continual_dialogue.py
python -m pytest -q
```

Validator supplement cần raw job log trong `local/` để xác nhận native reads;
gói công khai chứa validation receipt, ảnh và dữ liệu nhưng không có log riêng.
Mỗi acceptance tạo run mới. Không ghi đè run cũ hoặc thay ngưỡng nghiên cứu.

## Quyết định tiếp theo

**PIVOT cơ chế học, chưa tăng quy mô.** Ưu tiên representation/credit assignment
và cân bằng acquisition–readout stability. Cần bài temporal/binding không shortcut,
hidden ablation và đánh giá liên tục không task boundary theo gate đã đăng ký.
Nhánh vision tiếp theo phải thử encoder ảnh nhỏ nối cùng core, so với ablation
ảnh bị xáo/che và held-out ảnh khác nguồn. Hai ảnh annotation hiện tại chỉ kiểm
pipeline dữ liệu. Dataset lớn hơn không tự giải quyết giới hạn thuật toán.
