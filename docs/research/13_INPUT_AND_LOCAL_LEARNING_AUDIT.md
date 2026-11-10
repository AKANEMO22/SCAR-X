# P1/P2 — Thông tin đầu vào, hidden learning và readout

2026-11-09. Theo masterP1/P2, cùng core ALIF nhỏ, không model/framework mới,
không tăng quy mô, không thay live weights. Mục tiêu chat/vision bình thường
vẫn chưa đạt. Các prototype dưới đây chưa được bật trong Hermes.

## 1. Giới hạn trước khi tín hiệu vào neuron

[Audit](../../runs/20261109T083101Z-p1-input-information-audit-v1-1a6bac/report.md)
dùng128cặp câu có hai cập nhật đảo thứ tự: đáp án “giá trị mới nhất” khác nhau,
nhưng encoder G1 cho vector **giống hệt128/128**. Cùng neural state ban đầu tạo
cùng output. Trong các cặp đó, downstream deterministic không thể trả lời đúng
cả hai chỉ bằng tăng tick hoặc học thêm. Không suy rộng sang mọi câu tiếng Việt.

Hai glyph cùng hình nhưng RGBđỏ/xanh khác nhau cũng thành cùng64pixel xám;
vision hiện trả10digit classes, không chạy word decoder. Đây là giới hạn đường
vào/đường ra, khác với chưa đủ dữ liệu. Chưa kiểm chứng normal-photo recognition.

## 2. Prototype event encoder

`neural/input_events.py` biên dịch token theo thời gian, có speaker/boundary và
position; RGB16×16thành16patch4×4RGB trong64image channels hiện có. H48không
đổi, không pretrainedencoder. Giới hạn192token/message,12messages,256events;
quá giới hạn thì báo lỗi, không âm thầm cắt lịch sử. Không thêm nhãn vào input.

[V1](../../runs/20261109T092440Z-p1-input-events-v1-706f04/report.md) vẫn2/128cặp
bị trùng vì single signed hash bucket. Giữ failure; [V2đăng ký riêng](../../experiments/p1-input-events-v2.md)
dùng128digest bits và trigram, cùng channels. [V2](../../runs/20261109T092630Z-p1-input-events-v2-8d2f57/report.md)
không trùng384cặp, gồm256cặp mới seed29107; hai ảnh khác màu nay khác sequence.
Nếu lại sum/pool mọi events, cả384cặp lại mất thứ tự: learner phải nhận sequence.
Không bảo đảm hash injective với mọi chuỗi; resize vẫn mất chi tiết không gian.
Đây mới là information-preservation, chưa học ngôn ngữ/vision hay thay frontend live.

## 3. Hai vòng hidden local rule: dừng nhóm ứng viên

[V1local target-rate](../../experiments/p1-local-target-v1.md) dùng eligibility
riêng tại synapse và target-rate code cố định tại neuron. [V2random feedback](../../experiments/p1-random-feedback-v2.md)
thay modulator bằng projection cố định của output error. Không W_out transpose,
không truyền error qua hidden khác hoặc ngược thời gian; có supervised target/
error broadcast và pseudo-derivative, không được gọi fully-unsupervised STDP.

Nguồn đối chiếu: [DRTP](https://arxiv.org/abs/1909.01311),
[e-prop](https://www.nature.com/articles/s41467-020-17236-y),
[DECOLLE](https://arxiv.org/abs/1811.10766). Đây là tổ hợp thử nghiệm, không tái lập
trung thực toàn bộ phương pháp hoặc tuyên bố tính mới.

| Vòng | Trial phát triển | Gain order | Gain XOR | Gain strict-binding | Gate |
|---|---:|---:|---:|---:|---|
| [Local target-rate](../../runs/20261109T092029Z-p1-local-target-v1-d4efd9/report.md) |54|−1,43pp|+0,91pp|+3,78pp|Fail|
| [Random feedback](../../runs/20261109T092905Z-p1-random-feedback-v2-68c153/report.md) |54|+1,43pp|−1,56pp|+1,95pp|Fail|

Mỗi vòng3fresh model seeds,2048training examples/task, matched same data/updates
với readout-only/scalar/shuffled. Gain là refitted-probe so initial hidden.
Không mở confirmation vì development không qua; không nới gate hoặc thử vòng3.
**NO-GO cho nhóm ứng viên này ở cấu hình đã thử**, không NO-GO cho mọi local SNN.
Order initial probe93,75% ởv1gần trần, hạn chế độ nhạy. Control learnability mới
bằng BPTT chưa chạy; negative conclusions phải giữ giới hạn đó.

## 4. P2: Một cải thiện đọc tín hiệu đã xác nhận, retention vẫn fail

Probe trên initial hidden khá tốt nhưng online head còn yếu. [Screen3seed](../../runs/20261109T093801Z-p2-readout-conditioning-v1-c230d5/report.md)
so raw/centered/normalized, chỉ normalization hữu ích. Chuẩn hóa presynaptic
feature bằng mean/std từng neuron, không thay hidden. Calibration train-only,
đóng băng trước feedback; output cập nhật local categorical delta, không ridge
runtime/inverse covariance/backward. Mỗi condition cùng1536labeled updates.

[Confirmation8seed mới](../../runs/20261109T094008Z-p2-conditioning-confirmation-v1-09c3ef/report.md):

| Bài | Accuracy chuẩn hóa | Gain so raw [CI95%] | Holm p | Gate conditioning |
|---|---:|---:|---:|---|
| Order |95,53%|+35,69pp [28,93;42,07]|0,02344|Pass|
| XOR |86,99%|+35,25pp [27,83;41,67]|0,02344|Pass|
| Strict-binding |61,74%|+7,96pp [4,71;11,52]|0,02344|Fail|

Đây là lợi ích **readout conditioning**, không hidden representation mới. Không
đánh tráo hai gate. Chưa là lời nói/ảnh bình thường hay lợi thế so Qwen4B.

Trên20context ở cùng8seed,128outputupdates/stage, calibration chỉ context0:

| Cơ chế | Acquisition | Final | Mean forgetting | TB worst/context mỗi seed |
|---|---:|---:|---:|---:|
| Raw |41,52%|31,70%|21,02pp|50,49pp|
| Normalized |62,10%|29,02%|35,14pp|80,57pp|
| Raw + replay |32,94%|31,80%|15,06pp|45,02pp|
| Normalized + replay |54,06%|31,04%|27,94pp|70,21pp|

**Mọi condition fail acquisition/final/forgetting gate**, kiểm trên từng seed.
Normalization giúp tiếp thu nhanh nhưng ghi đè mạnh hơn; replay chưa cứu được.
Chỉ equal-update, chưa equal-time; không công bố hoàn tấtP2 hoặc triển khai.
Field decision trong raw receipt ghi `RETENTION_PENDING`; retention thực đã đo
và **thất bại** theo bảng, không phải chưa chạy. Giữ raw receipt, ghi rõ ở đây.
Đây là task-boundary-aware stress, chưa chứng minh lifelong không task boundary.

## 5. Dữ liệu Antigravity mới và kiểm soát nhãn

Lượt high-model ban đầu timeout sau398,9s, process kết thúc; log giữ riêng.
Bốn lô Flash-low tạo40dialogue/120response đúng language split; audit phát hiện
228targettokens và nhãn assistant sai/giả thực hiện tác vụ. Không nạp corpus đó.
Teacher sửa lại nhưng vẫn204tokens; dữ liệu gốc giữ nguyên, không nới trần200.

Phiên [curated](../../data/candidates/antigravity-events-curated-20261109T094142Z/validation.json)
giữ40dialogue,120response,194targettokens; train20vi/12en,dev4vi/4en. Có14sửa
editorial công khai trong manifest, không giả tất cả do teacher tự đúng. Đã bỏ
nhãn nhận sai “tam giác4cạnh”, màu trộn không rõ miền, tự nhận đã gửi/lưu/mở file,
và khẳng định tri giác âm thanh không có nguồn. Giữ toàn bộ quá trình sửa.

Tất cả vẫn synthetic/rubric, không độc lập factual verification. Tại thời điểm
audit này chưa train; sau đó [trial14](14_EVENT_DIALOGUE_AND_QWEN_REFERENCE.md)
đã dùng trong prototype cô lập, không live auto-import. Exact-template IDs
không bảo đảm không có paraphrase gần;
exact-match cũng không bao quát mọi câu đáp đúng. Không flatten mất dialogue
history khi huấn luyện mục cần nhớ sở thích. Chưa có final test mới cho normal chat.

## 6. Bước tiếp theo theo master

P1: đưa ordered/RGB events vào một đường học thử nghiệm có word decoder, đo
temporal/binding và image/question ablation, không chọn bằng corpusđãxem.
P2: thử normalization với bảo vệ readout/importance có nguồn gốc và kiểm
acquisition–retention đồng thời; không tăng replay/data để che thất bại. Cần
confirmation matched-time trước tuyên bố hiệu quả.

Chỉ sau chất lượng/retention mới tích hợp vào cùng model đang phục vụ, migration
checkpoint và regression text/digits/feedback/rollback. Giữ Fast/Deep/Idle cùng
core, CPU-first2GB; chưa ACT/halting/tự khám phá có verifier hiệu quả. Mục tiêu
cuối giữ nguyên, **PIVOT cơ chế, chưa tăng quy mô**.
