# Đề xuất: chuyển quy tắc paired-score đã khóa sang dự đoán từ

Ngày10/11/2026. **Pilot đã chạy và chưa đạt, chưa là model trong Hermes.** Phép đọc
state ởP2 không qua screen: localhead tốt nhất adaptation39,73%, spike38,66%,
ridge khoảng50%; không mở thêm view/timescalegrid. Kết quả decoder beam cũng
STOP. Cần thử learning representation, không chỉ sửa inference/ôn thêm.

Đề xuất giữ cùng ALIF48/K8/R12, wordinput/context/control layout củaG1 và
reservedimagechannels/readoutdigits (chưa trainvision trongpilot). Thay đúng
cơ chế hiddenlearning yếu của G1 bằng thuật toán paired-scorev2 đã khóa/
math-audit: scalarreturn × local forwardeligibility, per-synapsemoments, bốn
rotatingblocks×hai stochastictrajectories; không taskW_out feedback/backward.

Đây là **transfer của quy tắc đã chọn**, không vòng3tuning hoặc mở lại family
local-classifier/target-rate/byteworkingmemory đã dừng. Không đổiLR,temperature,
optimizer hoặc tìm thêm variant. Forwardconstants củaG1 khác bàiP1 nên kết
quả P1 cũ không chuyển thành bằng chứng language hoặc retention mới.

Mỗi targettoken học từ prefix trước đó: fixed prompt context và các token
trước do teacher cung cấp trongtraining; token đích hiện tại/future không vào
input. Counterfactualtrajectories reset canonicalstate để đúng estimator,
inference cũng dựng prefix từ token neuralđã sinh, không từ đáp án mẫu. Đây
không phải statepolicy never-reset của runtime; chỉ dùng trên prototype riêng.

Tokenizer/hash còn hạn chế và vocabulary chỉ gồm từ trong8styleresponses;
không đảm bảo giải được semantics với ítdata. Headmasks tách word/digit và
BOS/EOS nhưG1, không chọn template. Một core với một bộweights; adapter tạo
input số học không là model thứhai. Digitoutputs/imagechannels reserve không
được mô tả là vision đã học hoặc photoQA. MainmodelID/weights/history giữ nguyên.

Codec riêng chứa weights/state/eligibility/moments/rowclocks/calibration/
mask/RNG/vocabulary/decoderRNG. Không épformat mới vào UnifiedCore hoặc nhập
privateweights. Test target-noleak, maskedheadlocality, fullstatecontinuation
đã qua trước training; protocol khóa freshseeds/schedule/budget trước đo.

## Kết quả word-score transfer v1

[Protocol](../../experiments/g1-word-score-transfer-v1.md),
[run](../../runs/20261109T175033Z-g1-word-score-transfer-v1-1b1604/report.md).
Ba modelseed mới27103/27109/27127 × frozen/full/shuffled,1024responses/model,
64authored+117AGYtrain,31dev và48prior đã từng xem. Đây là exploratorydiagnostic,
không freshfinaltest hoặc thực nghiệm retention20context.

| Điều kiện | Train16 exact chuẩn hóa | Dev31 | Prior48 |
|---|---:|---:|---:|
| Frozenhidden |29,17%|3,23%|9,72%|
| Full localscore |20,83%|3,23%|4,17%|
| Shuffledreward |8,33%|2,15%|0,69%|

Full devteacher-tokenaccuracy52,24% so frozen45,43%, nhưng freeutterancegain0đ
(seedbootstrapCI95%−3,23..3,23). Không qua ngưỡng đã khóa: **STOP_WORD_SCORE_TRANSFER_V1**.
Không nhập candidate hoặc nới tiêu chí, tuneLR/epochs/hiddenrule hay scale.
Khoảng284giây, peakprocessgroupRSS155,1MB. Hiddenupdate và ΔW thật đã lên trace,
nhưng thay trọng số chưa tạo bằng chứng representation ngôn ngữ hữu ích.

## Input identity control: có nút thắt, chưa có giải pháp neural

[Run](../../runs/20261109T180124Z-g1-input-identity-control-v1-422c17/report.md)
so numericinputcontrol cùng128chiều; local8classhead là diagnostic, không dùng
chọn template trả lời. Devmean legacy53,76%, dictionary12861,29%, cùng từ qua
hash60,22%; dictionary299chiều73,12% là referencekhácbudget. Coverageworddev
tăng51,45%→69,86% khi giữ299từ. Ba curriculumorders không là ba modelseed mới.

Dictionary128 chưa đạt65%localdev dù ridge77,42%; **STOP_INPUT_IDENTITY_CONTROL_V1**.
Không suy ra đã học semantics hoặc giảm toàn bộ lỗi còn một nguyên nhân, không
mở vocabsize/hashseed/featuregrid từ control này. Bất kỳ giả thuyết mới nào phải
có protocol riêng và khác thực chất; các stoprule/gate P1/P2 giữ nguyên.
