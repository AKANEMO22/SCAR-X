# G1: biểu diễn ý câu hỏi và bộ sinh câu

Ngày10/11/2026. PhầnA/B đã đo; search chưa qua gate và đã dừng.
Không thay model live, không gọi một head chọn8template là modelngônngữ.

## A. Cả representation và decoder đều yếu

[Protocol](../../experiments/g1-representation-decoder-audit-v1.md),
[run](../../runs/20261109T172637Z-g1-representation-decoder-audit-v1-e98975/report.md).
Giữ nguyên3privateclones đã khóa, cùnghiddenweights, khác curriculumorder/head.
63casecũ+117AGYtrain,31dev,48prior diagnostic; không train neural core mới.
Target chỉ dùng ở phần teacherforcing/probe được ghi rõ, không vào prompt-only.

| Split | Input8classridge | Prompt-hiddenridge | Firsttoken | Teacherforcedtoken | Free styleexact |
|---|---:|---:|---:|---:|---:|
| Train |100,00%|89,44%|65,00%|80,60%|38,15%|
| Dev |41,94%|35,48%|34,41%|59,24%|11,83%|
| PriorAGY48 |72,92%|45,83%|44,44%|65,38%|13,89%|

Khớp inputtraining100% không là hiểu ngôn ngữ: devchỉ42%. Hidden cũng mất một
phần khả năng giải mã từ input trên các caseprior. Sinh tự do thấp hơn token
teacherforced, nên decoder/cross-tokenconditioning cũng là vấn đề. Những số
này không tách nguyên nhân duy nhất; OOVinput/featureconditioning/representation
và sai số tích lũy có thể đồng thời góp phần.

Featuremean sau teacherforcingtoàncâu chứa goldpreviouswords và giải mã intent
dev58,06%/prior79,17%, nhưng **đây là control có labelleak cố ý**, không là
representation inference đã học. Idlefeaturecontrol khoảng12–13%, phù hợp
chance8classes. Các diagnostic8classheads không được dùng chọn template để
trả lời; raw privateweights/vectors/outputs chỉ ởlocal.

## B. Decoder search có giới hạn: chưa đủ cải thiện

[Protocol](../../experiments/g1-bounded-beam-diagnostic-v1.md),
[run](../../runs/20261109T173540Z-g1-bounded-beam-diagnostic-v1-f972ce/report.md).
Ba checkpoint đã khóa, width1/4/8 và alpha0/0,6 đăng ký trước. Width1 khớp
greedy cả text/state. Devnormalizedexact12,90%→21,51% với width4/alpha0,6;
width8 không thêm gain. Prior48 tăng4,17điểm. Chưa đạt ngưỡng25%dev/gain10đ
dev/gain10đprior, nên **STOP_G1_WORD_DECODER_SEARCH**; không tăng width/tune
penalty hoặc đổi modellive. Khoảng7giây, peakRSS54,2MB cho lượt chẩn đoán.

Theo master: tokenizer/readout được dùng toán số thông thường; workingstate
phải hữu hạn và CPU/RAM có giới hạn. Thử beamsearch tổng quát trên xác suất
token của chính core hiện có; không vocabulary câu trả lời/intentlabel/target
hoặc Qwen đi vào search. Chỉ có một object trọng số shared; các hypothesis giữ
workingstatearrays, không thêm modelID hoặc mạngneuralthứhai.

Giữ virtualneuronclock/trạng thái theo từng hypothesis; computationalticks
tổng đo riêng. Đây là simulation/search trên diagnosticclone, **chưa phù hợp
trace monotonicclock và statepolicy runtime** nên không triển khai vào Hermes
từ kết quả này. Không gọi search là synapse học representation hoặc suy nghĩ
sâu/neurobiology đã được chứng minh. Mainloop luôn tiếp tục dùng decoder cũ.

Nguồn length-normalization: [GNMT, Wu và cộng sự2016](https://arxiv.org/abs/1609.08144).
Chỉ mượn công thức lengthpenalty; không attention/coverage/pretrainedweights,
không tái lập chất lượng công trình nguồn. Thêm compute không bảo đảm đúng hơn.

Protocol đã khóa width/penalty/budgets trước đo; actualticks/statebytes và
weights không đổi được ghi trongrun. Exactstyle và runtime-normalizedstyle
được chấm riêng. Giữ failure này; không mở thêm decodersearchgrid.

## C. Existing state channels cũng chưa giải được representation P2

[Run](../../runs/20261109T173923Z-p2-neuron-state-observability-v1-73a8f0/report.md)
đọc sáu checkpoint offlinejoint có sẵn, năm viewH48, không traininghidden mới.
Trained-adaptation localhead39,73% so filteredspikes38,66%; ridge tốt nhất
khoảng50,05%. Không view nào qua screen: **STOP_STATE_OBSERVABILITY_SCREEN_V1**.
Không lấy voltage/adaptation dễ đọc hơn đôi chút làm bằng chứng kỹ năng mới;
không mở thêm viewgrid hoặc tự sửa topology.
