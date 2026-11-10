# Chẩn đoán đường thông tin ảnh–câu hỏi

2026-11-10. **Đã hoàn tất; VISUAL_INFORMATION_DIAGNOSTIC_ONLY. Chưa có cải thiện model.**
[Protocol](../../experiments/visual-information-audit-v1.md).

Sau vòng33, ngay đối chứng global cũng chưa fittrain. Vòng này chỉ đo các
checkpoint32/33 đã có; không learner/encoder/grid mới, không sửa synapse,
không Transformer/BPTTmain hoặc mở đềxuất30. Main/privateweights giữ nguyên.

Đo exactcollisions và các cặp cùngcâuhỏi/khácảnh hoặc cùngảnh/kháccâuhỏi dọc
input→state→visualtrace→BOSfeature→normalization→readout. Một probe ridge cốđịnh
ở đúng input của readout, reset classifier, dùng đúng multiset1536nhãn train
đã thấy. Probe firstword so cùngfirstwordhead; semanticjointprobe chỉ phân loại,
không câu trả lời. Các rawinputreferences giúp kiểm thông tin trước neuralcore.
Không huấn luyện probe trên mọi stateview hoặc chọn representation theo điểm.

Mọi phépfit là diagnosticclassifier ngoài core, không học synapse mới và không
đưa output/nhãnclass thay câu neural trong Hermes. Phải giữ actualanswers,
nhãnPROBE, nguồn/thời điểm; localhost tựtheodõi. Dev đã xem là exploratory,
không test cuối, normalvision hoặc chứng minh học cục bộ/lifelong.

## Kết quả và điều được phép kết luận

[Run](../../runs/20261110T000339Z-visual-information-audit-v1-882afb/report.md)
đo10368lần prepare→BOS trên6checkpoint hiện có,373248tick forward. Không cập
nhật neuralweights hoặc học lại. Các câu đầy đủ trong archive vẫn là output
nguồn33 đã lưu, có thời điểm cũ; audit mới đo từ đầu và classifier chẩn đoán.

| Mean3pairedseeds, dev | Hiddenfrozen32 | BPTTcontrol33 |
|---|---:|---:|
| Từ đầu thực của wordhead |28,73%|35,48%|
| Probe từ đầu, cùng48feature |12,69%|16,06%|
| Probe semanticjoint20class |8,30%|11,60%|
| Probe semantic nhãn tráo |2,76%|7,62%|
| Số kênh chạm clip±5 |0,157%|1,079%|

**Không so semanticclassification với accuracy cả câu.** Semanticjoint yêu
cầu kind/language/attribute, không sinh từ; probe không thay câu trả lời neural.
Probe từ đầu và wordhead có cùng đầu vào/target/outputmask; ở chính phép so này
probe kém16,04điểm (frozen, CI95%−18,06..−14,13) và19,42điểm (control,
CI95%−23,16..−12,22). Flag BOS_READOUT_TRAINING_GAP không đạt. Kết quả không
chứng minh readout hiện tại tối ưu; chỉ bác bỏ lợi ích của đúng fixedridge này.

Semanticprobe control−frozen+3,30điểm, CI95%−4,19..8,14, không qua diagnostic
gate5điểm/lowerCI>0. Đây vẫn là GLOBALcontrol, không phải localhiddenlearning.
Probe semantic fit khoảng79–84% trên các trainexposures, nhưng chỉ8–12%dev:
feature có thể hỗ trợ fit đã thấy, chưa biểu diễn quy luật tổng quát đủ tốt.

RawRGB768+question128 có semanticprobe mean50,22%dev (48,76–52,26% theo seed),
shuffled4,51–9,55%; question128→kind/language đạt75%macrodev. Rawreference có
896chiều thay vì48, không cùng budget và không chứng minh encoder đã đủ cho
normalvision. Kết quả gợi ý thông tin sẵn có ở upstream dễ đọc hơn bằng phép
đo cốđịnh này; chưa tách nguyên nhân nén, động lực recurrent và generalization.
Rawfit gần100%train cũng không là thành công của neuralcore.

## Không nhầm vector khác nhau với representation có ích

Với tất cả6sources, inputtape, rawRGB+question, preparedstate, visualtrace,
BOSfeature, actualnormalizedfeature và logits không có contradictorylabel
collisions trên tập dev đã đo: empiricaloracleceilings≈100%. Các cặp khácảnh/
kháccâuhỏi không bị đồng nhất byte tại những vị trí đó. Không có cặp upstream
khác nhau thành postclip giống nhau; trainheadfeature rank43–47/48. Điều này
không chứng minh hiểu ảnh: một mã ngẫu nhiên cũng có thể cho mỗi ví dụ một
vector riêng mà không tổng quát hóa sang ví dụ mới.

Flag observed_component_information_collapse=True xuất phát từ **question-only**
branch: thiếuRGB theo thiết kế, nên oracle semanticceiling46,875% và100%imagepairs
trùng ở riêng nhánh này. Đây là sanitycheck đã biết, không là phát hiện core
làm mất ảnh hoặc lý do sửa neuron. Kết luận phải đọc từngstage, không lấy flag
True ra khỏi ngữ cảnh. Không điều chỉnh flag/threshold sau khi thấy kết quả.

Clipping chỉ0,157%/1,079% trung bình và không gây exactconflictingpair collapse
trong phép đo này. Chưa có bằng chứng đủ để mở lại normalization/cache/stateview
grid đã STOP. Cũng không khẳng định mọi giới hạn biểu diễn đều do readout.

## Integrity và phạm vi nghiên cứu

Mọi probe fit đúng multiset1536trainexposures/seed, không thêm các traincase
chưa được thấy; calibration/statistics không dùngdev. Labelshuffle giữ histogram
nhãn/exposure, không tráo ảnh hay thay neuralinput. Không Torch, backward hoặc
optimizer hidden; weights/stable/topology/calibration/counters/RNG của bản nguồn
giữ nguyên. Workingstate của các bản sao thay đổi do forward là dự kiến.

47,60s sampledmonitor, peakRSS133,39MB/commit150,06MB, Windowscap2GB. 176tests
qua trướcaudit, gồm nativeparity; thiếu RNG ở deterministiccore trong test đã
được sửa ở diagnosticguard trước phép đo, không đổi thuật toán hay kết quả.
MainALIF/privateweights/proposal30 giữ nguyên. Dataset là192ảnh hình học sởhữu,
32wordings từAntigravityCLI trước đó; không gọi teacher mới hoặc dùng nhãnQwen.
Qwen4Btext-only vẫn không có visioncomparison. Dev đã xem,3pairedseed và fixed
linearprobe không là freshfinal/generalphoto/lifelong evidence.

## Quyết định

**PIVOT_NO_SCALE, representation/generalization còn chưa giải quyết.** Không
chọn probe làm policy, không sửa normalization vì một giả định chưa đo được,
không thêm cache/decoder/encoder-grid hoặc modelscale để mở lại family32/33.
Các quan sát hiện tại chưa cung cấp learner cục bộ nào nói/nhìn bình thường.

Bước nghiên cứu learner mới cần ranh giới tín hiệu được chốt và protocol riêng.
Đềxuất30 vẫn là phương án chưa duyệt, không phải lời giải đã chứng minh: ngay
globalcontrol33 còn chưa fittrain, nên mở tín hiệu một bước cũng không bảo đảm
thành công. Không tự thay gốc ALIF hoặc đưa BPTT vào main để lấp khoảng trống.
