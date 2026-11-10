# Đối chứng learnability ảnh–câu hỏi–từ

2026-11-10. **Đã hoàn tất 3 đối chứng; CONTROL_LEARNABILITY_NOT_DEMONSTRATED_IN_BUDGET.**
[Protocol](../../experiments/visual-word-learnability-control-v1.md).

Vòng32 đãSTOP. Đối chứng riêng này phân biệt hạn chế representation/forward
với learnerreadout; không mở lại v1 bằng tuning. Cùng48ALIF/input/topology/
visualtrace/head/calibration và3pairedseeds/1536experiences. Chỉ trong
`evaluation/controls.py`, surrogateBPTT global/temporal học hidden; head vẫn
output-local. NumPy forward chuẩn, parity/codec/resume/timing trướcformal.
Compute graph thêm phải được đo, không coi cùngexperiences là cùngcompute.

Đây không là localrule/main/proposal30. Hermes/privateweights khôngđổi;
khôngTransformer/LLMproxy/scale. AGY32câu hỏi thật táisửdụng16/16wordings,
rendererlabels ngoàiinference. Qwen4Btext-only không cóvisionreference.
Gate dev/novel/perkind/gain/causal đăngký trước; fittrain riêng chưađủ.
Localhost tựhiện actualCONTROLanswers/learning/neuron, gồm cả864dev vàcâu sai.
Nếu thấtbại, dừng đốichứng; không đổi encoder/LR/epochs/slope/cache để tăng điểm.

## Kết quả thực

[Run](../../runs/20261109T234349Z-visual-word-learnability-control-v1-e51fe9/report.md),
[preflight](../../runs/20261109T233806Z-visual-word-control-preflight-v1-805213/report.md).
Ba seed ghép cặp33113/33119/33133, mỗi seed1536trải nghiệm. Đánh giá toàn bộ
864train và864dev bằng tự sinh từ, không chỉ kiểm một mẫu recall dễ.

| Seed | Macro train | Macro dev | Tổ hợp mới dev |
|---|---:|---:|---:|
|33113|53,78%|28,06%|30,73%|
|33119|42,93%|27,82%|32,12%|
|33133|48,59%|21,14%|19,48%|
|Trung bình|48,43%|25,67%|27,44%|

Baseline hidden đóng băng đạt18,88%dev. Gain ghép cặp+6,79điểm, CI95%
0,30..10,76; chỉ3seed và bộ dev đã xem nên là kết quả thăm dò. Không qua
ngưỡng gain20điểm/dev65%/novel60%/mỗi kind50%. Accuracy hình dạng chỉ5,21%,
màu đơn35,76%, đếm15,54%, màu bên trái46,18%. Chưa fit toàn bộ train ở ngưỡng65%.

Xóa ảnh giảm13,95điểm, xóa câu hỏi20,72điểm, tráo ảnh6,40điểm; chỉ tác động
xóa câu hỏi qua mức20điểm trung bình. Zero-input là can thiệp khác phân phối,
không tự chứng minh representation hữu ích. Mọi gate tổng hợp đã đăngký đều fail.

Hidden trong CONTROL đã đổi thật: normΔW_in lần lượt5,264/5,097/4,728 và
ΔW_rec1,085/0,956/0,910. Điều này không phải bằng chứng hidden của SCAR-X
chính đã học kỹ năng, hoặc quy tắc local đã thành công. Cùng forward/head
nhưng tín hiệu hidden, Adam/allrows và compute khác; chưa tách riêng từng yếu tố.

## Tài nguyên, tái lập và phạm vi

4608formalexperiences,10534output-localwordupdates,4608hiddenoptimizersteps/
backwardpasses. NumPytrainingticks183666 và thêm183666graphforwardticks;
không giả cùng compute với baseline đóng băng. Sampler/wordupdates/ticks
khớp baseline32 theo từng seed. Timing64response và các lần học trên bản sao
để kiểm continuation được báo riêng, không tính thành kiến thức mới hoặc
lượt formal. Không dùng thêm nhãn Qwen/teacher, không gọi AGY lại trong vòng này.

Formalround97,82s; process-tree peakRSS639,85MB/commit1025,50MB, từng process
có Windowscap2GB; không iGPU hoặc suy luận điện năng từ FLOPs. 3/3freshprocess
nextlearn→freeanswer→sampler→fullstatehash khớp. Max graphstateerror7,63e−6,
featureerror5,96e−8, losserror3,75e−6. 173softwaretests qua, gồm nativeparity;
đó là kiểm tra kỹ thuật, không chứng nhận chất lượng.

Localhost tự hiện đầy đủ864devoutputs/96ownedimages của tệp hoàn tất gần nhất,
có cả câu sai, nguồn/thời điểm và nhãn **ĐỐI CHỨNG ALIF-BPTT — KHÔNG MAIN**.
Learning feed có loss, ΔW_in/rec/out và số synapse đổi thật; không có guard
main trong control. GUI đã quan sát cả lúc chạy và bảng kết quả/đầu ra cuối.

## Kết luận và hành động tiếp theo

Không đưa control vào Hermes, không mở lại32/33 bằng tuning, không đổi gốc
ALIF/locallearning hoặc tăng neuron/tham số/dữ liệu để che thất bại. Main
privateweights/topology/vocab/encoding và core/unified source giữ nguyên;
đề xuất30 vẫn chờ quyết định rõ, không suy ra approval từ continuation.

Tín hiệu global/temporal giúp tăng điểm hẹp nhưng chưa giải bài hiện tại.
Do cả train và dev còn thấp, chưa đủ cơ sở nói lỗi chỉ do localcredit. Các
nguyên nhân còn mở: thông tin ảnh/câu hỏi sau mã hóa, mất quan hệ không gian
khi gộp neuralfeatures, tính phân biệt của recurrentstate, head interference,
và ngân sách học cốđịnh. Chưa xác định nguyên nhân nào là nhân quả duy nhất;
cũng không chứng minh ALIF hoặc purelocallearning bất khả thi.

Ưu tiên chẩn đoán chỉđọc đường thông tin input→state→visualtrace→wordhead,
trên các checkpoint hiện có, tách mất thông tin khỏi readout/decoder. Đây là
nhánh representation/rootcause của master, cần protocol riêng trước phép đo;
không một learner/encoder/memorygrid mới được tự triển khai từ báo cáo này.
Normalchat/normalphoto/OCR/actions và lifelongretention vẫn chưa đạt; dự án
giữ **PIVOT_NO_SCALE**, chưa hoàn thành mục tiêu cuối.
