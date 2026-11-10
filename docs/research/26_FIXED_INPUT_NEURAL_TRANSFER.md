# Biểu diễn đầu vào cố định cho thử học neural cuối

10/11/2026. **Đã chạy đủ27trials; STOP_WORD_SCORE_INPUT_TRANSFER_V2.**
[Protocol](../../experiments/g1-word-score-input-transfer-v2.md).

Kiểm tra nút thắt đầu vào bằng sinh từ thực từ cùng một coreALIF, không sửa
gốc thành Transformer. Giữ hai thất bại trước: word-score chưa hơn frozenhidden;
inputcontrol128wordIDs chưa qua ngưỡng. Không mở thêm numericintenthead hoặc
chọn template trả lời. Metadata từ trainprompt, không từ target/dev/prior.

Một từ đã có trong train được mã hóa thành128số cố định từ hashbits. Tất cả
từ dùng chung128channels, không thêm neuron, trainableembedding hoặc synapse.
So legacy và signedbucket cùng từ/IDF để phân biệt geometry khỏi coverage.
Đây là toán tiền xử lý master cho phép, không là representation đã học.
Chỉ công nhận lợi ích hiddenlearning khi ablation qua ngưỡng trên câu model
tự sinh; ΔW hoặc teacherforcinggain không thay được phép kiểm này.

27trials=3freshmodelseeds×3encoders×frozen/full/shuffled, cùng1024responses và
quy tắc đã khóa. Densecandidate chỉ định trước, không chọn winner theo kết quả.
CPU1thread×2workers, dự kiến10–15phút, giới hạn2400s và2GB. Dùng dataAGY và
Qwen đối chứng đã chạy thật; không nạp candidate vào phiênHermes.

Thiết kế trước đo quy định đây là vòng ứng dụng word-score cuối sauv1: nếu
không qua ngưỡng thì đóng hướng, không đổi tên để tune encoder/LR/epochs hoặc
tăng model. Kết quả dưới đây đã kích hoạt quyết định đóng. Mục tiêu đầy đủ
normalchat/vision/lifelongretention vẫn chưa hoàn thành.

## Kết quả thực, giữ nguyên điều kiện trước đo

[Run](../../runs/20261109T184028Z-g1-word-score-input-transfer-v2-c7a277/report.md).
Ba freshmodelseeds28001/28019/28027; không chọn encoder theo kết quả. Mọi giá
trị dưới là mean3seeds, exact tới một styleanswer đã công bố, không điểm hiểu
ngôn ngữ mở. Dev/prior đã từng xem nên không được gọi freshfinaltest.

| Mã đầu vào | Hidden rule | Train16 | Dev31 tự sinh | Prior48 tự sinh |
|---|---|---:|---:|---:|
| Legacy | Frozen |16,67%|3,23%|6,25%|
| Legacy | Full |10,42%|6,45%|4,17%|
| Legacy | Shuffled |6,25%|2,15%|2,08%|
| Signedbucket cùng299từ | Frozen |22,92%|4,30%|13,19%|
| Signedbucket cùng299từ | Full |31,25%|6,45%|9,03%|
| Signedbucket cùng299từ | Shuffled |6,25%|3,23%|3,47%|
| Densebits cùng299từ | Frozen |25,00%|8,60%|13,89%|
| Densebits cùng299từ | Full |12,50%|7,53%|9,72%|
| Densebits cùng299từ | Shuffled |8,33%|5,38%|7,64%|

Densefull so legacyfull devgain+1,08đ, CI95%−3,23..9,68. Densefull so
densefrozen−1,08đ, CI95%−3,23..3,23; priorgain−4,17đ, CI95%−6,25..0.
Ba seedbootstrap chỉ thăm dò, không chứng minh tác hại/lợi ích phổ quát.
Devteacher-tokenaccuracy densefull52,98% cũng không hơn densefrozen53,83%.
Không qua absolute40% hoặc bất kỳ gain10đ đã khóa nào.

**Đóng hướng word-score application sau hai vòng.** Không tune tiếp encoder,
vocabulary/LR/epochs/beam/hiddenrule hoặc scale để mở lại. Không chọn frozen/
sparse winner và không nhập vào Hermes. Chưa chứng minh hiddenrepresentation
ngôn ngữ hữu ích hoặc lifelongretention; kết quả P1order/XOR cũ vẫn giữ đúng
phạm vi hẹp của nó, không bị xóa hoặc suy rộng từ failure này.

## Tính toàn vẹn và tài nguyên

[Audit](../../runs/20261109T185653Z-g1-word-score-input-audit-v1-b8c38e/report.md):
27/27coldprocesscontinuations khớp loss/nextreply/fullbyteshash. Frozenhidden
khớp initialweights, full/shuffled thay trọng số thật. Codebook tái tạo đúng từ
trainprompt không nhãn; các file consumedcore/protocol/data khớp snapshot.
Servingcore.py/unified.py giữ SHA cũ. Mỗi trial cùng206.514byte neuralarrays,
metadata thêm hữu hạn; không thêm learnedembedding/neuron/synapse.

[Usage receipt](../../artifacts/g1-word-input-data-usage.json) tái tạo đúng
samplerstream và khớp word-updatecounter của từng checkpoint:27.648response
updates gồm17.811lượt mẫuAGY và9.837mẫuauthored (lặp nhiều lần, không phải chừng
đó kiến thức độc lập). Không dùng dev/prior hoặc privatefeedback để học,
không gọi thêm teacher, không coi corpusđượcxácminhstyle là mọi dữ kiệnđúng.

Lượt chính972,37giây≈16,2phút (dài hơn dự toán10–15phút, vẫn dưới trần2400s),
peakprocess-treeRSS163,45MB/commit118,40MB. Tree được giám sát; Windows cap
theo process báo riêng, không gọi sampledRSS là hardcap toàn cây. Không iGPU,
swap/offloadweights, đo điện năng hoặc teacherAPI mới.152tests qua gồm native
baselineparity. Testpass chỉ xác nhận phần mềm/persistence, không quality.

## Quyết định nghiên cứu tiếp theo

SCAR-X tổng thể tiếp tục **PIVOT**, không NO-GO cho mọi mạng phát xung. Cần tách
khả năng biểu diễn/đọc đầu ra của ALIF hiện tại khỏi độ hiệu quả creditassignment
bằng đối chứng learnability cô lập, cùng forward/input/head/data/budget. Nếu
dùng surrogateBPTT thì chỉ trong `evaluation/controls.py`, tuyệt đối không nhập
vào maincore/Hermes; phải preregister/parity và công bố chênh compute/objective.
Đây là bước đề xuất, chưa chạy, không là quy tắc hidden mới hoặc Transformer.

## Phép can thiệp đầu vào độc lập trên checkpointv1

[Protocol](../../experiments/g1-word-conditioning-audit-v1.md),
[run](../../runs/20261109T184755Z-g1-word-conditioning-audit-v1-5a2bef/report.md).
Đã đo cả9checkpointv1, không train hoặc sửa state/weights; chỉ dự đoán từ đầu,
không xem đây là câu trả lời đầy đủ. Cùng input còn BOS/control/timing và MC8,
chỉ bỏ/tráo128contextchannels. BatchedRNG khác decoder cũ, không chấm lạiv1.

Fullhidden meantrainfirsttoken66,67% với câu hỏi thật,35,42% khi xóa context.
Nhưng trêndev, thật34,41%, xóa37,63%, tráo sang familykhác31,18%. Xóa context
đổi argmax ở38,71%devcases và featureRMS≈0,300: mạng có phản ứng với câu hỏi,
không thể kết luận nó hoàn toàn bỏ qua đầu vào. Tuy nhiên lợi ích trên train
không chuyển rõ sangdev; tăng input sensitivity không tương đương hiểu câu hỏi.
Nhiều đáp án cùng từ đầu nên accuracy này không tách được mọi quan hệ ngữ nghĩa.

Kết quả củng cố ưu tiên conditionalrepresentation/generalization, không phải
chỉ chạy thêm decoder hoặc replay. Audit không chọn encoder, sửa schedule/gate
của v2 hoặc mở lại family đã dừng. Các số trên không là normalchat/P2retention.
