# Event→neuron→word: thử hội thoại thực và Qwen đối chứng

2026-11-09. Theo masterP1/P2/product MVP, không tăng neuron/framework hoặc thay
live weights. Đây là kiểm tra đầu ra ngôn ngữ thật sau audit input13; không dùng
điểm phân loại/order để tuyên bố model nói chuyện bình thường.

## Corpus đã dùng và phạm vi

[Antigravity curated40dialogue](../../data/candidates/antigravity-events-curated-20261109T094142Z/validation.json):
32train/8development, mỗi3cặp lượt,96train responses/24development.20vi/12en ởtrain,
4vi/4en ởdevelopment. Có review/editorial14sửa,194targettypes toàncorpus; train
chỉ164types, cộngBOS/EOS thành166. Không thêm30development words vào vocabulary.
Chỉ9/24đáp án reference có thể biểu diễn hoàn toàn bằng train-vocabulary.

Codec đóng này không thể phát ra từ mới chưa nằm trong output dictionary, dù
thay weights hoặc tăng ticks. Đó là giới hạn biểu diễn, khác generalization.
Tuy nhiên ngay cả câu reference biểu diễn được cũng chưa được trả lời ổn định.
Exact-match chỉ là một số đo: câu khác vẫn có thể đúng, nên xem thêm outputthật.
Mọi nhãn synthetic/rubric, không được gắn independently_verified hoặc đưa vào
phiên riêng tư. Raw batches/repair/sửaeditorial đều giữ nguyên nguồn/hash.

## Prototype thực thi

`evaluation/event_dialogue.py` nối chronological token events hoặc RGBpatches
vào **một UnifiedCore48ALIF**, rồi dùng word readout chung phát câu. Không
template/truy đáp án/pretrained LLM. V+128+64+6input, word+digit output slices;
không thêm hidden layer. Đường RGB→word có thể chạy, chưa có photoQAtraining
trong trial này; không phải vision đã học. Không sửa server/ID hiện hành.

[Protocol](../../experiments/event-dialogue-prototype-v1.md):3freshseeds5101/5113/5119,
4conditions,8bootstrap passes qua96responses, giữ neural state xuyên lượt/pass.
Readout calibration train-only; chronological và pooled control có cùng số
events/input history. Full chỉ là scalar-STDP rule hiện có, không dùng nhóm
local-target vừa bị NO-GO. Development/OOV không tham gia learning hoặc vocab.

| Condition | Dev exact sau8pass | Teacher-forced token accuracy | Train exact | Dev repetition |
|---|---:|---:|---:|---:|
| Bag/raw |0%|21,03%|1,04%|12,50%|
| Ordered/raw |0%|22,46%|0,35%|20,83%|
| Ordered/normalized |0%|19,79%|2,08%|13,89%|
| Ordered/normalized/full |0%|20,50%|2,08%|27,78%|

[Run đầy đủ](../../runs/20261109T094940Z-event-dialogue-prototype-v1-285565/report.md)
có mọi outputs/scores/checkpoints. Dừng theo protocol, không kéo dài pass hoặc
chọn seed để che failure. Teacher forcing/tokenaccuracy không là cuộc hội thoại
tự sinh thành công; normalization ở classification không chuyển thẳng sang
autoregressive generation. Hiddenfull chưa có lợi ích thấy được trong trial.

[Cold-process audit](../../runs/20261109T095432Z-event-dialogue-cold-audit-03d908/report.md)
12checkpoints khớp mọi devoutputs/NLL khi mở process mới. Đây là persistence,
không phải quality. Calibration statistics/readout/core state đều được phục hồi.

## Qwen chạy thật trên cùng24prefix

[Protocol đối chứng](../../experiments/qwen-dialogue-reference-v1.md),
[run Qwen](../../runs/20261109T095436Z-qwen-dialogue-reference-v1-1598dd/report.md).
Dùng đúng pinnedQwen3-4B-Instruct-2507 Q4_K_M đã kiểmhash; raw role/text prefixes,
không thêm system/tool/RAG. CPU4threads,context1024,max64tokens,temperature0.
Qwen không chạy đồng thời với trainer; process đã tắt sau24đầu ra. Không đổi
Hermes/provider hoặc dùng Qwen làm proxy. PeakRSS khoảng3,070GB/commit3,753GB,
không cùng budget2GB của SCAR-X; khác pretraining và độ dài outputtokenization.

Ví dụ từ một seed5101/ordered-normalized để nhìn lỗi, **không chọn winner**:

| User cuối prefix | SCAR-X prototype | Qwen reference, tóm tắt |
|---|---|---|
| Xin chào, rất vui được gặp bạn hôm nay. | `Thank` | Lời chào Việt có nghĩa và gợi ý trò chuyện |
| Bạn có khỏe không? | `Tôi ổn tắt trợ bạn.` | Giải thích là AI, không có sức khỏe cơ thể, mời hỗ trợ |
| Can you explain the next step? | `I nhận nhận nhận nhận nhận nhận nhận` | Hỏi rõ quy trình/bước đang nói đến |
| The final review step in the list. | `I nhận i am tin please nhận nhận` | Nhận chưa có danh sách, yêu cầu nguồn |
| A triangle has four sides right? | `Xin ghi không phần` | Bác bỏ4cạnh, nói3cạnh, nhưng câu bị length-truncate |

Qwen có11/24outputs bị cắt do64-tokenbudget; không thaybudget sau kết quả để
làm đẹp. Câu hỏi thời tiết khiến nó tự nêu “tháng4/2024” dù không được cung cấp
mốc đó; một câu khác hứa hỗ trợ tra cứu chính xác khi chưa có tool. Vì vậy
Qwen không là oracle/nguồn nhãn đúng. Phép xem này là exploratory/source-known,
không phải rubric mù hoặc đánh giá con người độc lập. Không gán phần bị cắt là
câu đầy đủ; không suy ra4B-equivalence hoặc chất lượng ngoài24prefix.

## Kết luận và bước kế tiếp theo master

**Chưa đạt mục tiêu nói chuyện/nhìn bình thường.** Ordered encoder đã sửa mất
thông tin đo được nhưng chưa tự tạo language representation. Readoutconditioning
đã giúp classification, vẫn failcontinual retention và không giúp trial sinhcâu.
Không triển khai candidate kém hơn, tăng model/data hoặc nhậpLLMweights.

Các vấn đề còn mở để P1/P2 giải quyết:

- Tokenizer/decoder phải biểu diễn được từ chưa có sẵn, ví dụ byte/character
  codec được thiết kế/kiểm riêng; không mở vocabulary từdev để giảgeneralization.
- Credit phải nối chronological input với nhiều bước sinhcâu. Việc chỉ giữ đủ
  input không chứng minh synapse đã học dependency; cần hidden/refit/lesion tests.
- Head/synapse protection phải giữ acquisition mới và oldskill cùng lúc, theo
 3pp/5ppgates hiện có; normalization chưa làm được.
- VisionRGBwordpath cần dữ liệu ảnh có nguồn và image-grounded generalization,
  gồm shuffle/blank-image ablation. Hai ảnh sample hoặc outputđigit không đủ.

Mục tiêu đầy đủ giữ nguyên. Runtrials là evidence để sửa cơ chế, không là một
phiên bản “AI bình thường” được đổi tên cho vừa kết quả.
