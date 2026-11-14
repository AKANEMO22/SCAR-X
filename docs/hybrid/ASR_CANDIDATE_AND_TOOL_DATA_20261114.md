# ASR chính xác hơn và dữ liệu công cụ có căn cứ — 14/11/2026

## Kết quả thực tế và quyết định

**Whisper-small FP32 là ứng viên dùng VRAM có ích hơn đã đo được.** Trên bộ
32 mẫu speech DEV mới, nó giảm lỗi nhận dạng tiếng Việt rõ rệt so với base.
Nhưng hỏi đáp video của cùng Qwen0.8B vẫn sai; cải thiện cảm nhận không phải
bằng chứng SNN đã học kỹ năng hoặc toàn trợ lý thông minh hơn.

Không thay ASR live. Lượt endurance554c19 đang kiểm profile base đã khóa;
không sửa source/weights để đổi profile giữa chừng hoặc lấy kết quả8h của base
chứng nhận cho small. Main gốc, ALIF/local rules và dữ liệu live giữ nguyên.
230Python checks (gồm CPU/native/CUDA và các comparator/resource regressions)
và10Node checks đã qua. Warning CuPy toolkitpath và ZIPtrùngmember được ghi rõ
ở các lượt trước; testpass không được dùng thay quality/retention gates.

## So sánh speech trên cùng dữ liệu

Nguồn [openai/whisper-small chính thức](https://huggingface.co/openai/whisper-small),
revision973afd24965f72e36ca33b3055d56a652f456b4d; source966.995.080byte,
SHA2561d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b.
479 tensor FP32 gốc được repack và kiểm từng byte độc lập, không bỏ tensor,
quantization, pruning hay chuyển FP16. Nativeexport967.535.936byte. Model card,
metadata giấy phép và upstream LICENSE giữ trong local candidate; không commit
hoặc phân phối lại trọng số.

Dữ liệu [FLEURS của Google](https://huggingface.co/datasets/google/fleurs),
CC-BY4.0,16mẫuVi+16mẫuEn, lấy theo quy tắc đã khóa trước inference, transcript
của dataset. Loại mẫu engineering cũ; không chọn theo điểm model. Model Whisper
có thể đã thấy loại dữ liệu tương tự khi pretrain, nên đây không là chứng minh
generalization ngoài mọi dữ liệu huấn luyện của nó.

| Chỉ số | Whisper-base | Whisper-small |
|---|---:|---:|
| Vi: word edits /457referencewords |210|97|
| Vi: microWER |45,95%|21,23%|
| En: word edits /332referencewords |32|20|
| En: microWER |9,64%|6,02%|
| Median wall time mỗi cold CLI request |1,024s|1,680s|
| p95 real-time factor |0,178|0,321|

Vi giảm24,73điểm WER, tương đối53,81%; CI95% paired utterance của
small−base là[-31,30;-18,94]điểm. En giảm3,61điểm theo point estimate,
CI[-7,84;0,00], chưa có bằng chứng chắc rằng En tốt hơn. Gate En chỉ yêu cầu
không kém quá2điểm ở pointestimate; không đổi tiêu chí sau kết quả.
WER là tỷ lệ edit trên số từ, không phải accuracy hoặc điểm intelligence.
CI bootstrap theo utterance trong mỗi ngôn ngữ; chưa điều chỉnh theo speaker
vì manifest nguồn này không cung cấp định danh speaker đủ dùng cho phép chấm.
Không suy thành mức cải thiện chắc chắn ở mọi giọng, hội thoại hay video.

Cùng binary whisper.cppb5454/CUDA0, t2, autolanguage, search mặc định, nguồn
weights FP32;32utterances/model, counterbalance thứ tự base/small. Không prompt
gợi ý transcript, retry-until-pass hoặc tự gán nhãn từ model. Latency bao gồm
nạp worker/model; batch residency phía dưới không thay số latency này.

- [Protocol khóa trước download/inference](../../experiments/hybrid-whisper-small-native-v1.md)
- [Source/conversion479tensor](../../runs/20261114T015122Z-whisper-small-native-v1-0f0231/setup-result.json)
- [Manifest speech32mẫu](../../runs/20261114T015303Z-asr-dev-data-v1-ddf1fd/manifest.json)
- [Paired64nativecalls](../../runs/20261114T015546Z-asr-candidate-benchmark-v1-68e3c7/result.json)
- [Auditor độc lập: original bytes + matrix edit distance + bootstrap](../../runs/20261114T015546Z-asr-candidate-benchmark-v1-68e3c7/independent-audit-v1.json)

Lượt data baf91a fail vì Python wave không đọc floatWAVformat3. Giữ nguyên;
đổi sang ffprobe đọc metadata của cùng bytes, không skip hoặc reencode sample.
Một lần setup và một lần benchmark completed; không thêm “seed” giả cho ASR
frozen deterministic hoặc coi từng từ là independent replicate.

## VRAM được đo khi ASR thực sự còn hoạt động

Benchmark đầu có lỗi harness: sharedGPUgrowth lấy finalsample sau monitor.close,
không phải peak; original resource gate=true không đủ bằng chứng. Giữ raw và
[review riêng](../../runs/20261114T015546Z-asr-candidate-benchmark-v1-68e3c7/resource-measurement-review-v1.json).
Code cho lượt sau đã sửa: quality/timing run không tự tuyên bố resourcequalified;
cần probe active riêng và audit. Không rewrite lượt cũ thành hoàn toàn hợp lệ.

[Follow-up6e53cb](../../runs/20261114T020114Z-asr-resource-followup-v1-6e53cb/result.json)
dùng một ownedCLI/arm xử lý lại32file trong cùng residency, không thêm HTTPservice
hoặc VLM khác. Đây chỉ là resource instrumentation, không phải32TEST mới hoặc
speed của pipeline per-chunk. Input/model/dtype/binary giữ nguyên.

| Resource khi owned ASR còn sống | Base | Small |
|---|---:|---:|
| FreeVRAM vật lý nhỏ nhất |2287MiB|1502MiB|
| SharedGPU tăng lớn nhất được ghi nhận |79.691.776byte|79.691.776byte|
| FreshPDH collections sau loại initialcache |4|7|
| File thực được xử lý |32|32|

Qua reserve512MiB và sharedgrowth≤256MiB. Sampling PDH2s/Nvidia0,25s có thể bỏ
sót đỉnh ngắn, không phải chứng nhận vô điều kiện hoặc8h của small. Không đóng
app khác, dỡ liveVLM, fallback CPU hay hạ precision để lấy số đẹp. Candidate này
đưa thêm khoảng677MB trọng số gốc so với base vào workload đã đo, còn working
buffers/CUDA allocation được đo riêng; không dùng VRAM đầy làm KPI.

## Hỏi đáp video vẫn có nút thắt

[IntegrationP0a98122](../../runs/20261114T020701Z-asr-video-integration-p0-a98122/result.json):
hai private engine cùngALIFseed, cùng frozenQwen live, cùng clip21s đã biết,
learningdisabled. Capture/ASR/vision đủ, không skip hoặc lỗi, checkpoint khớp;
không nhập dữ liệu lab vào live. Small model/revision/path được ghi đúng tên,
không giả thành base.

| Câu hỏi đã khóa | Base + cùngQwen/SNN | Small + cùngQwen/SNN |
|---|---|---|
| Arly Velasquez hạng nào? Chỉ số |“Arly Velasquez xếp thứ15.”|Cùng output; biết số nhưng fail strictformat|
| Adam Hall hạng nào? Chỉ số |“Adam Hall xếp thứ15.”|Cùng output sai, expected9|
| Màu hình tròn phải cuối clip, một từEn |Yellow, đúng|Yellow, đúng|

Strict1/3 ở cả hai arm; giữ nguyên scoring. Small nghe “Aramhorn … xếp thứ9”,
nhưng entity bị sai âm và câu hỏi/ranking/LLM không nối đúng quan hệ. Có
evidence9 mà model trả15 cho thấy giảmWER chưa đủ bảo đảm đúngQA. Chưa có
ablation để phân bổ toàn bộ lỗi giữa retrieval, readout và VLM; không tự kết
luận một thành phần là nguyên nhân duy nhất.

Đây là clip kỹ thuật đã thấy, không formalvideoP1. Các output/ref/citation và
frontiers thật tự hiện ở localhost; label “ứng viên”, WER và lỗiQA tách riêng.
Không gọi0updates trong lab là lifelonglearning hoặc positivehiddengain.

## Dữ liệu công cụ: 37 khác biệt đã được phân loại

[Qualification7cdc6a](../../runs/20261114T014559Z-tool-response-qualification-p0-7cdc6a/result.json)
đối chiếu266actual tool calls ở78conversations:

| Nhóm | Số call |
|---|---:|
| ExactJSON |229|
| Chỉ thứ tự collection được phép |6|
| Tool trả strictsuperset bản ghi |20|
| Chỉ generatedID |6|
| Lệch key/schema response |2|
| Khác payload còn cần rà |3|

Không làm rơi bản ghi trùng, không sort mọi loại list, đổi key hoặc ép bool thành
number. b77ce6 ban đầu giữ riêng; sửa type-strict comparison cho lượt7cdc6a không
đổi các số trên. Không có conversationID hay normalized toàn userdialog trùng;
near/template leakage vẫn cần đánh giá trước DEV/TEST.

[OracleSQLde5e49](../../runs/20261114T014457Z-tool-query-oracle-p0-de5e49/result.json)
chọn độc lập trên **state ngay trước call**, không đọc goldresponse,model text hoặc
futureturn.49readqueries qua5API khớp contract ở49/49actualresponses;23annotation
không phải fullqueryresponse tương ứng. Trong23có20subset mà upstreamchecker có
chủ ý cho phép; không gọi chúng sai theo định nghĩa benchmark. Ba còn lại có
khác date/content/attendees so snapshot đang chạy.

Hai AddReminderannotation dùng alarm_id thay reminder_id. GeneratedIDs khác có
thể cần binding, không mặc định map theo phỏng đoán. Các số này xác minh cần
tách tool-contract, annotation và userintent; chưa tự tạo nhãn học. Oracle dùng
state từ original mutation simulator trong prefix, không độc lập xác minh mọi
mutation hay hiểu yêu cầu tự nhiên. Corpus vẫn dataset_qualified_for_learning=false.

## Quyết định và việc tiếp theo

- **GO DEV cho Whisper-small như ứng viên ASR nguyên precision dùng thêm VRAM.**
  Chưa đổi live; cần xác nhận trên bộ speech/video khác và profile tích hợp.
- **PIVOT cho QA/representation/credit.** Đúng transcript hoặc thêm GPU không
  đảm bảo dùng đúng sự kiện/quan hệ; currentSNN chưa thắng matchedstrongRAG.
- Lượt8hbase tiếp tục kiểm đúng profile đã khóa. Nó không chứng nhận cho small,
  không thay qualityP1/P2, meanforgetting3/worst5điểm hoặc proofhiddenadvantage.
- Reward learning phải có task-effect và response-support contract, không lấy
  API success, WER tốt hơn hoặc lời self-critique làm nhãn kỹ năng đúng.
