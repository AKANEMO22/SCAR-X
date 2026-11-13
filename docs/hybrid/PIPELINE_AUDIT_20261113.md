# Kiểm tra pipeline SCAR-X Hybrid — 13/11/2026

**Kết luận: đường ống đã hoạt động ở mức thử nghiệm; chưa đủ hiệu quả để coi
là trợ lý video liên tục hoặc hệ tự học đã được kiểm chứng.** Đã sửa ba lỗi
kỹ thuật; không thay ALIF, quy tắc học, trọng số, precision hoặc context16K.
Không thực hiện yêu cầu Antigravity/LoRA đã được người dùng rút lại.

## Các phép kiểm tra thực

Trần toàn hệ4.500.000.000byte, CPU, Qwen3.5-0.8B BF16/F32 frozen và Whisper
base FP32. Native parity/integrity vẫn chạy khi khởi động. Video có21giây
giọng người Việt công khai FLEURS, nhãn độc lập và hình học do script vẽ rõ
là fixture. Không giả đầu ra model. Chỉ thu cửa sổ fixture của lượt kiểm tra;
không microphone, không nghe loa hoặc quay màn hình riêng tư.

- Lượt trước sửa: `20261113T152708Z-hybrid-pipeline-audit-v1-af1dd5`.
- Lượt kiểm tra sau sửa: `20261113T153749Z-hybrid-pipeline-audit-v1-d468f9`.
- [Protocol gốc](../../experiments/hybrid-pipeline-audit-v1.md),
  [phạm vi sửa/acceptance](../../experiments/hybrid-pipeline-repair-v1.md),
  [đính chính harness/cách ly](../../experiments/hybrid-pipeline-audit-harness-v2.md).
- [Chấm lại trước sửa](../../runs/20261113T152708Z-hybrid-pipeline-audit-v1-af1dd5/fixture-score-audit.json),
  [chấm lại sau sửa](../../runs/20261113T153749Z-hybrid-pipeline-audit-v1-d468f9/fixture-score-audit.json).

| Pipeline | Kết quả thực | Đánh giá trong phạm vi đã đo |
|---|---|---|
| Chat streaming |17+25→42; tổng0,48s, token nội dung đầu0,265s sau sửa | Chạy được trên câu rất ngắn; không suy rộng sang agent dài |
| Function/tool result | Model gọi lookup_number(ALPHA), đọc kết quả17 rồi trả17;2,78s | Protocol round-trip qua, chưa chứng minh chọn đúng mọi công cụ Hermes |
| Hermes native owner | Lượt chat thật trả PIPELINE_OK;4.128input/4outputtokens | Routing hoạt động; chưa có latency đáng tin từ probe này |
| Ảnh inline | Nhận màu hình tròn blue đúng;1,11s | Vision có đường xử lý thực, chưa là benchmark hiểu ảnh tự nhiên |
| Video upload |21s nhận đủ3đoạn, không skip; xong sau31,656s | Capture1× hoạt động; tổng thời gian lớn hơn độ dài nguồn, chưa chứng minh sustained real-time |
| Hỏi đầu/cuối video | Trước sửa blue/blue; sau sửa blue/yellow đúng;2,36/2,45s | Lỗi chọn bằng chứng đã sửa trên fixture đã biết |
| Cửa sổ đang phát | WGC thật, thu và phân tích tới8,328s; stage xong14s | Chạy trên cửa sổ fixture, không kiểm loopback audio/video tự nhiên |
| ASR tiếng Việt |26word-edits/38reference-words =68,42% WER, cả hai lượt giống nhau | Chưa đủ chính xác; không dùng bản phiên âm này làm nhãn đúng |
| Save/rollback | Trọng số/trạng thái/RNG/controller snapshot khớp byte; evidence và audit còn | Engineering persistence qua; không chứng minh neural đã học kiến thức |
| Cold restart | Snapshot đã pin và active state khớp hash,7evidence còn; hỏi lại đầu video đúngblue | Trí nhớ SSD/truy xuất còn sau restart |
| Tự xem lại | Lượt đầu scheduler tự kiểm đúng42,0synapse/hidden updates và0replay | Lifecycle có hoạt động; chưa có skill gain |
| Tự xem lại lượt2 |0review/75s, có12request Hermes thật và11tool requests trong giai đoạn này | **Không kết luận**: bị lẫn foreground, không phải bằng chứng scheduler hỏng |
| Tra nguồn web | Audit log production có5lượt search bắt đầu/kết thúc,7mục tiêu trước đó đang deferred | Chỉ xác nhận có truy xuất; không đo lượt web mới hoặc chứng minh giải đúng mục tiêu |

Số đo RAM bảo thủ gồm gateway, childworkers, Hermes và sharedGPU: lượt đầu
peak4.092.870.656byte; lượt sau3.907.923.968byte; sau cold restart3.457.626.112byte.
Đều dưới4,5GB trong các lượt này. Lưu lượng và bộ nhớ Hermes khác nhau, **không
coi chênh lệch peak là tối ưu RAM do bản sửa**. Không có kiểm thử8giờ; ASR/vision
frontier21s không có nghĩa hiểu mọi frame hoặc phiên âm chính xác21s.

## Nguyên nhân và phần đã sửa

1. **Chọn sai mốc thời gian.** Query cuối video vẫn gửi frame5s. Timeline có
   ảnh20,5s đúng nhưng relevance head chưa được kiểm chứng đẩy ảnh cũ lên trước.
   Nay ràng buộc đầu/cuối được áp trước relevance, chung parser có ranh giới từ
   Việt/Anh. So sánh cả đầu lẫn cuối giữ nhiều sự kiện; không gộp thành một.
   Sau sửa log xác nhận câu đầu dùng5s, câu cuối dùng20,5s. Đây là sửa chọn dữ
   liệu bằng luật, **không phải hidden learning hoặc model thông minh lên**.
2. **Ảnh màn hình lịch sử bị thay bằng ảnh hiện tại.** `ScreenMemory.snapshot`
   cũ luôn lấy latest. Nay câu hỏi mốc lịch sử dùng đúng source của observation
   đã chọn; thiếu/TTL-expired thì báo thiếu, không thay bằng ảnh mới. Fixture màu
   blue→yellow tái hiện bản cũ thật sự gửi ảnhyellow và bản mới gửiblue đúng.
3. **Lỗi ASR chặn vision của video upload.** Non-resource ASR/extraction error
   được ghi theo interval, giữ audio thiếu và cho vision tiếp tục. MemoryError
   vẫn truyền lên guard. Stream có `degraded/asr_errors` và Debug đánh dấu
   phân tích chưa đầy đủ; prompt có coverage thiếu. Không bịa transcript hoặc
   nâng asr_frontier khi đoạn đó lỗi. Đây là fault isolation của video file;
   cùng đường audio của screen còn cần kiểm tra riêng.

Regressions ép relevance thích ảnh sai thời gian, mất source lịch sử, ASR lỗi,
MemoryError, cold checkpoint và native ALIF parity. 70Pythonchecks và4Node
composerchecks đã qua, không skip/xfailed trong bộ cuối. Các test kỹ thuật không
thay thế đo chất lượng trên video mới.

## Những lỗi của chính phép kiểm tra

**V1 chưa cách ly request đầy đủ.** Dùng lại8775 để người dùng xem trace khiến
Hermes cũng có thể gửi yêu cầu thật vào kho kiểm tra. Trong lượt sau sửa, các
fixturechat/tool/vision/video đã xong trước khi foreground này xuất hiện; stage
idle bị lẫn. Không chấm stage đó PASS/FAIL hoặc tuyên bố đây là clean E2E run.
[Traffic/source audit](../../runs/20261113T153749Z-hybrid-pipeline-audit-v1-d468f9/traffic-and-source-audit.json)
ghi số lượng và thời điểm nguồn; không công bố nội dung hội thoại ngoài fixture.

Runnerv2 nay yêu cầu nonce riêng cho **mọi POST**, kể cả presence/upload/chat,
foreignrequest nhận503maintenance; GETtrace vẫn đọc được. Đã thử HTTP thật với
4route ngoại lai và request đúng nonce. Không mở lượt full-inference thứ3 chỉ
để làm báo cáo trông sạch. Muốn đo lại độc lập dùng run mới và source hash mới.

**Probe Hermes chấm nhầm trường.** `display_projection` trả row.text, parser
ban đầu chỉ đọc content nên báo timeout giả. Đọc lại cùng turn xác nhận
PIPELINE_OK; receipt false-timeout được giữ cùng
[đính chính](../../runs/20261113T153749Z-hybrid-pipeline-audit-v1-d468f9/hermes-owner-probe-audit.json).
Không gửi lại, không huấn luyện và không suy ra latency từ90s polling lỗi.

Test lịch sửscreen lúc đầu còn thiếu stub model.relevance và xfail nhầm
AttributeError. Đã sửa fixture rồi chạy lại nguyên committedsource cũ;
[bằng chứng RGB](../../runs/20261113T152708Z-hybrid-pipeline-audit-v1-af1dd5/baseline-structural-reproduction.json)
đúng là ảnhyellow khi yêu cầu đầublue. Giữ rõ lỗi test; không dùng xfail cũ làm
bằng chứng. Bản cuối không dựa vào xfail.

Raw v1 receipts/eventbatch có thể chứa dữ liệu Hermes thật nên giữ nguyên bytes
trong `local/<run-id>/raw-audit-receipts`, không commit. `private-raw-index.json`
giữ hash/đường dẫn; `*.public.json` chỉ giữ kết quả fixture và thông số kỹ thuật.
Hai SQLite sandbox còn nguyên để đối chiếu. Database/model production và lịch
sử Hermes không bị rollback hoặc đổi trọng số bởi các phép kiểm tra này.

## Nút thắt và thứ tự xử lý tiếp

1. **ASR/coverage:** tập Việt/Anh độc lập, silence và tên riêng, xác định mất
   tiếng do cắt chunk/nhận ngôn ngữ/model. Giữ transcript quan sát khác nguồn
   xác minh. Một sample WER68,42% không đại diện corpus nhưng đã đủ chỉ ra lỗi
   sử dụng thực. Chưa chạy diagnostic21s nguyên đoạn đã đề xuất; chưa tuneASR.
2. **Video dài liên tục:** đo backlog, lagp95, skip, frame coverage và chất lượng
   khi hội thoại xen kẽ; screen-loopback và lỗi ASR trong screen chưa được đo.
   Có thể thu1× nhưng xử lý chậm hơn nguồn, nên cần báo rõ phần chưa phân tích.
3. **Công cụ/agent:** native tool contract qua không có nghĩa0.8B làm việc nhiều
   bước tin cậy. Cần tasks riêng về chọn/callingtool, đọc kết quả, dừng vòng lặp,
   có hạn mức và giữ nguyên truy nguyên lỗi. Không suy ra từ lời chào hay PIPELINE_OK.
4. **Tự học có ích:** cần verifier/feedback có nguồn và đánh giá retention,
   hidden-vs-readout/frozen; hiện0updates ở các fixture và không skill-gain
   measurement. Review tự sinh chỉ tìm chỗ thiếu, không thành groundtruth.
   Hiện `PublicWeb.search` tìm theo chủ đề công khai đã được cấp, không biến
   nguyên câu hỏi riêng tư thành truy vấn. Cần kiểm độ phù hợp nguồn/mục tiêu,
   không chỉ số trang đã đọc hoặc số câu hỏi tự sinh.
5. **Vận hành dài hạn:** soak8giờ, save/rollback lúc tải cao, quota/cache/thread
   growth, nguồn hết hạn, khởi động lại giữa đoạn; cap4,5GB không là proof hoàn tất.

Dữ liệu kiểm tra sẽ hiện trong [trace localhost](http://127.0.0.1:8775) bằng event
`pipeline_audit.acceptance_summary` với thời điểm nguồn/run; đó là replay báo cáo
của phép đo thật, không mô phỏng model đang suy nghĩ. Normalgateway được khôi
phục với dữ liệu gốc; source sửa nằm ở `src/scarx/hybrid`, không chạm main/R13.

Mã: `scripts/hybrid_pipeline_audit.py`, `hybrid_hermes_pipeline_probe.py`,
`audit_hybrid_pipeline_receipts.py`; fixtures: `tests/test_hybrid_pipeline_audit.py`.
Source trước sửa lấy bằngbaseSHA trong manifest; source đã đo sau sửa có bundle
runtime/hash. Sourceharnessv1 lưu nguyên, v2 không chạy đè các run cũ.
