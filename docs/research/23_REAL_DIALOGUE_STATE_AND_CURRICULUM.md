# G1: hội thoại thật, state interference và curriculum có guard

Ngày09/11/2026. Mục tiêu normalchat/vision chưa đạt. Những phép thử dưới đây
dùng model thật và local synaptic updates thật; không dùng câu trả lời mẫu
để trả lời thay neural core. Các checkpoint thử chưa được thay vào model live.

## 1. Antigravity và Qwen đã chạy thực

[48prompts](../../artifacts/antigravity-g1-state-latest.json) do CLI Antigravity
đã cài sinh từ brief công khai, sau đó đọc toàn bộ và kiểm intent. Không gửi
private transcript/weights. Target là8câu style dự án đã có, không là oracle
sự thật.4case có word sequence trùng fixture cũ khi bỏ dấu câu cuối; không
gọi toàn bộ48case là năng lực ngữ nghĩa mới hoặc finaltest độc lập.

[Qwen4B](../../artifacts/qwen-g1-state-reference-latest.json) đã trả lời đủ
48exact userprompts, khôngsystem/history/tools/retrieval.17responses bị giới
hạn64tokens. Server đã tắt; không thay cấu hình Hermes/Qwen9B, không proxy vào
SCAR-X. Reference peakRSS3,20GB/commit3,88GB dưới cap riêng4,5GB; **đây không
là RAM2GB của SCAR-X**. Không chấm Qwen bằng exact-match tới một style câu ngắn
hay coi lời tự khai của Qwen về khả năng là sự thật đã kiểm chứng.

Raw private SCAR-X outputs/snapshots chỉ giữlocal. Bảng localhost ghép được
archived SCAR-X với Qwen mới bằng đúngcaseID/prompt, ghi riêng thời điểm và
nguồn; không giả Qwen đang chạy sau khi kết thúc.

## 2. State không giải thích hết lỗi

[Protocol](../../experiments/g1-conversational-state-audit-v1.md),
[896inferences](../../runs/20261109T161548Z-g1-conversational-state-audit-v1-e7b5e8/report.md):
112case(64cũ+48AGY)×4state×2checkpointscopes. Mọi weight/updatecounter bất biến.
Zero-inputidle100ticks, publicidleprobes, savedstate và canonicalreset chỉ trên
bản sao; không reset runtime hoặc xóa trí nhớ người dùng.

| Checkpoint / state |64câu quen|48AGYphrasing|
|---|---:|---:|
| Public release / saved |75,00%|12,50%|
| Public release / silent100 |85,94%|12,50%|
| Public release / publicidleprobes |82,81%|8,33%|
| Public release / resetclone |85,94%|12,50%|
| Current private / saved |1,56%|0,00%|
| Current private / silent100 |3,13%|0,00%|
| Current private / publicidleprobes |3,13%|0,00%|
| Current private / resetclone |3,13%|0,00%|

State gây khác biệt, nhưng reset không phục hồi bản current về chất lượng
release. Hai checkpoint khác cả weights, vocabulary và allocation; chưa đủ
tách riêng mỗi nguyên nhân từ so sánh giữa scope. Không kết luận chỉ idle
làm model quên, hoặc dùng reset production làm cách sửa. Các điểm exact-style
là diagnostic hẹp, không phải accuracy hiểu ngôn ngữ.

## 3. Ôn lại trên bản sao chưa qua acceptance

[Rehearsal](../../experiments/g1-basic-rehearsal-v1.md):63publiccases sau khi
loại1prompt có personal override,24passes, full/readout×3curriculumorders,
replay các arithmeticproof hiện đã biểu diễn được. Cùng privateinitialization,
không là3modelseeds độc lập. Không train48AGYdiagnostic prompts.

Câu quen sau học phục hồi khoảng84–94% ở reset, nhưng AGY chỉ khoảng10–19%.
Không condition nào qua toàn bộ ngưỡng. Một số “lost” của strict-token guard
chỉ là `15`→`15.`; không được gọi là mất giá trị toán học. Runtime normalized
guard vốn đã bỏ terminalpunctuation. Giữ kết quả strictstyle gốc và ghi phân
biệt, không sửa protocol/counter cũ để nhận thành công.

[Surfaceacceptance](../../experiments/g1-terminal-surface-acceptance-v1.md) đánh
giá lại các checkpoint bất biến với optional terminalpunctuationnormalization,
giữ đúng phép so guard runtime đã có. Cách này chưa qua gate và làm các proof
đã học dưới input có dấu câu trả lời sai. Không bật normalizer hoặc nhập các
weights đó vào live. Default codec/encoder cũ giữ nguyên.

## 4. Corpus mới đã học vào bản sao, không vào model live

Antigravity thực sự sinh thêm160publicphrases,128train/32development dự kiến.
Lô2 kết thúc bằng native `finish` chứa đúng structuredJSON; bộ lọc đã nhầm
tín hiệu schema-finalization là externaltool và dừng. [Recovery](../../artifacts/antigravity-g1-curriculum-latest.json)
kiểm exactpayload bằng kết quảCLI đã lưu, không gọi teacher lại; giữ journal
lỗi ban đầu. Không mở quyền filesystem/terminal hoặc skip-permissions.

Đọc toàn bộ160mẫu và loại12câu mơ hồ giữa chào/tạm biệt hoặc instruction vs
teaching. [Corpus reviewed](../../artifacts/g1-curriculum-reviewed.json) còn
117train/31development, chưa dùng performance model để lọc/đổi nhãn. Target
vẫn là8style responses; một số câu English ánh xạ tới câu Vietnamese theo
fixture cũ, nên đây chưa phải chuẩn normal bilingual conversation.

[Vòngcurriculum](../../experiments/g1-curated-curriculum-v1.md) train thật trên
12privateclones,4conditions×3orders,3072examples mỗi lượt, hidden/vocab không
đổi, output-localdelta hiện có.1/5slot replay3verifiedarithmetictype để tránh
ghi đè;4/5slot lấy8intent cân bằng. Nhánh surface áp dụng normalization cả
training/inference. Đánh giá dev-only rồi khóa condition trước acceptance.

| Condition |Mean development strict-style|
|---|---:|
| Raw + olddata |2,69%|
| Raw + expandeddata |9,68%|
| Surface + olddata |3,49%|
| Surface + expandeddata |11,56%|

Condition chọn bằng dev là surface_expanded. Sau acceptance, known/reset chỉ
58,73–68,25%; AGY/reset12,50–14,58%. Phần lớn mathproof đã giữ được nhưng
không đạt acquisition/ngôn ngữ. **CURATED_SCREEN_FAILED_KEEP_LIVE**; không
chọn lại condition thứ hai, nângepochs/LR hoặc giảm ngưỡng sau khi thấy điểm.
Thêm data giúp một số số đo, chưa giải quyết representation/generalization.
Data117train đã được dùng trong experimentclones;31dev chỉ đánh giá. Không
được mô tả toàn bộ corpus là chưa từng học, cũng không nói đã nạp vào live.

## 5. Teaching có nguồn qua learner đang dùng

Một đường product riêng dùng đúng projectfixture indexed và sourceSHA để
gửi explicitlesson qua controller hiện có. Role `supervisor`, không giả làm
humanfeedback; target chỉ cung cấp cho learning, không trả lời bằng lookup.
Idempotency và personaloverride được kiểm: target dự án khác lời dạy cá nhân
thì bỏ qua, không supersede. Checkpoint trước mỗi lesson, giữ guard/32attempts
và idlepolicy cũ; không import failedclone hoặc bật surfaceoption.

Hai ví dụ `chào`/`hi` đã chạy qua actualneuralprobe trước/sau. Câu sinh giữ
“Chào” và “Hello”, không coi đây là cải thiện.4micro-updates được giữ,60lần
rollback; counter241347→241363. Không nhập failedcurriculumweights hoặc bật
surfaceoption. [Receipt](../../artifacts/g1-project-supervision-latest.json)
lưu nguồnSHA, checkpointtrước và cácđầura thực.

Ngày10/11: trace cho thấy các đề xuất bịtừchối có thể lặp nguyên trạng sau
rollback. Controller nay dừng sau2signature modelstate+prompt+target giốnghệt
bịguardtừchối; không đổi guard/lr/learningrule. Signature/count lưu và validate
trong checkpoint; cậpnhậtthànhcông xóastreak. [Acceptance thật](../../artifacts/g1-guard-stop-latest.json)
quan sát2rejections rồi dừng, weights/counter không đổi, vẫn “Hello”. Đây là
sửa resource-control, không gainknowledge hoặc forgettingpass.

Hậu kiểm48publicdigits trêncanonicalclones:81,25%trước/sau, không mấtcase đúng
cũ; hai arithmetictrainingpoints4+4/7+8 không mất. Không suy ra ảnhthật hoặc
retentionlâudài.138testsfullsuite và14focusedcontrollerchecks qua ởmốc này;
các reader/observationchecks bổ sung được ghi trong deliveryreceipt cuối.

## 6. Giới hạn và bước kế tiếp

Không có thành công tổng quát từ các vòng này. Trọng tâm tiếp theo vẫn là
input/temporal representation và credit assignment, không chỉ tăng số câu
thuộc lòng. Dataset, protocol và các failure giữ nguyên để truy vấn lại. Mọi
architecture/learning-rule khác cần mô tả, signal/locality/budget và phép kiểm
riêng; không đổi tên family đã dừng để tiếp tục tuning.

Trace tự ghép archivedQwen khi **exact public prompt** trùng, không dùng
caseID đơn lẻ và không gửi prompt cá nhân đi đâu. Các câu hỏi mới không có
reference tiếp tục ghi thiếu. Lõi live đã được chạy trên48prompt đã review,
với record_experienceFalse; không tạo fakehumanepisodes/questions/nhãn học.
Counter giữ 241363 trước/sau; điểm này chỉ kiểm đường quan sát. Các câu trả
lời yếu được giữ nguyên, không dùng target/Qwen để thay đầu ra. Full suite cuối
có 140 kiểm thử qua; sửa race để completion chỉ được công bố sau khi lịch sử
quan sát đã ghi xong, không nới RAM guard hoặc research gate.
