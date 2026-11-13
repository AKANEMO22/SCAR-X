# Context, tự xem lại và đối chứng model online

Audit13/11/2026 trên nhánhHybrid. Người dùng muốn benchmarkSCAR-X hoàn thiện
với modelonline trong tương lai, không yêu cầu giả định bằng điểm trước khi đo.

## Context và công việc dài

Runtime phục vụ16,384tokens thực, không khai64Kgiả. CheckpointQwen gốc có
`max_position_embeddings=262144`; đó không phải proof runtime4GB xử lý được
ngần ấy. KVbuffer ởnativeBF16:6full-attentionlayers ×2(K/V) ×2KVheads ×256dim
×2bytes =12,288byte/token.16K→32K thêm201,326,592byteKV, chưa tính scratch,
ảnh/ASR/Hermes. Vì hệ từng sát/vượt4GB, chưa tăng context mà không qualification.

Trace`events`SQLite và transcriptHermes là dữ liệuSSD, không phải KVwindow.
Debug trước chỉ render150events; nay có phân trang đọc nhật ký cũ, không nạp
toàn bộ vàoRAM hoặc prompt. Currenttrace vẫn tự cập nhật, không cần chọnrun.
Đây là quan sát lịch sử, không tuyên bố model đã nhớ/hiểu toàn bộ những gì lưu.
Khả năng giữ chỉ dẫn và dùng đúng bằng chứng qua taskdài vẫn cần bộ đo riêng.

## Vì sao không thấy tự học?

Liveaudit trước sửa:neuralupdates0,hiddenupdates0,replay0,goals0,last_review0;
14`gap_proposal_rejected`. Hai draft cuối bị cắt giữaJSON với maxoutput96tokens.
Idleproposer đợi`present=false`(600s); replay chỉ xảy ra khi cóopengoal và
policy chọnconsolidate. Không cógoal/nhãn thì khôngreview/replay theo ý định.
Các con số này được ghi theo thời điểm, không biến thành claim toàn lịch sử.

Sửa engineering, không đổi ALIF/local-learning rule/modelweights:

- Mỗi answer thực đưa vào queuebounded32. Title-generationJSONhelper bị bỏ qua.
  Bootstraptốiđa8lượt cũ chưa xem, bảo toàn sourceID. Metadatareview/counters/
  queue nằm cùng checkpointcontroller; explicit additive migration cho checkpoint
  trước`review-v1`, không thay RNG/trọng số. Rollback giữ audit.
- Review có thể chạy sau30s không tương tác, cách nhau≥60s và dưới10%CPU duty;
  không phải đợi mấtpresence600s. Foreground/RAM/pause preempt, trace nêu blocker.
- Dùng JSONschema đầu ra có giới hạn,160outputtokens; vẫn kiểm parser/schema,
  không chấp nhận draftbịcắt. Reviewmột trải nghiệm có thật mỗi lượt, không bịa
  “đang suy nghĩ” hay tự luyện ngẫu nhiên để lấp trace.
- Verifier toán nguyên chỉ kiểm được câu trả lời số rõ ràng. Sai được verifier
  kiểm độc lập có thể dạy ALIF hành độngverify bằng đườngguardđãcó; lịch trình
  ban đầu thuộc runtime, không gán là policyneural đã chọn đúng. Đúng sẵn không
  tự tạo nhãn từ độ tự tin. Critique tự sinh chỉ tạo gap cần tìm nguồn/hỏi người.
- Replaynhãn human/machine có nguồn chạy được ngay cả khi không cóopengoal;
  replaykhông ghi chính nó thành trải nghiệm mới. Không replayunverifieddraft.
- Debug hiện queued/reviewed/failed/replayupdates/blocker/lastresult. Tách
  reviewactivity, weightupdate và measuredqualitygain, không đánh đồng chúng.

VLM vẫn **frozen**: tự review/policy/memorylearning không tự nâng kỹ năng viết,
code/suy luận trong trọng sốQwen. SNNhiddenadvantage, generalization và forgetting
vẫn chưa chứng minh trên Hybrid. Không hứa hệ càng chạy càng giỏi vô điều kiện.

Kiểmtralive sau sửa tại receipt`20261113T135806Z-hybrid-review-lifecycle-82405b`:
4reviewthật đã hoàn tất,0JSONreject,0weightupdates; RAMbảo thủ3.593GB<4GB.
Gap cần xác minh đã được tạo từ review, không tự dùng critique làm nhãn đúng.
61Pythonaffected/native/migrationchecks và4Nodecomposerchecks qua. Đây là
engineeringlifecyclePASS, **không** phải benchmarkqualitygain hoặc self-improvementPASS.

## Benchmark đối đầu model online khi đủ điều kiện

Không chạyAPItrảphí hoặc mở freshscientificbenchmark trong đợt sửaengineering.
Trước lượt đối đầu cần chọnmodelID/version thực, phí tối đa và quyền đưa dữ liệu
raonline. Chỉ dùng public/licensed fixtures nếu chưa được phép gửi dữ liệu riêng.

Hai bài đo tách biệt:

1. **Runtimefidelity:** cùngcheckpoint gốc, tokenizer/chattemplate, precision,
   prompt/media, decoding, tools, sourceversions. Đối chiếu logits/answers và
   metric sai lệch; CPU/GPUfloat có thể không bit-identical. Ba ca next-tokenP0
   hiện tại chưa đủ chứng minh ngang năng lực trên benchmark tổng quát.
2. **Capabilitycompetition:**SCAR-X hoàn thiện so với online khác. Cùngquestion
   split/nguồn/toolpermission và ngân sáchtool/time đã đăng ký; báo riêng chế
   độ commoninputbudget và fullsystembudget, không giấu chênh lệch RAM/context/
   compute. ĐoVi/En/instruction/knowledge/math/code và video/audio/temporal,
   retention20stages riêng. Giữ các ngưỡng đã chốt, không nới sau xem kết quả.

Chỉ gọi“ngang trên bộ đo”khi lower95%CI(SCAR-X−online)≥−5pp **từng nhóm**.
Không gộp điểm mạnh bù điểm yếu rồi gọi ngang tất cả. Savefullsample outputs,
groundtruth/evidence, latencies/costs/modelversions và mọifailures. FinalTEST
khóa trước, không dùng để chọnprompt/rule. Frozenbench tắt learner/idlefeedback
và chạy datastorecô lập; continualbench cho phép học từ cùng feedbackstream,
không lộ futurelabels/taskboundary. Seeds không thay cho independentvideoitems.

Có thể bảo đảm quy trình công bằng/tái lập và công bố kết quả thật; không thể
bảo đảm trước modeloffline4GB sẽ bằng/better một modelonline lớn hơn.
