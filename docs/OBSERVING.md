# Quan sát SCAR-X trên localhost

Mở **http://127.0.0.1:8765/#agent-work**. Đây là bảng theo dõi tự cập nhật
những gì Codex đang thực hiện và các giao dịch model thực tế. **Không cần chọn
model hoặc bấm chạy.** Server vẫn hoạt động khi đóng trang. Poll chỉ đọc,
không tự tạo job, gọi teacher hoặc huấn luyện để làm trang trông đang bận.

## Công việc kỹ thuật

`#agent-work` hiện những công việc Codex công bố trong lúc cập nhật hoặc kiểm
tra: nội dung thay đổi, thời điểm, trạng thái bắt đầu/hoàn tất/lỗi và bằng chứng
như hash nguồn, exit code, số đo API. Mục mới tự lên đầu mỗi1,5giây. Các phần
câu trả lời, học, neuron và thí nghiệm cũng tiếp tục tự theo lượt mới nhất.
Không cần chọn nguồn hoặc khởi chạy. Các điều khiển ở phần trò chuyện/dạy
thủ công là tùy chọn riêng, không là điều kiện để xem trace.

Đây là nhật ký kỹ thuật do runner/Codex ghi qua `runtime.work_trace.activity`,
không tự suy ra mọi thao tác từ filesystem hoặc lời nói. Không gọi nhật ký này
là suy nghĩ neuron. Mốc “đã bắt đầu” là sự kiện có thời điểm; trạng thái job
đang chạy được xác minh riêng bằng process identity ở `#research-lab`.

Lượt [kiểm tra thật](../artifacts/passive-work-trace-latest.json) chạy148tests,
một inference serving21ticks và ba lần đọc trace, giữ counter241363; không
huấn luyện mới để lấp giao diện. 100logloss đo thật trước đó vẫn đọc được.
Kết quả chỉ xác nhận phần mềm/luồng quan sát, không xác nhận model nói/nhìn tốt.

## Câu trả lời kiểm thử

`#test-answers` hiện ngữ cảnh corpus, đầu vào thực sự gửi model, câu model sinh,
một đáp án mẫu và đầu ra Qwen khi có phép đo cùng case. Không lấy đáp án mẫu
hoặc Qwen làm neural input/nhãn tự học. Lượt chat thông thường cũng tự ghi đầu
ra neural gốc, tách khỏi câu hỏi/acknowledgement do controller định dạng.

Qwen hiện là **bản đo thực đã lưu**, có run/thời điểm và finish_reason; không
phải server đang chạy. Trường hợp chưa đo ghi thiếu. Corpus có thể dùng prior
assistant replies do dataset cung cấp; đây không phải output SCAR-X tự sinh.
Serving hiện chỉ nhận latest user text, Qwen đối chiếu từng nhận full prefix;
trang ghi rõ khác biệt, không gọi đây là benchmark cùng điều kiện.

Qwen4B đã đo thêm48publicphrases, reader tự ghép khi prompt khớp chính xác.
CaseID một mình không đủ; không gửi privateprompt cho Qwen. Những rawoutputs
của model cá nhân chỉ giữlocal và hiện trênlocalhost, không đónggói.

Lượt xác nhận [live-observatory](../artifacts/live-observatory-latest.json)
đã chạy24 inference thực. Mỗi lượt được ghi ngay trong transaction lưu state;
không đợi điền một bảng kết quả mẫu. Inference không nạp các target thành lesson.

## Quá trình học

Mục [Toàn bộ đầu ra đã lưu](http://127.0.0.1:8765/#answer-archive) tự theo tệp
output nghiên cứu hoàn tất gần nhất trong lượt đang chạy/mới nhất. Hiện mỗi
tệp có95case (dev31/prior48/train16), gồm cả câu sai; không chọn theo accuracy.
Đây là bản đo đã lưu, thời gian ghi là mốc lưu tệp, không giả từng câu đang
được sinh lúc mở trang. Nguồn/cờBPTTcontrol tách rõ khỏi modeltrongHermes.
Observedassociation29 dùng nhãn riêng (khôngBPTT), ghi rõ answerobserved chỉ
trongtraining và gold-freeinference; không trộn vào nhãncontrol hoặcmain.
Visionpathaudit31 hiển thị240calls đãđo, ảnhsyntheticRGB và ảnh8×8gray được
phóngnearestneighbor. Đầu ra rawdigitJSON được ghi rõ khôngVQA/ngônngữ; target
từrenderer chỉđối chiếu. Reader kiểmsourcehash vàwhitelist12ownedPNG, không
đọc arbitrarylocalphotos. Qwenvisionmissing ghi rõ vì modelhiệncótext-only.
Jointvisualword32 có864DEVwordoutputs từtệp hoànthành mới nhất, cóRGB/query/
rendererreference vàcờfull/no_image/no_question, hiddenfrozen. Sinhwordneural
khác rawdigit/class-nameformatter; cácbaseline bịzeroInput ghi rõ. Allcases
đượcgiữ kểcảsai; filetimestamp khônggiả inferenceđangchạy. Macro/drop theo
điểmphầntrăm được tách khỏi technicalcoldresumepass/qualitySTOP.
Qwen chỉ ghép khi khớp chính xác publicprompt và có outputđãđo; thiếu thì ghi
thiếu. Reader không chạy inference/học, chỉ công bố publicprompt/targetpairs
từ corpus và dữ liệu đãreview/hash. Các fieldprivate/weights/guard không hiện.

`#learning-observer` hiện các lượt dạy/học thực: số lượt, loss teacher-forced
token NLL có EOS, số bước cập nhật token, số weights khác ở cuối lần thử và
L2 norm thay đổi `w_in/w_rec/w_out`. Số weights đổi là thay đổi **ròng**, không
phải số lần ghi vào synapse. Không suy ra representation từ ΔW khác0.

Guard giữ hoặc rollback hiện riêng. Sau rollback, ΔW giữ lại là0; ΔW/loss của
lần thử vẫn lưu nếu đã đo được. Lỗi xảy ra trước khi có số đo được ghi thiếu,
không giả loss0. **Cập nhật 2026-11-10: dạy thủ công NAY CÓ chạy guard.**
Trước đây đường dạy thủ công (form web/Hermes) bỏ qua guard hoàn toàn, nên một
bài dạy có thể được ACCEPTED trong khi câu trả lời đang đúng bị phá (đo được:
`chào bạn` từ `Chào bạn.` thành `Bạn có bạn.` với `guard_lost` không hề được
tính). Nay guard chạy trên **mọi** đường dạy; `guard_checked` và `guard_lost`
luôn có trong metrics. Hệ quả trung thực: tỷ lệ chấp nhận trên bộ sản phẩm chỉ
còn **12,5%** — đó là trạng thái interference thật, không phải lỗi mới.

**Cập nhật 2026-11-10 (lần 2): guard chấm theo HAI tiêu chí độc lập.**
Trước đây chỉ có một phép so `normalized()`, phép này **bỏ dấu câu cuối**
(`.strip(' .!?')`), nên `'Hello'` và `'Hello.'` được coi là bằng nhau. Hệ quả đo
được: guard **bảo vệ dạng cụt** và từ chối mọi bài dạy sửa nó — một cơ chế bảo vệ
đang bảo vệ khiếm khuyết. Nay `Generation._pair_verdict()` trả hai cờ:

| Cờ | Nghĩa | Guard có chặn? |
|---|---|---|
| `content_ok` | Đủ từ **và** không lặp (so khớp **chính xác**, không chuỗi con) | **Có** |
| `complete_ok` | Kết thúc bằng `.` `!` `?` **và** `content_ok` đúng | Không — chỉ báo cáo (`guard_style_lost`) |

Vì sao tách như vậy: một dạng cụt **phải còn sửa được**. Nếu tính thiếu dấu câu là
"mất kỹ năng" thì guard sẽ khoá vĩnh viễn khiếm khuyết đó. Ngược lại, **lặp**
(`'Chào bạn. chào bạn'`) và **mất từ** là mất nội dung thật và **bị chặn**.

`complete_ok` yêu cầu `content_ok` trước: đo được trước khi sửa, câu rác `'Lời.'`
được tính là "hoàn chỉnh" chỉ vì có dấu chấm.

Dashboard (`guard_report`) và acceptance suite gọi **cùng** `_pair_verdict`, nên
không còn trường hợp dashboard báo PASS trong khi guard coi là fail.

**Đã kiểm chứng sau khi sửa:** retention violations **0/9 lượt**; `content_ok` so
chính xác nên câu chứa đáp án kèm nội dung thừa vẫn bị chặn (`'Hello world.'` vs
`'Hello.'` → `content_ok=False`). Ngưỡng `if metrics['guard_lost']: raise` giữ
nguyên, **không nới**.

Điểm loss trên đồ thị thuộc các giao dịch có thể khác bài, không là một
learning curve của dataset cố định. Counter tích lũy có thể khác số log giữ lại.

Lượt thật `7 + 8 → 15` có17 micro-updates được giữ,34 token update steps,
loss7,155→1,144 và lần thử cuối sinh15. Không có rollback mới trong lượt này;
93 rejection trước khi thêm instrumentation không được bịa loss/ΔW. Đây là
training-point acceptance, **không chứng minh tổng quát hóa hoặc retention**.

Chat/learning trace nằm trong SQLite data-dir, không nằm trong neural checkpoint
export. Poll đọc tối đa180 events, hiện tối đa60 answers/100 learning events;
answer/rejection events có cap200/800. Backup data-dir để giữ lịch sử cục bộ.
Không tự suy ra hoặc tái dựng loss của những log cũ.

Controller dừng khi cùng modelstate/prompt/target tạo2đềxuất bịguardtừchối
giốnghệt; trace ghi rõ stopped-identical-proposal. Đây là chống lặp vô ích,
không thay guard, ép học hoặc chứng minh model đã học đúng.

## Neuron theo task

`#capacity-lab` tự hiện lượt đo gần nhất do Codex thực hiện. Ba chế độ runner:

- **Lõi đang dùng:** chạy trên đúng core trong Hermes. Lưu checkpoint trước
  lượt thử, giữ các trạng thái và cấp phát mới sau khi chạy. Không học, thêm
  lesson hoặc tạo câu hỏi từ các task quan sát. Runner có thể dừng; dừng không tự
  xóa phần đã cấp. Muốn quay lại topology cũ thì khôi phục checkpoint trước lượt.
- **Bản sao phát hành:** bắt đầu từ public checkpoint 48 neuron, không đổi
  phiên trò chuyện. Số đang tính đã đo: 48 → 64 → 96 → 128 → 48.
- **Bản sao phiên đang dùng:** dùng cùng checkpoint hiện tại để thử riêng.

Lượt thực tế trên core người dùng đã đo 64 → 64 → 96 → 128 → 64, vì các
readout đã kết nối yêu cầu giữ ít nhất 64 neuron khi trả lời. Nền khi rảnh có
thể dùng 48. Không giả mọi phiên đều phải có đúng dãy này.

Bảng phân biệt **đang tính**, **đã cấp**, **ngủ** và **cấp thêm**. Sau task dài,
neuron đã cấp vẫn ở RAM; task ngắn chỉ giảm compute, không phải shrink/pruning
hoặc tự xóa kiến thức. Trace tự theo task mới nhất: xung/điện thế, prompt,
số tick, mảng neural và đầu ra chưa xác minh. Màu phản ánh spike thực tại mẫu đó.

Bộ chọn hiện dựa trên lượng đầu vào và loại dữ liệu, cùng mức bảo vệ readout.
Nó **chưa tự hiểu độ khó**, chưa có ACT tự quyết định số bước nghĩ và chưa chứng
minh rằng cấp thêm neuron làm task đúng hơn. Không dùng số neuron làm điểm IQ.
Khoảng chờ 0,9 giây để nhìn chuyển đổi được tách khỏi thời gian xử lý. Với core
live, thời gian cả lượt gồm cả ghi trạng thái/checkpoint, không chỉ kernel.

Kết quả quan sát gần nhất được lưu trong `local/generation1/capacity-observation.json`.
Sau restart trang ghi rõ đây là lịch sử đã lưu, không giả job vẫn đang chạy.
Log quan sát không nằm trong neural checkpoint export; backup cả data-dir nếu
cần giữ log. Không sửa trọng số để khớp một bản demo đẹp hơn.

## Codex đang thử gì?

Audit tínhiệu synapse chỉđọc (báo cáo28) hiện bảng cosinehaihalf và tỷ lệ
phương sai; đây không phải loss/accuracy hoặc learningupdate. Giátrịâm/missing
được giữ, surrogate chỉbiasedcontrol. Crossfitbaseline dùng nửađối diện nên
haihalfmeans phụthuộc; không coi như kiểm lặp độc lập hoặc qualitygain. Không
tạo answer/learningevent giả cho một audit không học hay sinh câu.

Mở **http://127.0.0.1:8765/#research-lab** để xem lượt chạy mới, giai đoạn, số
trial, RAM tiến trình và trace được lượt đó công bố. Tự theo lượt đang chạy
hoặc lượt mới nhất; không cần chọn hoặc khởi chạy. Không tự bắt đầu job khác.

- Trạng thái **Đang chạy** được kiểm tra bằng PID và thời điểm tạo process.
  PID mất hoặc bị tái sử dụng sẽ thành “Đã dừng / thiếu kết quả”. Không restart
  job chỉ vì file không được cập nhật hoặc mất kết nối một lần.
- **Đã có kết quả** không phải Pass. Kết luận GO/PIVOT/NO-GO và screen flag được
  giữ riêng; phần lịch sử dẫn tới báo cáo nguồn, kể cả thử nghiệm thất bại.
- Trace là mẫu thật gần nhất được tác vụ công bố. Job chưa có trace sẽ ghi
  “chưa công bố”; không tạo spike hoặc độc thoại nội tâm thay thế.
- Các bài P1 khóa 48 neuron theo protocol sẽ hiển thị 48. Đây là kiểm soát
  thí nghiệm, khác tính năng cấp phát linh động của sản phẩm.

Mọi script dùng `evaluation.runs.new_run` được đăng ký trên bảng. `write_json`
cập nhật kết quả cuối và một số progress records. Runner muốn có trace/tiến độ
chi tiết gọi `runtime.experiment_status.progress` từ process điều phối, đưa số
đo thật vào. Workers trả dữ liệu cho process điều phối, không cùng ghi journal.
Lượt tạo batch Antigravity cũng được theo dõi, với nhãn **chưa học vào neuron**.
Không chạy lại job hoặc gọi CLI chỉ để làm bảng trông đang bận.

Runner muốn công bố câu trả lời/loss riêng gọi `runtime.work_trace.append(run,
'answer'/'learning', measured_data)` ngay sau phép đo thật; log cross-process
ở `local/work-trace.sqlite`, tối đa600 records×32KB, API đọc200. Không tự bịa
loss của runner chưa instrument. `scripts/observe_live_work.py` gọi chính
serving API và existing learning channel, không dùng model stub hoặc proxy.

`activity(run,title,detail,status=...,evidence=...)` dùng cùng log hữu hạn,
API hiển thị tối đa24activity. Technicalactivity không đi vào tập answer/loss
hoặc neural checkpoint. Runner `scripts/verify_passive_work_trace.py` thực hiện
kiểm tra phần mềm và một inference thật, không tạo lesson.

Journal mới ở `local/experiment-status/`, mỗi file có trần 192 KB và tối đa
40 sự kiện; API đọc tối đa 16 journal gần nhất. Dữ liệu huấn luyện, test outputs,
source snapshot và báo cáo gốc vẫn ở các `runs/<run-id>/`, không bị journal ghi đè.
Các lượt cũ trước khi có journal chỉ hiện kết quả đã lưu; không bịa trace cũ.

## Lõi đang trò chuyện

Khu vực này vẫn là cùng model ID `scarx-generation1`, cùng weights/state và
vòng idle review đã có. Trace của thí nghiệm không được nhập vào model này.
Số hoạt động không chứng minh model hiểu ngôn ngữ, ảnh hoặc đã học được kỹ năng;
xem [STATUS](STATUS.md) và master [RESEARCH_PLAN](RESEARCH_PLAN.md) để biết gate.
