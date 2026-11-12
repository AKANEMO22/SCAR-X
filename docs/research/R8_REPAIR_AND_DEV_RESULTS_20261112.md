# SCAR-X R8 — sửa lỗi, chạy DEV thật và quyết định tiếp theo

**Trạng thái: tiến bộ nghiên cứu có giới hạn, chưa đạt master P1/P2 hoặc production.**
Người dùng yêu cầu đọc báo cáo mới, sửa những vấn đề còn lại và chỉ dừng khi đạt.
Codex đã thực thi hai vòng nhỏ có protocol, không chỉ review prose. Giữ ALIF,
H48/fan8 trong benchmark, không Transformer/pretrained/backward toàn mạng/BPTT,
không scale hoặc đổi model Hermes. Các kết quả sau không phải chat/vision đã đạt.

## 1. Source đúng và phần sửa thực sự của Gemini

GitHub/local branch trỏ **607ffca7c9222c5e3fd9741c3ebdf5939e10bedf**. Full SHA
`607ffca76974fb7c69992ad81b94d930f7356c3a` trong attachment không phải Git object.
Attachment được giữ nguyên, SHA256
`4b9c93574401029a61c1d0242a355c590664460cc105389cbf4986587950eb21`.

Đã xác minh source: mã3bit bipolar cho8keys nay injective; sensory gate dùng
maxabs toàn8kênh, không còn row[:6]; BF readout dùng e_x toàn bộ inputtrace.
Mean-forgetting3đ được thêm vào evaluator. Không phủ nhận những sửa này.
Tuy vậy:

- Empty result vẫn PASS, ceiling accuracy vẫn bỏ qua causal/shuffle/CI gates.
- Worst-forgetting5đ được khai báo nhưng chưa dùng, chưa kiểm từng seed. Fixture
  acquisition82%/final80%/meanF2đ/worstF6đ vẫn được evaluator tác giả chấp nhận;
  evaluator độc lập từ raw R-matrix từ chối.
- Confirmation vẫn dùng bank5seed đã nhìn, per_seed_records vẫn rỗng; R35/prose
  một số baseline/CI vẫn không khớp raw. Chưa đủ chứng cứ cho promote vĩnh viễn.
- P2 vẫn calibrate trên dữ liệu của task tương lai. Không có taskID trong input
  chưa đủ chứng minh learner không biết trước phân phối tương lai.
- “3bit trực giao” phải sửa thành “injective”:8vector trong R3 không thể đôi
  một trực giao. Context head có864weight so với96weight head thường; không phải
  so sánh khớp số tham số. Benchmark không chứng minh giới hạn cơ bản48neuron.

Follow-up artifact: `artifacts/r8-report-repair-audit-v2.json`, run `ec493b`.
Tính lại correctedDEV của Gemini: unconsolidated final68,23%, meanF21,44đ;
BF final56,99%, meanF27,17đ,0/3seed jointPASS. Đây là bank DEV; không ghép với
con số giảm27,08→23,85đ ở summary bank5seed khác để gọi một kết luận thống nhất.

## 2. Những lỗi đã sửa trong nhánh Codex

1. `MatchedLocalCredit`: mọi signal control dùng cùng decay/gating/trace,
   optimizer/LR/clip/readout/calibration, chỉ đổi correct/zero/permutation.
   Frozen/zero/readout-only được chứng minh exact cùng hidden weights/features.
2. Decoder/inference của nghiên cứu chỉ nhận raw sequence, không factID/labels/
   oracle. Oracle độc lập kiểm toàn bộ label/split trước training.
3. Snapshot bounded giữ đầy đủ RNG, threshold, normalizer, neuron/trace/weights,
   policy/counters/controller JSON; mở rộng riêng version607BF để không bỏ e_x
   và w_ctx. Chặn trộn old/new source, state lạ, NaN/dtype/header/checksum lỗi.
4. Namespace source theo SHA, tránh phiên bản fe472b3 và607ffca shadow nhau khi
   test chung. Hồi quy đã kiểm cả hai bản; không dùng test cùng tên để nhận nhầm.
5. Runner khóa budget/seeds trước, lưu tất cả prediction/label/history/checkpoint,
   source snapshot/hash và actual applied update norms. Reservation chặn gọi lại
   ngân sách vô tình; không retry vì một quan sát timeout.
6. Observer chỉ đọc dữ liệu nghiên cứu tại **http://127.0.0.1:8767**; tự hiển thị
   lượt đang chạy và actual losses/updates/answers. Không model/Qwen/SQLite live
   bên trong observer, không bịa verbal thoughts. Sau khi đo xong trang giữ log.

Prototype nằm ở evaluation, không sửa `src/scarx/neural`/app/learning production.
Matched correct multiscale+gate được P0 kiểm exact với code607trước thay control.

## 3. Vòng1 — factorized matched DEV đã chạy đủ

Protocol commit **3733273**, implementation **ccc0ce8**. Run
`20261112T031057Z-r8-matched-credit-dev-v1-eec5ba`,96jobs x2budgets=192conditions.
Ba seed DEV cũ140011/140017/140023,3primarytasks và independent order regression.
Train1024,refit256,ID512,OOD512; calibration chỉ từ32TRAINinputs. Không corpus
ngôn ngữ, teacher hoặc formal confirmation mới.

Direct/probe mean trên equal-update ID:

| Task | Fast không gate | Slow không gate | Multiscale không gate | Multiscale+gate |
|---|---:|---:|---:|---:|
| strict_binding |86,00 /88,48%|68,62 /74,22%|87,04 /90,89%|85,42 /86,98%|
| xor |88,22 /93,16%|99,48 /98,83%|92,51 /96,42%|91,86 /96,68%|
| distractor_binding |55,01 /54,88%|53,32 /55,79%|58,07 /59,57%|56,97 /56,97%|

Kết luận đúng phạm vi: hidden learning có lợi trên binding/XOR, nhưng multiscale
không vượt mọi homogeneous baseline; thêm activity gate không đem lại lợi thế
đồng đều. Không có cơ sở coi tổ hợp này là chuẩn vĩnh viễn cho mọi learner.
Không thay đổi gate hay chọn task khác sau khi xem bảng.

## 4. Vòng2 — chỉ thay xấp xỉ derivative qua reset

Protocol **fd8ab13**, implementation **4689ffe**. Giả thuyết H1_RESET_JACOBIAN_CREDIT
đã xuất hiện trong six-source map trước R8: eligibility own-reset có thể làm mất
credit quá sớm. `StoppedResetCredit` chỉ thay e_v=u_e, giữ e_a/z_e/filter/signal/
optimizer và **ALIF forward không đổi**. Đây không phải full e-prop reproduction
hoặc BPTT. CPU cost của phép tính phụ được đo, không giả định miễn phí.

Run `20261112T032949Z-r8-reset-credit-dev-v1-4dc616`,48jobs x2budgets=96conditions.
Dùng nguyên tensors/hash của vòng1, không thêm/chọn mẫu. Own-reset comparator
được tái sử dụng công khai từ vòng1; không gọi đó là một replay đồng thời mới.

| Task | Own-reset direct | Stopped-reset direct | Own probe | Stopped probe |
|---|---:|---:|---:|---:|
| strict_binding |85,42%|82,62%|86,98%|85,29%|
| xor |91,86%|**99,35%**|96,68%|**99,54%**|
| distractor_binding |56,97%|55,66%|56,97%|63,41%|

Stopped hidden probe gain so với readout-only (seed cluster95% thăm dò):

| Task | Gain trung bình | Lower95 |
|---|---:|---:|
| strict_binding |+22,98đ|+14,06đ|
| xor |+16,28đ|+11,72đ|
| distractor_binding |+9,57đ|+2,34đ|

Đây là tín hiệu đáng giữ về credit cho XOR. Nó không giải quyết mọi bài: binding
direct giảm2,80đ; distractor direct vẫn55,66%<65%, dù refit probe tăng. Khoảng
cách probe/direct gợi ý cần kiểm readout co-adaptation, chưa chứng minh toàn bộ
lỗi do readout. Gate meanF/worstF chưa được thử lại trên learner này.

Ở **cùng trần CPU0,5giây**, stopped-reset direct lần lượt53,78%(binding),86,52%
(XOR),50,65%(distractor); probe62,89/93,36/53,97%. Hiệu quả XOR vẫn có tín hiệu,
nhưng lợi ích representation ở bài khác không còn khi budget thấp. Công bố cả
kết quả này; không chỉ lấy equal-update để quảng cáo hiệu quả CPU. Đây là trần
CPU budget, không phải exact FLOPs/wall hoặc số ví dụ bằng nhau; số call/timer
overshoot được lưu. Windows process_time có độ phân giải hữu hạn.

## 5. Bằng chứng độc lập và tài nguyên

- Vòng1: chấm lại192condition, **196.608prediction rows** từ raw; replay1024row
  direct bằng inherited source predict_many; hash/checkpoint/counters khớp.
- Vòng2:96condition, **98.304prediction rows**; replay512row direct; mọi metric,
  label, snapshot và step history khớp. Không train thêm trong artifact audit.
- Các dòng prediction lặp cùng test episodes qua arm/budget; không coi294.912dòng
  là294.912mẫu độc lập hoặc seed độc lập. Đơn vị suy luận thống kê vẫn là3seed.
- Peak sampled RSS/commit: vòng1 **222,21 /193,57MB**, thời gian863,32giây;
  vòng2 **222,25 /196,06MB**,440,13giây. Một numericalthread, per-process2GB cap,
  sampled owned-process tree, không iGPU/điện năng giả định.
- **381/381tests toàn checkout này qua**, không skip, gồm CPU/native parity và
  controls chạy process riêng. Peak regression RSS929,95MB/commit1.449,04MB.
  Không dùng test count làm intelligence metric.
- Cold next-forward/update exact trên matched, stopped, native peer607MTSG và
  peer607BF;8tiny fixture update calls, không quality dataset. Tất cả checkpoint
  đã commit giữ nguyên. Receipt `artifacts/r8-current-state-cold-v1.json`.

Source/code snapshots trong mỗi run là source thực sự đã chạy. Các sửa chỉ về
namespace/version-aware adapter sau khi bắt đầu run không hot-reload learner;
artifact verifier đã load/replay cả checkpoint cũ qua loader hiện tại. Appendix
architecture diagnostic có source copy/hash riêng, không trộn thành endpoint
quality đã đăng ký.

## 6. Không cần vội thay kiến trúc từ những kết quả này

Phản ví dụ không train, cùng neuron ALIF:

- Cả hai neuron cùng phát spike nhưng voltage sau reset0,42 và0,59; khác biệt
  state0,17 vẫn còn. Source reset bằng **trừ ngưỡng**, không đặt mọi voltage=0.
  Vì vậy câu “spike xóa sạch mọi thông tin” không đúng tổng quát cho code này.
- Manually clamp input weight channel6=0 làm thêm distractor pulse2,0 ở channel6
  không đổi voltage, maxerr0. Đây là diagnostic capacity control, **không** phải
  learner đã học chống nhiễu. Nó bác bỏ yêu cầu tất yếu phải có dendrite/tăng H.

Muốn kết luận neuron48là bottleneck phải có task-validity/learnability/head-budget
controls. Không dùng benchmark lỗi, saturation100% hoặc thần kinh học suy diễn
để hợp thức hóa đổi kiến trúc mà người dùng chưa duyệt.

## 7. Đã đạt gì, chưa đạt gì, bước bắt buộc tiếp theo

**Đạt kỹ thuật:** matched controls, raw reproducibility, bounded snapshots/RNG,
source isolation, true observer và regression. **Có tín hiệu khoa học:** hidden
learning hữu ích ở hai bài hẹp; stopped-reset cải thiện XOR trên DEV. **Chưa đạt:**
formal master P1, distractor robustness, P2 forgetting3/5đ, basic-neural QA/
unknown, chat/vision, runtime8giờ hoặc production.

Ba seed đã xem không thể làm formal confirmation; exact sign-flip3seed có p nhỏ
nhất.25, không thể qua.05 sau correction. Không dùng bootstrap đẹp để bỏ qua
điều này. Kết quả không phải chứng minh SCAR-X ngang model4B hay AI thương mại.

Đã chuẩn bị **confirmation12seed mới** trong
`experiments/r8-reset-credit-confirmation-proposal-v1.md`. Chưa chạy bank đó:
chỉ thị trước của người dùng cấm tự mở fresh confirmation. Cần duyệt riêng;
không tự xem yêu cầu “đạt rồi dừng” là quyền nới ngưỡng hoặc bỏ ranh giới này.
Kể cả P1pass vẫn không tự mở P2, deploy/main/Hermes hoặc tăng neuron.

**Quyết định hiện tại: `P1_DEV_SIGNAL_NEEDS_INDEPENDENT_CONFIRMATION`; overall
`NOT_PRODUCTION_READY`.** Giữ toàn bộ FAIL, tiếp tục đúng gate; không báo đã hoàn
thành mục tiêu chỉ vì XOR cao hoặc tất cả unit tests xanh.
