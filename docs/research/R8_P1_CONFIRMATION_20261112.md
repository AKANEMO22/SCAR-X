# P1 confirmation — 12 seed mới, đã được người dùng duyệt

**KẾT QUẢ: `P1_FAIL_RESEARCH_ONLY`.** Đã chạy đủ204jobs/408conditions và qua
kiểm toán độc lập. Chỉ XOR đạt toàn bộ tiêu chí; cần ít nhất2/3bài primary.
Không đổi ngưỡng hoặc tune rồi chấm lại bank đã mở.

## Phạm vi và quyền thực hiện

Người dùng trả lời **“Duyệt confirmation 12 seed”** cho yêu cầu
`call_c9e5cfcca5aa4a7cac0e937a8b852134`. Approval record:
`experiments/r8-confirmation-approval-20261112.json`.
Proposal đã duyệt giữ nguyên tại
`experiments/r8-reset-credit-confirmation-proposal-v1.md`; execution registration
`experiments/r8-reset-credit-confirmation-v1.md` chỉ làm rõ cách thực hiện.

- Run: `20261112T060357Z-r8-reset-credit-confirmation-v1-84547e`.
- Source commit khóa trước mở bank: `bc6865003cc63dd846b4321d5050a0efbe27ef24`.
- Peer source: `607ffca7c9222c5e3fd9741c3ebdf5939e10bedf`, checkout detached riêng.
-12seed2026111201+6*k, k=0..11, không dùng bankDEV140011/17/23 để xác nhận.
-204jobs,408conditions, ba bài primary và một bài order bảo vệ;1024updates
  primary và trần0,5giâyCPU secondary. Không thêm/tune phương pháp sau khi xem bank.
- CPU một numericalthread, toàn cây process owned+observer đo theo PID không
  đếm trùng, per-process2GBcap và watchdog60phút;120giây/job.
- Không P2, không tăng H48/fan8, không đổi forward ALIF, không Transformer/
  pretrained/backward toàn mạng/BPTT, không main/Hermes hoặc dữ liệu live.

## Kiểm chứng trước mở bank

51test kỹ thuật qua. Kiểm đủ48task/seed datasets với independent event-label
oracle và split-overlap/hash trước khi train cell đầu tiên. Protocol, source và
approval được lưu trong `approval-and-lock.json`; source snapshot, peer-source
copies và exact worker command/PID đi cùng run.

Supervisor và worker có handle/process thực; một lock directory không được xem
là bằng chứng tiến trình còn sống. Ngân sách này chỉ chạy một lần; lỗi/timeout
giữ partial evidence, không tự mở lại bank hoặc thay seed.

## Endpoints và cách kết luận đã khóa

Seed là cluster độc lập. Bootstrap10000lần/RNG919,95%percentileCI; exact4096sign
vectors, Holm7primarytests. P1đòi ít nhất2/3bài có lower95representationgain>=5đ,
direct>=65%, readout/shuffle contrasts và các bảo vệ NI theo proposal; không có
ngoại lệ100% hoặc thiếu-dữ-liệu-PASS. P2cửa65%/65%/3đ/5đ vẫn khóa riêng.

Đo riêng “master P1 numeric gates” và “reset-credit thực sự thắng own-reset trên
XOR”; hai kết luận không được đánh đồng. Mọi kết luận chỉ áp dụng bộ bài event
đã công bố, không chứng minh khả năng ngôn ngữ, vision hay AIđa năng.

Trước công bố: chấm lại mọi prediction/label/counter/hash, replay inherited
forward trên subset, kiểm zero/readout feature invariance, full-revert recovery,
RNG schedule. Một implementation scalar độc lập kiểm lại bootstrap, quantile,
permutation, Holm và gate arithmetic. Thiếu cell hoặc audit lỗi chặn PASS.

## Quan sát

**http://127.0.0.1:8767** tự nhận run mới; chỉ đọc progress/loss/update norms và
actual neural predictions so với oracle. Observer không nạp model hoặc SQLite
live. Các điểm trên trang chưa phải quyết định cuối cùng.

## Kết quả

| Task | Stopped direct | Own-reset direct | Stopped refit probe | Kết luận theo gate đã khóa |
|---|---:|---:|---:|---|
| strict_binding |76,69%|82,55%|82,19%|FAIL bảo vệ OOD |
| xor |99,28%|97,18%|99,71%|PASS trong phạm vi bộ đo |
| distractor_binding |63,64%|66,28%|66,68%|FAIL direct65% và bảo vệ OOD |
| order protected |99,98%|99,85%|99,82%|Qua NI; không tính ceiling là representation success |

### Điều mới đã được xác minh về hidden synapses

Trên cả ba bài primary, learned hidden có common-refit gain và routing contrast
dương so với matched readout-only/shuffled, sau Holm7tests:

| Task | Hidden probe gain | CI95% theo12seed | Holm p |
|---|---:|---:|---:|
| strict_binding |+17,97đ|[+14,06;+21,65]đ|0,003418|
| xor |+14,16đ|[+9,85;+18,67]đ|0,003418|
| distractor_binding |+13,88đ|[+10,95;+16,89]đ|0,003418|

Đây là bằng chứng **có học representation hữu ích** trên các event tasks đã
công bố; không nên tiếp tục nói hidden hoàn toàn không học. Nhưng hiệu quả không
đồng nghĩa candidate cải thiện toàn bộ kỹ năng, không quên, hoặc đủ để deploy.
Các task được train riêng, không phải một checkpoint đã tích lũy cả ba skill.
Generalization được kiểm chủ yếu qua noise/timing của các task hữu hạn; không
suy rộng thành ngôn ngữ, kiến thức đa năng hoặc novel concepts ngoài bộ đo.

### Vì sao candidate vẫn FAIL

- Binding có representation gain rõ nhưng kém own-reset ngoài phân phối: OOD
  direct difference−5,58đ, CI95%[−9,44;−1,69]đ; OOD probe difference−3,50đ,
  CI95%[−9,46;+2,39]đ. Không đạt lower95>=−3đ đã duyệt.
- Distractor direct63,64% dưới65%; OOD direct/probe NI lower bounds−4,10/−4,02đ,
  cũng không đạt. Không hạ ngưỡng vì thiếu khoảng1,36đ.
- XOR là thành quả riêng đáng giữ: probe tăng**+1,81đ** so own-reset,
  CI95%[+0,80;+3,30]đ, Holm p0,003418; NI OOD và order protection đều qua.
  Cửa XOR superiority được xác nhận, còn cửa tổng P1 không qua.

Chúng ta có một tradeoff credit-path, không có bằng chứng một công thức reset
duy nhất đã giải mọi task. Own-reset đối chứng còn đạt66,28%distractor ID ở cùng
bộ đo; vì vậy không thể quy mọi thất bại cho định luật “ALIF một ngăn không thể”.

### Chất lượng phép đo và giới hạn nguồn lực

- Chạy đủ12seed,204jobs/408conditions, không replace/loại seed xấu; error=null.
- Chấm lại độc lập**417.792prediction rows**,544direct rows được replay qua
  inherited source prediction; kiểm dataset/checkpoint/counter/RNG/invariance.
  Những dòng này lặp mẫu qua các control/budget, không phải417.792seed độc lập.
-50contrasts được crosscheck bằng scalar bootstrap/quantile/permutation/Holm
  và gate arithmetic độc lập, kết quả khớp. Integrity audit PASS.
-51preflighttests qua trước mở bank. Source hashes/peer source không đổi trong
  run; main giữe94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541. Không ghi live data.
- Wall **1710,25giây (~28phút30giây)**, dưới60phút. Peak owned-tree+observer RSS
  **258,04MB**, private/commit **244,36MB**, không quan sát vượt2GB. Sampling không
  phải hard cap tổng vật lý; per-process OS cap được ghi riêng. Không đo điện năng.
- Equal-CPU0,5giây và OOD đều được giữ trong raw/summary; không bỏ kết quả bất lợi
  để thay phán quyết. R9review dùng worktree forensic riêng, bank191xxx không
  trùng bank12seed này và không được dùng để đổi model/tiêu chí đang xác nhận.

Receipt cuối: `artifacts/r8-reset-credit-confirmation-v1.json`; run chứa approval,
source.zip, peer-source, datasets/hashes, initial/final checkpoints, learning
histories, all predictions/interventions và independent audit results.

### Quyết định tiếp theo

**Giữ kết quả XOR và bằng chứng hidden learning, không promote stopped-reset làm
chuẩn chung. PIVOT cơ chế thích nghi tín hiệu/credit, không tăng neuron hoặc đổi
kiến trúc từ kết quả này.** Không mở P2, dataset ngôn ngữ, fresh bank khác hoặc
deploy trong phạm vi approval này. Bank12seed đã bị xem, mọi corrected replay
phải gắn nhãn đúng; không gọi lần chạy lại là fresh confirmation.

P2forgetting65%/65%/3đ/5đ, chat/vision/unknown và product acceptance vẫn chưa đạt.
`P1_FAIL` này không phải `NO-GO` cho toàn bộ SNN; nó bác bỏ việc dùng candidate
stopped-reset hiện tại như giải pháp chung. Overall **NOT_PRODUCTION_READY**.
