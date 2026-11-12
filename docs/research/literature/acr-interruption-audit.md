# ACR — kiểm toán lượt bị gián đoạn, 12/11/2026

**Kết luận: `INCOMPLETE_NO_ADVANCE`.** Có55cell hoàn tất, hợp lệ về các kiểm
tra raw dưới đây; chưa đủ dữ liệu chọn cơ chế thắng hoặc mở state router.
Không chạy lại training, không resume, không sửa source/raw của ACR trong
lần kiểm này. Ponytail không có trong tool inventory.

## Nguồn chứng cứ và tình trạng tiến trình

Research worktree ACR:
`scar-x/local/worktrees/scarx-acr-20261112`, branch
`codex/scarx-adaptive-credit-routing-20261112`.
Run `20261112T073944Z-acr-dev-v1-f947fb`, source đã khóa
`950718bd410371f8ef15dab2d78d722b04ab506d`, base
`93b5b96bebaf22dbc39a3f96b2cac7c917d5e377`.

Đã đọc manifest, progress,15dataset hashes, toàn bộ55result/checkpoint/history.
Tại lúc kiểm, Python workerPID35452 và observerPID24452 trong receipt không
còn xuất hiện trong inventory tiến trình Python. Các tiến trình nghiên cứu
khác và Hermes vẫn tồn tại, không bị thao tác. `progress.json` còn chữ
`TRAINING` là trạng thái ghi cuối, **không phải bằng chứng worker đang chạy**.
Nguyên nhân credit interruption được parent ghi nhận từ session; không có
exception/final receipt trên đĩa để tự suy ra nguyên nhân dừng từ artifact.

Main vẫn là `e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541`, bằng manifest.
Không chạm main, live model, Hermes, SQLite hoặc port8767.

## Số lượng cell chính xác

Trên đĩa có55job directories và55`result-0.json`, đúng progress55; không phải56.
Mỗi directory có đủ đúng4artifact: `initial-0.zip`, `learning-0.npz`,
`model-0.zip`, `result-0.json`. Không thấy job có checkpoint đã train nhưng
thiếu result, không có directory dở dang hoặc file ngoài schema này.

| Profile đã đăng ký | Cell hoàn tất | Cell AB còn thiếu |
|---|---:|---:|
| static_0 |15|0|
| static_25 |15|0|
| static_50 |15|0|
| static_75 |10|5|
| static_100 |0|15|
| linear_up |0|15|
| linear_down |0|15|
| groups_up |0|15|
| groups_down |0|15|
| intrinsic |0|15|
| readout_only |0|15|
| **Tổng AB** |**55/165**|**110**|

Mỗi profile đủ dự kiến5environment×3seed; joint là một checkpoint/head pooled
riêng, không được đánh đồng với4isolated task checkpoints. static75 đã hoàn
tất3seed cho binding/XOR/distractor và order140011. Năm cell kế tiếp trong
thứ tự khóa: order140017, order140023, joint140011, joint140017, joint140023.

Không có `stage-ab.json`, `nominee-lock.json` hoặc `result.json` cuối run.
State-adaptive C, shuffled-state, random-alpha, nominee shuffle, actual-norm
paired controls, equalCPU secondary và cơ chế/OOD diagnostic stage **chưa chạy**.
Alpha100 chưa có quality outcome trong lượt này; technical endpoint tests
trước DEV không thay thế outcome đó. Không được gọi static.5 hoặc static.75
là candidate thắng từ một study đang thiếu cả baseline stop/readout.

## Kiểm tra độc lập raw đã thực hiện

-15/15source hashes và2/2peer source hashes khớp manifest950718bd.
-15/15dataset ZIP hashes khớp `dataset-hashes.json`.
-55cell keys duy nhất, đúng các DEV seeds140011,140017,140023; không confirmation.
-Chấm lại **419.840 prediction entries** bằng so sánh nhãn nhị phân từ frozen
  dataset: ID/OOD direct/probe và cả3hidden-weight interventions. Mọi metric
  lưu khớp số đếm độc lập, không chỉ tin phần summary.
-55final checkpoint và55initial checkpoint khớp hash receipt. Tất cả55final
  checkpoint load rồi dump lại khớp byte; signal RNG và router RNG khớp metadata.
-Mỗi cell có history1024×4 finite, step1..1024; episode/hidden/core update
  counters đều1024. Tổng **56.320actual learning updates** đã hoàn tất.
-ACR extra array bytes đều16.704B, khớp thiết kế; không dùng số này làm processRAM.

Digest tập55result/model/history hashes, sắp theo foldername:
`42822bf9c5b5a3d9b84ed99997a09f4044feb93aa78ecef3a474f3a8f6fc7ce6`.
Đây là SHA256 của JSONsorted list gồm folder/result_sha256/model_sha256/
history_sha256; original artifacts được giữ nguyên tại worktreeACR.

Không dùng các accuracy này để chọn hyperparameter, không thay criterion,
không gọi accuracy cao trên order là bằng chứng representation.

## Đối chiếu endpoint với reference DEV đang chạy

Reference run ở worktree literature riêng:
`20261112T080533Z-literature-reference-dev-v1-839e07`.
Tại lúc kiểm, đủ12`own` fixed1024update isolated cells đã hoàn tất.
Đã đối chiếu **cả12task×seed cell** (4task×3seed) với ACRstatic0:

-ID/OOD direct/probe predictions: **exact**.
-Ba hidden intervention prediction sets và mọi metric: **exact**.
-Final core fingerprint: **exact**.
-Toàn bộ1024loss values theo thứ tự: **exact**.

Checkpoint wrapper ACR và originalown khác schema/extra state nên không đòi
ZIP bytes giữa hai class trùng nhau. Việc trùng core và raw outputs xác minh
endpoint outcome hữu ích mà không train lại ACR. Joint không có trong reference
DEV này, nên không bịa thêm3joint comparisons. Chưa đối chiếu stop endpoint
outcome vì ACRstatic100 chưa chạy. Đây là kiểm reproducibility, không là
phép so CPU efficiency: ACRstatic0 vẫn duy trì cả hai eligibility streams.

## Tài nguyên lịch sử và cách đọc đúng

Tổng training CPU từ55history là **305,375giây**; từng cell2,671875–7,1875giây.
Đây chỉ CPU đoạn `learn_episode`, không gồm evaluation/source hash/I/O.
Progress resource snapshot cuối ghi elapsed **1.132,998488giây** (~18phút53),
peakRSS **135.348.224B**, peakcommitted **178.630.656B**, không measurement error,
không observed over2GB. Resource scope có worker và adopted observer, các
git subprocess con được quan sát trong lúc sống. `observed_owned_pids` là
tập PID từng thấy qua thời gian, không phải tất cả cùng chạy hoặc neuron.

Các số peak/elapsed này là **snapshot đã ghi trước interruption**, không có
final resource receipt nên không gọi đó là peak chắc chắn của toàn lifetime.
`hard_cap_enforced=false` trong monitor; run chưa ghi final `cap` receipt,
do đó lần audit này chỉ xác nhận observed monitored RAM, không tự khẳng định
hard cap đã được OS enforce. Commit là capacity đã commit, không phải lượng
swap đọc/ghi; không có bằng chứng pagefile weight streaming.

## Điều kiện nếu parent quyết định tiếp tục

Chỉ thị literature-first mới yêu cầu đối chiếu published references trước
khi mở rộng ACR. Còn110ABcells, sau đó mới xét C theo protocol và directive;
không được bỏ qua controls để tiết kiệm rồi tuyên bố ACR_ADVANCE.

Nếu tiếp tục, cần continuation receipt riêng: dẫn tới run/source/seed lock
cũ, hash toàn bộ55complete cells và skip chúng nguyên vẹn. Hiện không có
mid-job checkpoint cần resume. Không overwrite originalraw/progress để che
interruption, không tái chạy55cell rồi lấy điểm tốt hơn.

Budget9000giây ban đầu không được reset âm thầm. Snapshot đã ghi tiêu thụ
ít nhất1.132,998488giây; còn **không quá7.867,001512giây** active budget nếu
coi thời gian offline không tính. Cần nêu rõ đây là upper bound do thiếu final
receipt, dành overhead kiểm/chuyển tiếp, theo dõi cumulative active time và
giữ120giây/job/2GB trước khi thực thi. Nếu không đủ budget hợp lệ, công bố
partial study và giới hạn; không giảm seed/sample rồi gọi confirmation.

Không có seed mới hoặc permission fresh confirmation/P2 trong audit này.
Quyết định tiếp tục ACR chỉ có thể dựa vào comparison tham chiếu và protocol
được giữ nguyên; kết luận hiện tại vẫn **chưa có quality winner**.
