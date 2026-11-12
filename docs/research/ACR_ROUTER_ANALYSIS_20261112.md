# ACR state router — diagnostic không cập nhật trọng số

## Phạm vi khóa trước khi tính

Ngày12/11/2026. Chỉ postprocess nghiên cứu đã được giao, không mở candidate,
không chọn/tune profile hoặc xem nhãn test. Ponytail không có trong inventory.
Chỉ chạy khi đủ60Ccell đã đăng ký:4profiles ×5environments ×3DEVseeds.

Phân tích state, state_shuffle, random_alpha, state_signal_shuffle trên bốn
task strict_binding/xor/distractor_binding/order và ba seed140011/140017/140023.
Mỗi final equal_updates checkpoint dùng đúng bốn TRAINindices0,63,511,1023;
48checkpoint/192rollout. Không dùng joint trong bảng correlation này; joint
vẫn thuộc guard60Ccell và được chấm trong báo cáo chất lượng riêng.

Chỉ clone rồi reset/forward/trace; tuyệt đối không gọi learn_episode, không
thay weight, optimizer, stats, signal RNG, learning counters của clone; snapshot
original trước/sau phải khớp byte. Router RNG của clone tiến theo mỗi tick như
source; đây là trace state tiến, không phải học router. Không đọc train_y,
ID/OOD labels hoặc task ID vào candidate. Tên task chỉ dùng để nhóm kết quả
ngoài learner; candidate nhận từng vector8chiều.

Kiểm số học q và alpha từ trạng thái tại chính tick đó: cùng frozen state,
bias, featurevectors và permutation được sao RNG trước step. Random-alpha
giữ nguyên instantaneous marginal bằng hoán vị cùng vector alpha, không giả
định phân phối của hai model đã học riêng sẽ matched qua các trajectory.
Đồng thời tính alpha state-correct/state-shuffled/random-alpha giả định trên
CÙNG state để xác nhận hai controls thực sự khác nhau khi bias theo neuron.

Lưu alpha theo tick/neuron, spike/adaptation/q/distance, cosine own/stop filtered
trace, correlations, range/saturation và state-control distinctness. Lưu rawNPZ
và source/checkpoint/data SHA trong output mới router-analysis-v1.json dưới
run thực, từ chối ghi đè. Tối đa600s kể từ đầu main, một numericalthread,
process/tree RSS và committed dưới2.000.000.000B; Windows cap nếu khả dụng.
Không sửa main/Hermes/sourcelearner/port8767, không confirmation/P2.

## Giới hạn diễn giải đã khóa

Router v1 có hệ số cố định, chưa học router. Correlation không chứng minh
nguyên nhân hay task semantic reasoning. Learner cập nhật cuối episode nên
alpha ở các tick trước không trực tiếp tạo weight update; trace còn có history
nhưng không feedback từ mixed alpha. Những rollout này từ finalcheckpoint,
không phải tape các bước của training trajectory gốc. Không dùng diagnostic
này để cứu ACR hoặc nới gates. Kết quả chất lượng/direct/OOD/retention vẫn do
protocol và audit chính quyết định; analysis này có0learningupdates.

## Kết quả đã chạy

Run nguồn `20261112T121634Z-acr-dev-v1-resume-59ced4`; đã chờ đủ60Ccell rồi
mới đọc48checkpoint không-joint. **PASS192/192rollout,0learningupdates**.
Thời gian5,174s, trong600s. Source/checkpoint/data hashes khớp;48original
snapshots trước/sau khớp byte. Clone chỉ tiến neural/eligibility/router-RNG;
weight/readout/optimizer/statistics/signal-RNG/learning counters giữ nguyên.
Không đọc bất kỳ label nào để chạy trace diagnostic.

| Router | Task | Alpha trung bình | Temporal std trong từng neuron | Corr(alpha,spike) | Corr(alpha,adaptation) | Corr(alpha,q) |
|---|---|---:|---:|---:|---:|---:|
| state | Binding |0,28971|0,09316|0,42982|0,28175|0,13342|
| state | XOR |0,29887|0,09064|0,48473|0,39096|0,14180|
| state | Distractor |0,28216|0,09556|0,42819|0,20520|0,16239|
| state | Order |0,30636|0,11411|0,52517|0,36166|0,12035|
| state_shuffle | Binding |0,29769|0,12883|0,02501|−0,04692|−0,01859|
| state_shuffle | XOR |0,29878|0,13174|0,04489|0,03534|−0,00787|
| state_shuffle | Distractor |0,29561|0,13303|0,00789|−0,10507|−0,02432|
| state_shuffle | Order |0,31837|0,15731|0,04850|0,07106|0,02345|
| random_alpha | Binding |0,29914|0,23110|0,03871|0,01809|0,05468|
| random_alpha | XOR |0,28996|0,22556|0,04431|0,01504|0,02804|
| random_alpha | Distractor |0,29462|0,22615|0,04993|−0,00068|0,02051|
| random_alpha | Order |0,32197|0,23748|0,02034|0,06030|0,11456|
| state_signal_shuffle | Binding |0,27488|0,09763|0,58179|0,50157|0,26367|
| state_signal_shuffle | XOR |0,28110|0,09313|0,53992|0,46336|0,17484|
| state_signal_shuffle | Distractor |0,28004|0,10560|0,61418|0,52139|0,20025|
| state_signal_shuffle | Order |0,28002|0,09396|0,55955|0,49394|0,22896|

Mỗi hàng là trung bình12rollout (3seed×4TRAINindices), correlation tính trong
từng rollout rồi lấy trung bình, không phải12independentseed hoặc p-value.
Raw alpha per-tick/per-neuron lưu trongNPZ; các mảng48neuron/12tick trung bình
theo từng profile/task đi trong JSON `summary` để vẽ lại, không suy ra cấu trúc
“suy nghĩ bằng lời”. Alpha toàn bộ nằm trong[0,006693;0,970060].

### Những gì được xác minh về controls

- q, alpha và mixture được tính lại độc lập bằng trạng thái cùng tick/RNG sao
  trước step: khớp exact ở2.304ticks. Không dùng target/task/accuracy đầu vào.
- Random-alpha là permutation của chính vector state-correct tại cùng frozen
  state: sorted distribution khớp exact ở2.304/2.304ticks; thứ tự neuron thay
  đổi ở toàn bộ2.304ticks đã quan sát.
- Với neuron bias không đồng nhất, state-shuffle và random-alpha cho vector
  khác nhau ở2.304/2.304same-state comparisons. Vì vậy hai control không chỉ
  là đổi tên cùng phép toán trong implementation hiện tại.
- Correlation alpha–spike bị giảm mạnh khi shuffle state hoặc alpha. Điều này
  xác minh routing gắn với feature được cung cấp, không chứng minh routing
  đó cải thiện năng lực. Hệ số spike/adaptation vốn cố định dương trong formula.
- Signal-shuffled model vẫn có alpha–spike correlation lớn. Do đó chỉ thấy
  alpha “động” hoặc correlation lớn không đủ chứng minh hidden learning có ích.
- Own/stop filtered traces có cosine trung bình khoảng0,932–0,959 trong các
  nhóm rollout này. Đây là mô tả hình học tại finalweights của từng model;
  không phải matched-weight so sánh chất lượng hoặc bằng chứng dư thừa hoàn toàn.

Không kết luận marginal alpha của state và random-alpha matched qua các model
đã train riêng: chính weights/neural trajectory của chúng đã khác nhau. Chỉ
phép hoán vị instantaneous trên CÙNG frozen state mới được xác minh exact.

## Tài nguyên và tái lập

Peak measured treeRSS59.736.064B, commit78.356.480B; nativeprocess high-water
RSS60.764.160B, commit79.151.104B. Không measurementerrors hoặc overlimit.
Windows per-process cap receipt `enforced=true`; trường sampler
`hard_cap_enforced=false` riêng chưa nối receipt, không dùng nó để suy ra cap.
Không GPU/iGPU/điện năng hoặc model ngoài. Chỉ owned analysis process được đo,
không gộp thành chi phí worker huấn luyện đang chạy song song.

```powershell
python -X utf8 scripts/audit_acr_router_state.py --run runs/20261112T121634Z-acr-dev-v1-resume-59ced4
```

Runner từ chối ghi đè `router-analysis-v1.json` hoặc raw directory. Artifact
đăng ký trước compute ở `router-analysis-v1-raw/registration.json` chứa script
SHA, protocol text/hash lúc chưa bổ sung kết quả, parent manifest/source hashes.
48rawNPZ có checkpointSHA/rawSHA receipts trong `router-analysis-v1.json`.
`router-analysis-v1-raw/data-and-raw-integrity-receipts.json` chấm lại48rawhashes,
object-free array schema và12frozen dataset hashes, không gọi model thêm.
Không cần train để đọc lại số liệu; muốn thực thi lại phải dùng output nghiên
cứu riêng có ghi rõ, không xóa evidence/reservation hiện tại.

**Kết luận hẹp:** state router thật sự thay alpha theo trạng thái và controls
đã hoán vị đúng; router chưa học tham số và correlation không chứng minh
credit hữu ích, generalization hoặc retention. Chưa qualityGO từ diagnostic.
