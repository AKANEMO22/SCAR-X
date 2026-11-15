# SCOPE-ALIF P1b — temporal transfer: PIVOT, không vượt G2

Ngày 16/11/2026, Asia/Saigon. Phạm vi: nghiên cứu cô lập; không main, Hermes,
checkpoint live, push, merge hoặc deploy. Nhánh `research/scope-alif-p1b-20261116`.
P1 gốc tại `6e08a3616bc1ca908d8303f6321d759fc57804eb` giữ nguyên NO-GO.
Preflight mới khóa ở `0e2d20cc5e67e5fda6e5f1e32c464f0b4169df01`; protocol và
learner trước training khóa ở `a0f5ac4c1a146bd07a554c6ed32919dc8a0ed8fb`.

**Kết luận:** temporal teacher giúp học hidden có ích trong điều kiện dấu
readout thuận, nhưng không giải được điều kiện dấu bất lợi và chưa vượt đối
chứng teacher-tráo một cách đủ mạnh, nhất quán. Dừng cấu hình này; không mở
TEST, G3 restart hoặc G4 reuse. Đây là **LEARNER_NO_GO đối với yêu cầu toàn
bộ của pilot**, không phải chứng minh ALIF không có khả năng biểu diễn task.
Không có bằng chứng đạt continual learning hoặc giải phóng dung lượng P.

## 1. Kiểm toán P1 gốc: xác nhận NO-GO, không chạy lại 144 forward

[Auditor mới](../../scripts/research/audit_scope_p1_saved_v2.py) dùng CSV,
Decimal, đại số và archived source; không import model hoặc gọi forward.
[Kết quả](../../runs/20261115T174514Z-scope-p1b-v1-b30f0b/p1-original-audit.json)
đối chiếu raw, stdout, result và hashes, không sửa artifact gốc.

- 144 bản ghi = 12 input duy nhất × 3 topology seed × 4 vai trò. Không có
  learning update. Ba seed không là ba thử nghiệm học độc lập: useful weights
  là cấu tạo cố định và recurrent weights bằng 0.
- `p1` là xác suất lớp **1** của softmax hai lớp. Với y=0, đúng một giá trị
  `0.47242799401283264`; với y=1, `0.527571976184845`. Phương sai Decimal
  trong từng lớp bằng 0 trên các số đã lưu, không chỉ do làm tròn sáu chữ số.
- Teacher/witness dựng bằng tay đạt 36/36 từng vai trò; hai student ban đầu
  18/36. Initial student chỉ có một feature vector: RO ceiling 50% trên tập
  đã lưu. Bỏ đóng góp P khỏi saved feature/readout teacher cho 50%.
- Timing/amplitude có input weight khác 0; công thức membrane đối chiếu tape
  sai số tối đa 7,63e-8. Không thấy label channel/oracle codec trong input.
  Witness topology/bounds được kiểm từ source; không giả thành độc lập tải
  lại toàn bộ trọng số lịch sử khi artifact không chứa chúng.
- **Giới hạn của báo cáo cũ:** G6 được gán False bằng lập luận trong runner,
  rồi old audit assert lại; đó không phải một gate thần kinh được đo độc lập.
  Không thấy mâu thuẫn số học cần sửa raw. Thời gian/RSS khớp các log và mã đo
  đã lưu, nhưng không thể đo lại peak lịch sử một cách độc lập nếu không chạy
  lại; lượt này không chạy lại. P0 75%/100% do người dùng mô tả chưa tìm được
  raw gốc để xác minh; không dùng nó làm kết quả tái lập.

## 2. Terminal-logit-only tương đương label-derived target khi nào?

Nếu `T(x)=c_y`, baseline dùng bảng hai target `c_0,c_1` có cùng target vector
với teacher. Cố định input, W, normalizer, eligibility, head, optimizer state,
RNG, thứ tự update, masks, clipping và arithmetic thì:

\[
\delta_T=c_y-p_S=\delta_L,\quad
m_T=W_o^T\delta_T=m_L,\quad D_T=m_TE=D_L.
\]

Momentum, energy normalization, clipping và bounds là cùng hàm xác định của
cùng trạng thái/đầu vào, nên physical update bằng nhau. Quy nạp theo update
cho cùng toàn bộ trajectory. Đây là tương đương với **label-only dùng cùng
bảng soft target**, không nói mọi one-hot learner tự động giống soft targets.
P có thể cần cho teacher ra quyết định, nhưng terminal output này không mang
sample-specific teaching content ngoài bảng theo nhãn. Giữ PREFLIGHT_NO_GO.

## 3. Tín hiệu mới và preflight attribution

Chỉ một giả thuyết: filtered-spike trajectory của P, gắn đúng input theo thời
gian, giúp S học hidden hơn label-only và cùng trace bị tráo trong từng nhóm
`(label, readclock)`, kể cả khi readout ban đầu sai dấu.

Đã đối chiếu negative results và prior art trong
[capacity prior-art matrix](SCOPE_ALIF_CAPACITY_PRIOR_ART_20261115.md) và
[protocol preflight](../../experiments/scope-p1b-temporal-preflight-v1.md).
Task mới là **directed temporal coincidence**: hai event A/B, nhãn dương khi
A trước B và gap ≤2; thứ tự ngược hoặc gap dài là âm. 32 ticks, input 64.
Không phải terminal-only pulse-gap cũ, XOR/OR một tick hoặc MOVE. Timing,
order và amplitude thực sự đi vào membrane qua trọng số khác 0.

Teacher H128 có S64/P64; student chỉ S64. P có coincidence detectors và event
relays **được dựng bằng tay**, không phải P đã tự học. Correspondence từ
P64..119 tới S8..63 được cung cấp trước; không có learned aligner. Student
witness cùng topology, precision, head, bounds và trainable coordinates giải
được old+new. Mọi recurrent weight R=0 và giữ nguyên: có ALIF membrane/
adaptation theo thời gian, nhưng chưa thử học recurrent inter-neuron credit.

Teacher và witness đạt 100% trên qualification riêng, không dùng chúng làm
kết quả học. Teacher trace biến thiên trong lớp (min conditional variance
0,122474); median khác biệt **physical delta sau optimizer/bounds** giữa
teacher đúng và tráo khoảng 28,89–29,04% trên cả ba seed/hai dấu. Causal-prefix
check không thấy target tương lai đi vào trace. Initial RO có collision và
ceiling 75%, còn headroom. Preflight đủ **phân biệt cơ chế**, chưa chứng minh
cơ chế hữu ích. Trace là hàm input, không là thông tin Shannon mới vượt input.

[Recovery audit](../../runs/20261115T174514Z-scope-p1b-v1-b30f0b/preflight-audit-recovery.json)
tự dựng lại gates/physical delta từ raw, 0 model call. Audit đầu tiên lỗi
serialize `numpy.bool`; lỗi và INTAKE_RUNTIME_STOP được giữ nguyên. Chỉ sửa
serialization, kiểm lại raw; không đổi dữ liệu, gates hay số học learner.
Trước training chỉ thêm diagnostic arrays vào giá trị trả về; AST numeric
forward/proposal và computational prefix learn giữ nguyên so với preflight.

## 4. Đã train thật: đúng 24 arm-runs, không sweep

[Protocol khóa](../../experiments/scope-p1b-temporal-transfer-v1.md),
[registration](../../runs/20261115T174514Z-scope-p1b-v1-b30f0b/learning/registration.json).
Seeds 261161/261167/261173 × readout sign +/− × A/B/C/D = **24 tổng cộng**.
Ba worker CPU chạy các arm độc lập, mỗi worker một BLAS thread.

Mỗi seed: TRAIN192, DEV128, sealed TEST192; salt tách biệt. TRAIN clock16/20,
DEV18/22; TEST17/21/23 và các amplitude bands chưa train. DEV không phải bằng
chứng OOD toàn diện; TEST chưa được model nhìn. Mỗi arm 192 update đúng thứ
tự khóa; snapshot DEV ở 0/48/96/192, không chọn best epoch. Lớp dương chỉ 25%:
**75% accuracy có thể hoàn toàn là dự đoán lớp đa số**. Protocol yêu cầu thêm
min-class recall ≥65%, gain ≥5pp trên từng seed và cả hai dấu.

| Arm | Hidden | Target cuối | Tín hiệu phụ |
|---|---|---|---|
| A label-only | Local eligibility | Cùng one-hot label | Không áp dụng hint |
| B teacher | Cùng rule | Cùng label | Filtered P trace đúng mẫu/thời gian |
| C shuffled | Cùng rule | Cùng label | Cùng trace multiset, tráo trong label/clock |
| D readout-only | Frozen | Cùng label | Không áp dụng hint |

ALIF forward dùng đúng source core, subtractive reset. Với `q=clip(f/.1,−5,5)`,
eligibility E_v/E_a/E_f cập nhật tiến theo own-neuron surrogate derivative:

\[
\begin{aligned}
u_t&=.85v_{t-1}+W_ix_t,\
\theta_t&=.65+.08a_{t-1},\
z_t&=1[u_t\ge\theta_t],\
v_t&=u_t-\theta_tz_t,\
a_t&=.95a_{t-1}+z_t,\
f_t&=.92f_{t-1}+.08z_t.
\end{aligned}
\]

Eligibility dùng sigmoid slope width .4, gồm reset/adaptation derivatives ở
chính neuron; không BPTT/RTRL/autograd. Với end error `e=onehot(y)−p_S`:

\[
m_j={(W_o^Te)_j\over .1}\,1[|f_{j,T}/.1|<5],\qquad
H_{ji}={1\over32}\sum_t{q^P_{j,t}-q^S_{j,t}\over.1}
1[|f^S_{j,t}/.1|<5]E^f_{ji,t}.
\]

`D=m E_f,T` cho A; `D=m E_f,T+H` cho B/C. Stable rows0..7 bị mask; chỉ Win
rows8..63, columns0/1/3/4 được học. Physical update cuối:

\[
M'=.9M+.1D,\quad V'=.99V+.01D^2,\quad
\Delta W=\operatorname{clip}_{[-3,3]}
\left(W+\operatorname{clip}_{[-.01,.01]}
{.008M'\over\sqrt{V'}+10^{-4}}\right)-W.
\]

D không áp dụng hidden update. Tất cả head dùng local output error, rate .04,
delta clip .025 và weight bound3; normalizer cố định, oldhead frozen. Readout
feedback lấy head **trước update**. No P argument trong student DEV inference.

Teacher cache tính 192 mẫu/seed một lần. Mọi arm cùng tính hint-bank arithmetic
và cùng student forward/update count; A/D bỏ direction phụ. Tuy nhiên khi
triển khai A/D có thể bỏ cả cache/bank; không gọi chi phí deployment ngang
nhau tuyệt đối. Manual teacher/correspondence là prior thêm, không miễn phí.

| Công việc đã chạy | Full model episode forwards | Learning calls |
|---|---:|---:|
| Kiểm toán P1 cũ từ raw | 0 | 0 |
| Profile CPU/CUDA, kể cả teacher setup | 18 | 0 |
| Preflight mới | 942 | 0 |
| Shared TRAIN teacher cache | 576 | 0 |
| 24 student trajectories và eval/revert | 21.120 | 4.608 |
| **Tổng study mới** | **22.656** | **4.608** |

Trong 4.608 calls có **3.456 hidden-enabled updates**, tất cả có physical
hidden delta khác 0; 322.560 scalar hidden-weight write events, không phải
322.560 phiên học. Audit riêng tính 96 student reference rollouts +576
teacher reference episodes từ phương trình, **không fitting**; đây là 672
diagnostic reference evaluations, không cộng lẫn thành training forwards.
Teacher reference chỉ ba prototype neuron, không cùng FLOPs với full model.

## 5. Hidden có học hơn readout-only không?

DEV cuối, trung bình ba seed; dấu là dấu **khởi tạo**, không chọn sau khi xem:

| Khởi tạo readout | A label-only | B teacher | C teacher-tráo | D readout-only |
|---|---:|---:|---:|---:|
| Thuận (+) | 75,00% | **90,36%** | 87,50% | 75,00% |
| Bất lợi (−) | 75,00% | **75,00%** | 75,00% | 75,00% |

B dấu thuận theo seed: 87,50 /91,40625 /92,18750%. Revert Win về ban đầu,
giữ learned head: cả ba về 75%. **Có useful hidden contribution trong điều
kiện này**, không chỉ readout learning hoặc retrieval. C cũng có lợi ích
hidden: phần lớn gain chung không đòi đúng sample alignment của teacher.
B so C dấu thuận là −3,125 /+7,03125 /+4,68750pp; một seed C tốt hơn B.

Ở dấu bất lợi, B thay đổi hidden đáng kể nhưng recall lớp dương **0%** ở cả
ba seed. Readout detector cuối đã đổi sang dấu dương: không được giải thích
đơn giản là “head vẫn âm”. Descriptive label/hint cosine có nhiều xung đột
hơn, nhưng chưa là causal diagnosis phân tách decoder, hidden hoặc optimizer.
Witness cho thấy có lời giải; learner không tìm được lời giải tin cậy trong
budget này. Chưa có căn cứ quy thất bại cho thiếu neuron/capacity.

Gộp hai dấu, B−A/D=7,6823pp, B−C=1,4323pp. Paired stratified bootstrap95%
cho B−C là [0,6510;2,2135]pp; Bonferroni-adjusted98,333% là [0,3906;2,3438]pp.
Đây là CI **theo episode, giữ cố định ba fitted models**, không phải CI tổng
quát qua model seeds. Seed-population uncertainty vẫn INCONCLUSIVE_N3.
Phép sign-flip hậu kiểm trên đúng ba seed cho B−C p=.5, chỉ sensitivity mô tả,
không thay metric/gate đã đăng ký. Không dùng pooled CI để che thất bại từng
dấu, gain <5pp so C, hoặc min-class recall chưa đạt.

[G2](../../runs/20261115T174514Z-scope-p1b-v1-b30f0b/learning/G2-decision.json)
**FAIL**: acquisition/min-class, hidden-revert trên mọi seed/dấu và gain từng
cell không đạt. CI input-conditional đạt riêng không biến G2 thành PASS.
Old valid 100% ở mọi arm đến từ 8 stable neurons/oldhead được khóa: **bảo vệ
bằng cấu trúc**, không chứng minh learner tự chống catastrophic forgetting.

## 6. Loại P và restart: chưa được kiểm chứng

DEV chạy trên student S64 không nhận P/teacher archive hoặc hint; đó là
P-free inference trong tiến trình học. Nhưng **G3 hard-removal + fresh-process
reload chưa chạy vì G2 fail**. Có 24 final S-only checkpoints với hashes và
đầy đủ trạng thái học; việc lưu được chúng không chứng minh restart retention.
TEST vẫn sealed, 0 TEST model forward. Không lấy preflight witness làm learned
restart, không tự mở G3 để tìm điểm đẹp hơn.

## 7. Reset/reuse P chu kỳ sau: chưa chạy

[Conditional stages](../../runs/20261115T174514Z-scope-p1b-v1-b30f0b/learning/conditional-stages.json):
G3=NOT_RUN_G2_FAILED, G4=NOT_RUN_G2_FAILED. Chưa reset/huấn luyện P2, chưa học
union kỹ năng vòng hai. Conditional code đã đăng ký nhưng nhánh này **chưa
được thực thi/qualification**. Không reset bất kỳ P của serving model.

## 8. Phần cứng, tốc độ và tài nguyên thực tế

Đo trực tiếp: i5-13500HX, 14 physical/20 logical; RAM34.065.997.824 bytes;
RTX4050Laptop6141MiB, CC8.9, driver616.92; CuPy14.2.0, CUDA runtime12090,
driver API13040. Không cài backend/driver hoặc dừng Hermes.

Workload đúng shape ALIF32ticks gồm trace/proposal, transfer/kernel/sync:
median hai episode CPU **3,9087ms**, CUDA **237,8319ms**; CPU nhanh khoảng
**60,85×**. Maxabs parity2,98e-8. Vì workload nhỏ và kernel tuần tự, chọn CPU
NumPyFP32, không ép GPU theo sở thích cũ. Không suy rộng GPU luôn chậm hơn.

| Phase | Wall đo | Sampled experiment-tree peak RSS | Sampled private commit |
|---|---:|---:|---:|
| Intake: profile/preflight/audit đầu | 9,6341s | 591.507.456B | 596.291.584B |
| Learning + metrics, ba worker | 18,1823s | 186.839.040B | 130.551.808B |
| Final independent arithmetic audit | 0,6629s nội bộ | xem audit process logs | không lấy số sampler cũ |
| Data-only CSV/plot export | 0,7442s nội bộ | không chạy neural | không chạy neural |

Tổng wall24 trial cộng riêng là50,0351s do có chạy song song; không cộng nó
vào parent18,1823s để nhân đôi. Các phase thực thi chính khoảng30s; đây
**không phải** toàn bộ thời gian đọc, thiết kế và viết code. Budget ledger
dùng reservation75s cho chi phí trước learning, cộng18,18s và audit/export;
vẫn dưới100s trước closure bookkeeping, rất xa trần1800s. Receipt closure
ghi phần tính lại/custody riêng, không biến thời gian biên soạn thành training.

Tree được sample50ms; per-process Windows commit cap2GB và OS process peaks
được ghi. Đây không phải hardcap2GB toàn tree hoặc một chứng minh không có
peak cực ngắn giữa samples. Các peak đã đo thấp hơn trần rất nhiều.

GPU chỉ dùng profile, context đã thoát; learning không tạo CUDA context.
CuPy retained pool111.616B **không phải full VRAM peak**. WDDM runtime budget
free khác NVML physical free; whole-card trước/sau khoảng2885/2800MiB free,
delta~85MiB không thể quy toàn bộ cho thí nghiệm. Không có reliable full
per-process physical VRAM peak; ghi **UNKNOWN**, không dựng con số peak.
Nhiệt GPU54→55°C, CPU throttling/temperature chưa đo được độc lập.
Gateway59864 và llama61708 cùng PID/creation time trước/sau; không sửa model,
config hoặc checkpoint live. Điều đó không thay latency benchmark của Hermes.

## 9. Prior art: adaptation, không phát minh kiến trúc

Lifecycle chuyển active/plastic sang stable rồi reuse là gần
[Progress & Compress](https://arxiv.org/abs/1805.06370); correspondence temporal
features là feature distillation, local forward sensitivities kế thừa
[e-prop](https://www.nature.com/articles/s41467-020-17236-y). Đối chiếu
[Progressive Networks](https://arxiv.org/abs/1606.04671),
[HLOP-SNN](https://arxiv.org/abs/2402.11984) và ROW nằm ở prior-art matrix.
Không thêm frozen columns hoặc projector trong P1b; không nhận kết quả của
các tác giả như SCAR-X đã tái lập. Không tìm thấy artifact SCAR-X đã trả lời
đúng temporal sample-alignment hypothesis này trước khi đăng ký; đó là phạm
vi kiểm tra repository, không là chứng minh tính mới toàn cầu.

Đóng góp hiện tại là một **negative controlled pilot** cùng bằng chứng useful
hidden learning có điều kiện; chưa có teacher tự học, robust local transfer,
functional compilation hoặc capacity release được chứng minh. Hai reviewer
PASS của paper-spec chỉ nói tính nhất quán, không thay kết quả này.

## 10. SCAR-X đã giải tự học liên tục chưa? NOT YET

Chưa chứng minh chuyển kỹ năng robust trước dấu credit bất lợi, G3 restart,
G4 plastic reuse, sửa tri thức sai, boundary-free learning hoặc 20-stage
retention. Các ngưỡng mean≤3pp/worst≤5pp sau20stages vẫn nguyên và chưa được
đo trong study này. Một oldhead frozen và task temporal nhỏ không chứng minh
ngôn ngữ, video memory, reasoning đa năng hoặc lợi thế so RAG.

## 11. Quyết định và đúng một bước tiếp theo

**PIVOT; STOP cấu hình P1b hiện tại.** Preflight chứng minh khả năng biểu diễn
và phân biệt signal; training bác bỏ claim rằng rule này khắc phục reliably
credit sign sai và tạo ≥5pp lợi thế trước matched shuffle. Không có lý do
tăng neuron, chọn dấu thuận, sweep hệ số hoặc chạy thêm seed để cứu kết quả.

**Đề xuất duy nhất, chưa thực hiện:** preregister một diagnostic **teacher-blind
readout refit trên các checkpoint P1b đã lưu**, dùng TRAIN/DEV, giữ hidden
đóng băng và cùng ngân sách fit giữa các điều kiện. Mục tiêu chỉ phân biệt
“hidden đã có discriminative features nhưng decoder không khai thác” với
“hidden chưa học representation hữu ích”. Đây là fitted diagnostic mới,
cần protocol/phê duyệt riêng, không nằm trong24 arm đã hoàn tất. Không dùng
TEST hoặc lấy diagnostic upper bound thành thành tích ALIF-local.

## Bằng chứng và tái kiểm toán không train

Root: [run duy nhất](../../runs/20261115T174514Z-scope-p1b-v1-b30f0b/).

| Artifact | Nội dung |
|---|---|
| [Raw CSV](../../runs/20261115T174514Z-scope-p1b-v1-b30f0b/summary/raw-predictions.csv) | 16.512 prediction rows; DEV, old-valid, hidden-revert; không TEST |
| [Biểu đồ](../../runs/20261115T174514Z-scope-p1b-v1-b30f0b/summary/accuracy-by-readout-sign.png) | Means và đủ ba seed, cả hai dấu |
| [Raw trials](../../runs/20261115T174514Z-scope-p1b-v1-b30f0b/learning/trials/) | 24 learning.csv, eligibility/delta arrays, predictions, 48 checkpoint hashes |
| [Audit mới](../../runs/20261115T174514Z-scope-p1b-v1-b30f0b/learning/independent-audit.json) | 4.608 physical updates/loss/head updates; metric/bootstrap/checkpoint/data checks |
| [Custody](../../runs/20261115T174514Z-scope-p1b-v1-b30f0b/closure/custody.json) | Hash mọi artifact sau closure, source/protocol và file gốc được bảo toàn |

Auditor không import learner, tự tính metric/bootstrap, giải ALIF literal cho
96 student và576 teacher reference episodes, đối chiếu label từ input,
trace/cache, masks, initpairing, fixedR/normalizer, checkpoints và actual
delta. Max update algebra error3,30e-8; max reference error7,92e-8; failures[].
Đây là **implementation độc lập do cùng Codex viết**, không phải một người
độc lập hay một replication training mới. Audit PASS là tính nhất quán của
bằng chứng, không quality PASS.

Muốn đọc lại bằng mã, dùng `audit_scope_p1_saved_v2.py` hoặc
`audit_scope_p1b_learning.py` với `--run` tương ứng và **output mới**; không
gọi runner learning. Closure verifier chỉ CSV/hash/AST, không forward/refit.
Nguồn đầy đủ có snapshot và hash trước học; mọi lỗi/negative raw giữ nguyên.
