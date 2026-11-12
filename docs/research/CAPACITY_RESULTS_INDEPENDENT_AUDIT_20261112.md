# Capacity H64/H128 — kiểm toán kết quả độc lập

Ngày12/11/2026. Reviewer `/root/capacity_results_independent_audit`. Chỉ đọc
source, protocol, dữ liệu và checkpoint; không gọi `learn_episode`, không
huấn luyện hoặc chọn/tune model. Ponytail không có trong tool inventory.
Main/Hermes/liveSQLite/process/port không bị tác động bởi reviewer.

**INTERIM: các cell đã hoàn tất hợp lệ; chưa có quality verdict.**
Lượt chạy vẫn đang thực hiện lúc viết phần kiểm bước đầu; chỉ audit đầy đủ
252record mới được kết luận theo gate đã đăng ký.

## Nguồn và phạm vi đã đọc

- Worktree `scarx-capacity-h64-h128-20261112`, execution source
  `691de7c02127b099a79a752fdd89a940a92b1331`.
- Run `20261112T132737Z-capacity-dev-v1-160b6e`, protocol
  `experiments/capacity-h64-h128-approved-v1.md`, phê duyệt trực tiếp H64/H128.
- Ba seed DEV140011/140017/140023, không confirmationbank; bốn nhiệm vụ sự
  kiện hai nhãn. H là fixedallocation của mỗi trial, không phải dynamicgrow.
- Primary144records =48H48reuse +96new H64/H128. Secondary108new equalCPU,
  tổng252records/204newtrainingjobs. Configs/arms/tasks không thành seed độc lập.
- ALIFhardforward, localeligibility và sourceoptimizer kế thừa; không
  pretrained/Transformer/globalBPTT/RTRLcandidate. H48 policy giữ nguyên.

## Kiểm bước đầu đã thực thi

Lệnh độc lập không có cập nhật học:

```powershell
python -X utf8 scripts/verify_capacity_independent.py --run runs/20261112T132737Z-capacity-dev-v1-160b6e
```

Snapshot97completedrecords (48reuse+49new) đã kiểm:

| Hạng mục | Số lượng / kết quả |
|---|---:|
| Locked source bodies / peer bodies |15 /2, hashes khớp |
| Frozen dataset ZIP |12, khớp byte với parent |
| Threshold-based event oracle độc lập |27.648 nhãn, exact |
| H48 copied evidence |192file/48cell, source/destination hash exact |
| Prediction entries chấm scalar lại |496.640, exact |
| Full checkpoint byte-roundtrip |97, exact |
| Manual literal ALIF replay |1.552 predictions, exact |
| Probe qua SVD độc lập |528 predictions, exact |
| Within-H full initial-state pairs |72, exact |
| Accepted history update calls kiểm |99.328 |
| Full-revert probe recovery về frozen |96, exact |
| Learning update do reviewer tạo |0 |

Script oracle không gọi generator/event_oracle của runner để chấm labels.
Nó tìm event bằng threshold trên semanticchannels, kiểm unique keys/events rồi
tính binding/XOR/order. Manualforward tạo statearrays riêng và thực hiện đúng
subtractive adaptive reset, adaptation và featurefilter, không gọi model.step.
Probe giải cùng ridgeobjective bằng SVD, không dùng normal-equation solver
của runner; center/scale chỉ lấy từ refit. Script kiểm độc lập label/prediction
schema, metrics direct/probe và3hiddenreverts, count/shape/history/RNG, topology,
readout-onlyhiddeninvariance, checkpointsha và initpairing cùngH.

## Diễn giải bắt buộc

1. So cùngarm khi đổiH. Không chọn bestarm theo task; `groups_shuffle` là
   control, không hậu nghiệm đổi thành winner.
2. Pairedseed quaH không là weightsbitwiseidentical. Fan8/H, gammaresampling
   và số lượng localparameter đổi cùngH; đây là covariates đã công bố.
3. H48rawquality có thể reuse sau exactparity. Các số CPU từ parentdualtrace
   chỉ là lịch sử; CPUcomparison dùng108new contemporaneouscells hiện tại.
4. Arraybytes trong rowH48 được quy về adapterlayout; `reused_source_array_bytes`
   vẫn giữ bytes của sourceH48dualtrace. Không gọi arraybytes là processRAM.
5. Readout-onlyrandomfeaturegain khi tăngH không tự chứng minh hiddenlearning.
   Phải tách direct, refitprobe, gain-of-gains và causalrevert; ceilingorder
   không miễn causalchecks hay OOD/protectedgates.
6. Ba seed chỉ đủ khám phá. Script thống kê liệt kê đúng27clusterbootstrap
   combinations và8signflips, minimumtwo-sidedp0,25, không coi252record là252seed.
7. Không có P2retention, chat/vision tổng quát hoặc phép đo energy trong scope.
   H128 chưa tiến bộ vẫn chưa đủ chứng minh architecturalceiling.

## Phần cần kiểm khi đủ dữ liệu

Script có `--require-complete`: từ chối thiếu cell, workererror, khác204new/
48reuse/252records, hoặc đổi main/resource/time/scope. Sau khi đủ, nó tái dựng
capacitygates theo từngH và từngarmown/groups_up, giữ mọi margin2đ/regression3đ,
causal3đ/order và sourceCIrule. EqualCPU được chấm riêng; không thay primary.

```powershell
python -X utf8 scripts/verify_capacity_independent.py --run runs/20261112T132737Z-capacity-dev-v1-160b6e --require-complete
```

Finalmeans/gates/resources chưa được phê duyệt từ snapshot bước đầu này.
Không có blocker integrity mới trong các cell đã đọc; cần finalaudit trước
promotion hoặc kết luận capacity đã giải lỗi. Reviewer chưa đề nghị mở scale
vượtH128, confirmation, P2, main hoặc Hermes.

## Kiểm độc lập khi primary144records đã đủ

Reviewer đọc đủ144primaryrecords và tính lại các điều kiện theo **cùngarm**,
không chọn từngtask. H128Own có directgains soH48 lần lượt Binding+1,888đ,
XOR+7,552đ vàDistractor+12,435đ. ID/OODpreservation,order vàcausalguards đều
qua. Nhưng tăng advantage của learned-hidden so với frozenprobe lần lượt
−4,167đ,−5,664đ,+13,281đ: chỉ1/3family đạt+2đ, protocol đòi ít nhất2/3.
H128groups_up cũng chỉ1/3family đạt directmargin2đ và1/3 đạt hiddenadvantage
growth2đ. H64 còn regressBinding/OOD quá3đ ở cả hai learnedarms.

H128readout-only có directgains+8,984đ/+10,612đ/+6,510đ và các bảo vệ ID/OOD/
Order qua. Vì vậy mapping `CAPACITY_ONLY_SIGNAL` trong parentauditor đúng
definition đã khóa nếu fullgrid/integrity cuối còn hợp lệ.

**Nhãn đó không nghĩa là không có hiddenlearning.** H128Own causal directdrop
khi revert-both là32,878đ/36,458đ/18,685đ ở ba primarytasks; hiddenprobe và
directskill có ích vẫn tồn tại. Gate đang hỏi liệu **phần lợi ích tăng thêm**
của hiddenlearning sau khi trừ frozenrepresentation tăng đủ ở>=2family không.
Nhiều gain do thêm randomfeatures là confound đã kiểm, không là phủ nhận mọi
learned-synapse contribution. Không đổithreshold để biến rawdirectgain thành
`CAPACITY_DEV_SIGNAL`. Chưa finalqualityverdict ở snapshotprimary vì equalCPU/grid còn chạy.

Giữ nguyên sourceWoutstd0,03 và per-coordinate learningrate/clips khi đổiH
còn thay aggregate readoutvariance và norm của updatevector. Đây cũng là
optimization-conditioningcovariate, ngoài fan8/H và gammaresampling; không
âm thầm rescale sau scores. Phép thử đo learner thực tại mỗi kích thước,
không là purecapacitylaw loại hết mọi covariance.

## FINAL — đủ252records, integrityPASS

Worker kết thúc với `error=null`,204newjobs+48reuse đúng grid, wall
996,3980644giây dưới3.600giây. Reviewer chạy lại verification độc lập trên
toàn bộ dữ liệu, không cập nhật model. Các bản JSON audit của reviewer ở
`local/capacity-independent-final.json` và `local/capacity-independent-final-v2.json`;
bản v2 bổ sung crosscheck đầy đủ parentcontrasts/policy, không đổi số liệu.
Muốn xuất chứng cứ riêng phải chọn pathmới, không overwrite file cũ.

| Kiểm độc lập cuối | Kết quả |
|---|---:|
| Expected / observed records |252 /252 |
| Reused evidence file hashes |192 /192exact |
| Threshold-oracle labels |27.648exact |
| Scalar prediction entries, gồm interventions |1.290.240exact |
| Checkpoint coldload+byte-roundtrip / counters |252exact |
| Manual literal hard-ALIF predictions |4.032exact |
| Probe bằng SVD độc lập |768exact |
| Same-H full initial-state pairs |216exact |
| Full-revert frozenprobe recoveries |144exact |
| Fixed groups router RNG initial/final |108exact |
| Actual arm-policy checks |252exact |
| Accepted history update calls |164.202 |
| Learningupdate do reviewer tạo |0 |

15lockedsourcebodies+2peerbodies, peerHEAD/clean,12frozendatasets và parent
manifest/auditreceipts còn khớp. Reusedsource/path/initial/checkpoint/RNG và
actualH64/128array/accounting đều đúng. Historytimestamps monotone trong
CPUcolumn và primary1.024updates; CPUcells tuân thủ0,5s/episodeboundary/cap4.096.
Policygroupfixed không âm thầm tiêu thụ routerRNG, readout-onlyhiddenweights
không đổi; allsameHinitialcore/normalizer/optimizer pairs exact.

Reviewer tái dựng độc lập đủ192meanmetrics; **168contrasts** gồm120size/
hiddenadvantage contrasts và48groups-vsshufflecontrasts. Toàn effects,
mean/CI/exactp khớp parentaudit; không pooling config/tasks như independent
seeds. Sáu scalarfixtures bootstrap/sign-test cũng qua (NumPy và scalar mean
có groupingdifference tối đa1,39e−17, không đổi bất kỳ gate nào).

**Finaldecision `CAPACITY_ONLY_SIGNAL` đúng protocol.** Không cặpH/learner
nào đạt toàn bộ `CAPACITY_DEV_SIGNAL`; các lý do primary đã ghi ở trên không
thay đổi khi fullgrid hoàn tất. Đây là kết luận về **mức tăng thêm của hidden
advantage**, không phủ nhận directgains hoặc hiddenlearning đang có.

EqualCPU được đo mới108cells và giữ tách khỏi primary. H128Own tại0,5s có
directBinding56,84%,XOR73,70%,Distractor52,28%,Order97,92%; H48Own lần lượt
51,04%/64,84%/49,02%/98,76%. H64Own có XOR81,64% ở budget này, nên không
được lấy H128winner ởequalupdates để tuyên bố larger luôn tốt hơn khiCPUfixed.
Vòng này không đăng ký matchedphysicalnorm giữaH, không claim tách hết
capacity khỏi optimization/initialvariance/computecovariates.

Resourcecontract giữ: peakownedtreeRSS148.815.872B, committed191.148.032B;
Windowsper-processcap receipt `enforced=true`, không measurementerrors hoặc
overlimit. `ResourceMonitor.hard_cap_enforced=false` là flag riêng chưa nối
receipt, không phủ nhận cap thực hoặc dùng nó như giảRAMcap. Mainref vẫn
`e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541`; no freshconfirmation/P2/production.

**Đề nghị:** giữ H128Own như candidate/đối chứng nghiên cứu có ích và giữ
toàn bộ tests/checkpoint/evidence. Chưa automaticpromotion main/Hermes và
không tự mở H256/P2/freshconfirmation. H48 quả thực không nên mặc định đủ
dung lượng; H128 cải thiện một số kỹ năng nhưng learning/readout/continual
retention còn là câu hỏi riêng. Không suy ra hiểu ngôn ngữ/ảnh hoặc ngangLLM
từ bốn bộeventtasks2nhãn này. Reviewer không huấn luyện thêm.
