# D1/D2 P1b forensic: thông tin có ích, decoder và representation đều chưa ổn

16/11/2026 (Asia/Saigon). **PIVOT, không đổi P1b G2 FAIL.** Chỉ đọc checkpoint
P1b, chạy frozen inference trên TRAIN/DEV và fit probe; 0 hidden update,
không TEST/restart/reuse hoặc thay head của model đã lưu/đang dùng.

Run [D1/D2](../../runs/20261115T192506Z-rc-d12-v1r1-73f7fe/), source khóa trước
execution `73f7fe733e6fd6a63182a4e94b2fb3d0529c42f5`; evidence base P1b
`1ea065be194d44000ba795138c5917f3d13adda0`. Các số mới không thay raw/kết luận
của P1b, không biến 24 subjects thành 24 model seeds.

## 1. Phép kiểm đã khóa

[Protocol](../../experiments/rc-p1b-diagnostics-v1.md): mọi3seed, cảhai dấu,
4arms, initial vàfinal. Khóa Win/R/indices; dedup các hidden weights bằng nhau
trong mỗi seed →21 hidden states, không train lại. Mỗi state chỉ forward
192TRAIN+128DEV. Labels chỉ dùng trong probe/scorer, không ALIF forward;
không materialize teacher targets/identity hoặc sealed TEST.

Primary probe: raw endpoint f, TRAIN-only mean/std floor1e-6, class-balanced
binary ridge alpha1, intercept không phạt, targets±1. Quyết định score>0;
|score|≤1e-12 là numericaltie. Secondary q=clip(f/.1,±5), cùng fixedprocedure;
không chọn lambda/variant/threshold bằng DEV. Float64 chỉ cho phép giải đại
số diagnostic; checkpoint/forward ALIF giữ FP32. Không dùng probe coefficient
làm learned local-ALIF head hoặc cập nhật production.

## 2. Kết quả trực tiếp và probe khác nhau thế nào?

Balanced accuracy trung bình3seed, luôn giữ mẫu từng seed trong raw:

| Sign | Arm | Online head | Common raw probe |
|---|---|---:|---:|
|+|Label-only|50,00%|50,00%|
|+|Teacher|80,73%|100,00%|
|+|Shuffled teacher|75,00%|97,22%|
|+|Readout-only|50,00%|50,00%|
|−|Label-only|50,00%|50,00%|
|−|Teacher|50,00%|80,56%|
|−|Shuffled teacher|50,00%|90,28%|
|−|Readout-only|50,00%|50,00%|

Không so balanced accuracy ở đây với raw accuracy75% cũ như cùng metric.
Đặc biệt teacher dấu−:

| Seed | Online BA / recall+ | Probe accuracy | Probe BA | Probe recall− / recall+ | Probe AUC | TRAIN probe BA |
|---|---|---:|---:|---|---:|---:|
|261161|50% /0%|68,75%|70,83%|66,67% /75%|0,8457|75,00%|
|261167|50% /0%|100%|100%|100% /100%|1,0000|100%|
|261173|50% /0%|68,75%|70,83%|66,67% /75%|0,8333|62,50%|

Teacher dấu+ có TRAIN/DEVprobe100% cả3seed và AUC1. Online teacher dấu−
seed261167 đã có AUC1 nhưng threshold hiện hành không nhận một positive nào;
đây là bằng chứng rõ trong **checkpoint cụ thể đó** rằng ranking có ích chưa
được decoder khai thác đúng. Hai seed còn lại có AUC<1 và fixed linearprobe
TRAIN không hoàn hảo, nên chưa thể quy toàn bộ thất bại cho bias/readout.
Một fixed ridge thất bại không chứng minh mọi linear/nonlinear decoder đều
không thể giải hoặc raw information bằng0.

Điều kiện diagnostic đăng ký trước đòi mọi adverse seed probeBA≥75%, cảhai
recall≥65%, gainBA≥10pp so online. Hai seed BA70,83% làm điều kiện **FALSE**;
không hạ threshold xuống70%. Initial/hidden-reverted và RO commonprobe50%BA;
teacher đã tạo endpoint features hữu ích kể cả ở dấu−, nhưng còn biến thiên.
Shuffled teacher lại tốt hơn teacher ở dấu− sau commonprobe: không có căn cứ
claim sample-specific teacher transfer robust, chỉ vì B hơn label-only.

## 3. D2 raw features và thông tin quan sát được

- Label-only và RO: endpoint raw rank0, variance0,64/64 neuron không spike
  trên các episode của task mới; commonhead chỉ đạt baseline. Weight movement
  trong A không tạo được discriminative spikes ở task này.
- Teacher+ TRAIN centered rank5/11/7; teacher−7/9/6. Không dùng rank như số
  kỹ năng còn chứa được. Teacher− observed-never-spiking19/8/24 neurons;
  trong đó8 stable old units không nhận channel của task mới. “Silent ở task
  này” không đồng nghĩa neuron chết vĩnh viễn hoặc mất oldskill.
- Teacher TRAIN class centroid distance+≈.1027/.1244/.1210; −≈.0266/.1371/.0440.
  Seed−yếu có class gap nhỏ hơn nhưng đó là association, không chứng minh
  hướng credit cụ thể là nguyên nhân.
- Không có opposite-class **exact** endpoint collision trong6teacher states
  trên TRAIN; không loại được nearcollision hoặc convex overlap. Rank0 của
  A/D là khác hẳn việc teacher features có độ phân biệt nhưng khó đọc.
- q saturation trên toàn32tick bằng0; raw và q probe classification trùng.
  Với các checkpoint này, normalization clipping không giải thích failure.
  D3 chỉ có endpoint cũ; D2 bổ sung all-tick aggregate bằng frozen inference.

## 4. Uncertainty và mức kiểm toán

Paired stratified episodebootstrap,10.000draws, giữ3fittedmodels cố định:
teacher commonprobe−online BA gain+19,27pp, conditional95%[17,19;21,35]pp.
Adverse gain+30,56pp; interval degenerate ở30,56pp vì predictions cố định
trong các strata order/gap/clock tại bộ dữ liệu này. Nó không phải độ chắc
chắn qua model seeds hoặc assurance tương lai. Adverse Bprobe−Cprobe
−9,72pp, conditional95%[−11,81;−7,64]pp. Các CI không phải joint-qualitygate,
không projectCI95% qualification; seedpopulation N3 vẫn INCONCLUSIVE.

[Auditor độc lập implementation](../../scripts/research/rc_audit_p1b_probes.py),
[protocol](../../experiments/rc-probe-audit-v1.md), source trước audit
`3017442ac4b357ffc1dde08d58c46148e8eb777d`,
[raw result](../../runs/rc-audit-p1b-20261115T192828Z-301744/result.json):
24subjects/21hidden states,42 augmentedweighted `lstsq` verification solves,
không gọi solver normal-equations của runner. Max coefficient error8,64e-14,
score6,62e-10, metric6,44e-12, D2info2,09e-8, direct probability3,28e-7.
Source/custody/checkpoint/input invariants khớp; G2FAIL giữ nguyên.

Auditor là cùng hệ Codex, không external peer review/trainingreplication.
42lần giải lại cùng fixedproblem là công việc audit, không42 fitted models
mới. Không neuralreplay/TEST/teacher-H; tick statistics chỉ kiểm aggregate
arithmetic/provenance, không independent extraction lại spike tape. Conditional
bootstrap chưa được auditor này tái tính; ghi **runner-computed**, không nâng
toàn bộ CI lên independent verification.

## 5. Thực thi, scheduler và sự cố được giữ

Một intake trước [192329](../../runs/20261115T192329Z-rc-d12-v1-c22e39/FAILED.json)
dừng vì byteguard CRLF/LF,0forward/fit. [Recovery riêng](../../experiments/rc-p1b-diagnostics-provenance-recovery-v1.md)
chỉ ghi cả raw-source/git-blob hashes, yêu cầu canonicalLF bằng nhau và copy
nguồn thực chạy. Không sửa production source hoặc số học/data/seed/metric.
Giữ failure/stdout/stderr; reservation2s nằm trong300s budget mới.

| Công việc | Số lượng/thời gian thực |
|---|---|
| CPU/CUDA frozen profile |14episode; CPU1,5104ms vsCUDA36,5584ms/twoepisodes, CPU≈24,20x |
| Worker calibration |1/2/3workers:655,5/956,8/1058,2episode/s, gồm startup;1152forwards |
| Feature extraction |21×320=6720forwards,0hiddenupdates; chọn3workers,1BLASthread/worker |
| Diagnostic fits |42 fixedridge solves, không lambda sweep |
| D1/D2 elapsed |8,7285s, gồm profile/extract/fit/stats; không là toàn bộ thời gian viết code |
| RSS tree lấy mẫu |Profile605.356.032B; CPUanalysis180.342.784B |
| Independent audit |0,9065s, sampledRSS41.172.992B,42algebra solves,0forwards |

Toàn study mới có7886frozen episodeforwards (14+1152+6720), không lấy đó làm
trainingupdates. Pool memory không là fullphysicalVRAMpeak dướiWDDM. D3/D4 đã
xong trước root profile; không dùng riêng peak của mộtprocess như toàn hệ.
Hard caps là perprocess commit, tree được sample50ms, không có chứng nhận
peak tức thời giữa samples. Budget/resource receipts giữ riêng từng phase.

## 6. Quyết định nối với archaeology

[D3](RC_D3_SIGNAL_FORENSIC_20261116.md) cho thấy label/hint conflict cao hơn ở
dấu−, nhưng physical:D conflict gần giống hai dấu; weight/endpoint bounds
không bị chạm. [D4 index](AUTONOMOUS_RESEARCH_INDEX_20261116.md) cho thấy các
vấn đề readout scaling, preconditioning và feedback direction đã có tiền lệ;
không mở lại sweep momentum/HLOP/EMA hoặc đổi tên family cũ.

**Giữ:** commonhead/rawfeature diagnostics, matchedcontrols, actualdelta audit,
ALIF/local eligibility làm baseline; representation contribution có điều kiện.
**STOP:** claim “chỉ thiếu neuron”, “teacher đúng luôn hơn shuffled”, hoặc
“ridge100% là local learner đã sửa xong”. Không lấy probe này thành inference
head live, không mở P1bG3/G4/TEST. Nút thắt có bằng chứng là decoder/credit
coordination cùng sự ổn định của representation; chưa cần đổi neuron chỉ vì
P1b thất bại. Hướng operator riêng phải qua adequacy/primitive gates, không
được lấy một state interface lập trình sẵn thành learned composition.

[Bảng CSV](../../runs/20261115T192859Z-rc-diagnostics-export-v1/diagnostic-metrics.csv)
và [biểu đồ](../../runs/20261115T192859Z-rc-diagnostics-export-v1/direct-vs-common-probe.png)
được xuất từ saved metrics,0fit/forward. Không có cải thiện nào được promote
vào sản phẩm; user đã yêu cầu tắt Hermes backend trong chiến dịch này.
