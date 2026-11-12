# BrainTrace: phương trình có thể kế thừa và kiểm chứng độc lập

Trạng thái: **reference adapter đã chạy và qua kiểm tra số học; chưa kết luận
chất lượng học**. Đây là NumPy adapter tự viết theo phương trình D-RTRL, không
phải chạy compiler BrainTrace hay tái lập điểm benchmark trong paper.
Ponytail không xuất hiện trong tool inventory khi kiểm toán.

## Nguồn, phiên bản, giấy phép

- Paper: Wang et al., *Model-agnostic linear-memory online learning in spiking
  neural networks*, Nature Communications 17,1745 (2026),
  [DOI](https://doi.org/10.1038/s41467-026-68453-w).
- [Repo chính thức](https://github.com/chaobrain/braintrace), pin
  `fafa25753b743a2e2d5b47209c4ee947abc02cfb`, checkout ngoài source production:
  `scar-x/local/references/braintrace`, `git status --short` sạch khi kiểm.
- [LICENSE tại pin](https://github.com/chaobrain/braintrace/blob/fafa25753b743a2e2d5b47209c4ee947abc02cfb/LICENSE): Apache-2.0.
  Không nhập framework hay dependency JAX/BrainState vào app. Adapter dùng
  phương trình công bố, không sao chép compiler.
- [D_RTRL, dòng25–46](https://github.com/chaobrain/braintrace/blob/fafa25753b743a2e2d5b47209c4ee947abc02cfb/braintrace/_algorithm/d_rtrl.py#L25):
  eligibility forward `E_t = D_t E_(t-1) + B_t outer x_t`, rồi kết hợp learning
  signal. [axes.py, dòng240–241](https://github.com/chaobrain/braintrace/blob/fafa25753b743a2e2d5b47209c4ee947abc02cfb/braintrace/_algorithm/axes.py#L240)
  khóa mặc định Jacobian recursion và diagonal scope.
- [hidden_group.py, dòng332–394](https://github.com/chaobrain/braintrace/blob/fafa25753b743a2e2d5b47209c4ee947abc02cfb/braintrace/_compiler/hidden_group.py#L332)
  giải thích block theo vị trí, giữ coupling giữa các state cùng neuron.
  Recurrent ETP mixing bị cắt trong diagonal scope, gồm cả đường tự nối qua
  recurrent operation; đây không phải full recurrent gradient.

Đã đọc upstream numerical tests: `d_rtrl_test.py`,
`tests/axis_acceptance_test.py:118–148` (tách nội động lực và recurrent mixing),
`tests/exact_correctness_test.py:237–255` (unfiltered symmetric EProp trùng
D-RTRL ở điều kiện tương ứng). Các test JAX upstream **chưa chạy trong máy
này**; chỉ đọc source và thực hiện các kiểm tra NumPy riêng bên dưới.

## Adapter giữ nguyên gì, thay gì

`src/scarx/evaluation/literature_braintrace.py::BrainTraceDRTRL` giữ H48, D8,
fan-in8, một lần hard-spike ALIF forward/tick, topology và toàn bộ source
readout/normalizer/optimizer/RNG schedule. Không task ID, không autograd/BPTT.
State local là `(v,a,f)`: voltage sau reset, adaptation, và feature readout
thực sự sử dụng. Với `lambda=leak`, `rho=adaptation_decay`, `beta=adaptation_strength`,
`kappa=feature_decay`, theta và spike hiện tại z, surrogate psi:

```text
D_j = [[lambda*(1-theta*psi), beta*(theta*psi-z),       0],
       [lambda*psi,         rho-beta*psi,               0],
       [(1-kappa)*lambda*psi, -(1-kappa)*beta*psi,   kappa]]
B_j = [1-theta*psi, psi, (1-kappa)*psi]
E_j(t) = D_j E_j(t-1) + B_j outer presynaptic_values(t)
```

Mỗi synapse chỉ giữ độ nhạy state của neuron đích. Giá trị presynaptic spike
được dùng làm input, không truyền độ nhạy qua recurrent mixing. Feature
sensitivity là `e_f` luôn, không lọc thêm lần thứ hai.

Khác source SCAR-X own-reset: own có gamma_j từ.25 tới.92 và sensory gate
cho recurrent eligibility, trong khi forward feature decay=.92. Adapter đưa
trace khớp feature thật và bỏ gate này. Ở số thực, block recurrence tương
đương own-reset với homogeneous gamma=.92, ungated; hai implementation có
thể lệch rounding float32. Không coi chúng là hai thuật toán độc lập nếu
numerical equivalence đã chỉ ra cùng estimator.

Khác stopped-reset: block giữ derivative reset qua spike và adaptive theta;
stopped bỏ cả hai trong voltage eligibility. Khác e-prop adapter: e-prop
detaches reset-spike nhưng vẫn giữ `-z*beta*e_a_old` do theta phụ thuộc a.

Quy ước phép cộng trong estimator giữ đúng pin R8 `(leak*v+i_rec)+i_in`;
core forward vẫn `leak*v+(i_in+i_rec)`. Có thể lệch1ULP vì reassociation.
Đây là quy ước estimator được công bố, không phải tuyên bố finite-difference
của chương trình float32 hard-spike cho gradient phi tuyến trơn.

## Numerical correctness đã chạy

`python -X utf8 -m pytest -q -p no:cacheprovider tests/test_literature_oracle.py tests/test_literature_braintrace.py`

**13 test PASS** trong kiểm độc lập ngày12/11/2026:

- Local3x3 Jacobian và current injection khớp centered finite-difference của
  đồ thị smooth được neo vào cùng hard-spike state.
- Toàn chuỗi: full forward surrogate sensitivity khớp centered finite-difference
  của một perturbed graph độc lập, gồm recurrent weight×spike và self-edges.
  Fixture3neuron/11tick kiểm mọi12hiddenparameter; tolerance2e-8 relative,
  2e-9 absolute. Đó là control nhỏ, không đổi primaryH48.
- Vector JVP và scalar VJP khớp nhau ở full oracle.
- Với recurrent weights=0, full oracle giảm về local owned rows; khi recurrence
  khác0, các đường bị diagonal approximation bỏ đi được phát hiện.
- Forward/core weights không đổi bởi oracle; readout/normalizer/signal RNG giữ
  schedule baseline ở các technical fixture.

Các kiểm tra không lấy finite-difference của Heaviside rồi gọi đó là gradient.
Surrogate được định nghĩa rõ: quanh hard trajectory, `z~=z_hard+psi*delta_margin`.
Oracle full giữ trọng số cố định, truyền sensitivity về phía trước, không tối
ưu hoặc đưa full-gradient oracle vào candidate. Nó xác minh phương trình,
không xác minh hiệu quả học trong mạng đang cập nhật trọng số.

## Thí nghiệm hướng credit đã chạy — không huấn luyện

Runner `scripts/verify_literature_oracle.py`; evidence nguyên bản:
`runs/20261112T080231Z-literature-oracle-v1-3ffd75/`.
Ba numerical fixture seed422,773,1127, mỗi21tick random bounded input; shared
fixed random output-sensitivity vector. Không dùng nhãn dataset, không chọn
candidate, không cập nhật synapse. Bảy estimator đều có forward state giống hệt.
Oracle giữ toàn bộ cross-neuron recurrent sensitivity của cùng surrogate graph.

| Estimator | Cosine trung bình với full oracle | Khoảng3fixture | Tỉ lệ norm trung bình |
|---|---:|---:|---:|
| Own-reset hiện tại |.94345|.92722–.95903|1.17278|
| Stopped-reset hiện tại |.95512|.94704–.96906|4.09591|
| eprop_matched |.94544|.93563–.95988|2.63129|
| eprop_aligned |.99054|.98925–.99210|2.15061|
| BrainTrace D-RTRL adapter |.99829|.99761–.99885|.99775|
| ACR static.5 |.96343|.95749–.97180|2.60460|
| ACR state heuristic |.92637|.90714–.93970|3.14307|

Đây là bằng chứng cơ chế hẹp: trên3frozen fixtures, khớp đúng feature filter
và giữ local reset Jacobian làm hướng estimate gần full surrogate hơn. Recurrent
weights lúc khởi tạo nhỏ; không suy rộng các cosine này sang trajectory đã học,
task accuracy, OOD hay retention. Norm khác biệt lớn nhắc lại cần control actual
update norms sau optimizer/cast/clip; cosine cao chưa bảo đảm update hữu ích.

## Bộ nhớ và tài nguyên

H=48, K=D+fan-in=16: ba trace float32 chiếm `3*H*K*4=9.216B`, thay ba trace
cũ nên **không thêm persistent trace bytes**. Tổng mảng model gồm core và
optimizer theo `memory_bytes()` là29.472B, bằng own baseline; ACR là46.176B
vì thêm16.704B. Đây không phải process RAM.

Các buffer có tên của local block gồm previous/current mỗi9.216B, Dblock1.728B,
Bblock576B, presynaptic3.072B; NumPy còn có temporaries/runtime nên không dùng
tổng này như peak RAM. Full oracle giữ4state-sensitivity matrices float64
`4*H*(H*K)*8=1.179.648B`, chưa tính injection/intermediate copies; nó là control
bậc cao hơn, không cơ chế linear-memory của candidate.

Lượt oracle21rows: wall.463giây, peakRSS51.384.320B, peakcommit68.308.992B,
không vượt2GB; Windows per-process committed cap được thiết lập thành công
trong `cap` receipt. ResourceMonitor riêng vẫn ghi `hard_cap_enforced=false`
vì không được truyền flag, không dùng trường đó để phủ nhận hoặc khẳng định cap.
Không iGPU/GPU/điện năng. Snapshot/CPU DEV của parent là phần kiểm riêng.

## Kết luận kế thừa

- `ALREADY_SOLVED_IN_LITERATURE`: forward eligibility theo intrinsic local
  Jacobian, multi-state coupling, explicit approximation khi cắt recurrent mixing.
- `REPRODUCED_IN_SCARX`: NumPy block adapter, ALIF forward invariance, independent
  surrogate Jacobian và full-sensitivity numerical checks trong phạm vi trên.
- `PARTIALLY_SOLVED`: credit estimate gần oracle trên frozen fixtures; chưa đo
  chất lượng học và độ trôi gradient khi normalizer/optimizer thay đổi.
- `UNRESOLVED`: direct đa kỹ năng, distractor/OOD, continual stability, readout
  forgetting và task-agnostic credit có lợi thế qua nhiều seed.
- Chưa có cơ sở gọi estimator này hay ACR là đóng góp mới hoặc production-ready.
  Ưu tiên paired DEV nhỏ của reference trước khi phát minh thêm router.
