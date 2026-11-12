# E-prop: tái sử dụng phương trình, không gọi adapter là model gốc

**P0 kỹ thuật: GO cho DEV đã đăng ký; chưa kết luận chất lượng.**
Ponytail không có trong inventory. Không cài TensorFlow, không train dataset,
không mở confirmation/P2, không đổi main/Hermes/live.

## Nguồn đã kiểm

- Paper: Bellec et al., *A solution to the learning dilemma for recurrent
  networks of spiking neurons*, Nature Communications 11,3625 (2020),
  [DOI](https://www.nature.com/articles/s41467-020-17236-y).
- [Repo chính thức](https://github.com/IGITUGraz/eligibility_propagation), pin
  `efd02e6879c01cda3fa9a7838e8e2fd08163c16e`; clone tách khỏi production tại
  `scar-x/local/references/eligibility_propagation`, checkout sạch khi kiểm.
- [Source forward/eligibility đã pin](https://github.com/IGITUGraz/eligibility_propagation/blob/efd02e6879c01cda3fa9a7838e8e2fd08163c16e/Figure_3_and_S7_e_prop_tutorials/models.py#L291),
  SHA256 file thực `53360882ef676442fac80311d0a48769cfa1e979ff68bb87e2f095dcb90a3c3b`.
- [LICENSE](https://github.com/IGITUGraz/eligibility_propagation/blob/efd02e6879c01cda3fa9a7838e8e2fd08163c16e/LICENSE):
  điều khoản BSD 3-clause, copyright nhóm e-prop 2019–2020. Adapter giữ thông báo,
  điều kiện và disclaimer. Không mang TensorFlow hay code DECOLLE vào sản phẩm.
- Hai numerical verification upstream có thật:
  `numerical_verification_eprop_factorization_vs_BPTT.py` và
  `numerical_verification_eprop_hardcoded_vs_autodiff.py`. Đã đọc source; chưa
  chạy toàn bộ TF1.15/autodiff scripts trong môi trường này. Không lấy sai số
  khoảng1e-14 trong header làm kết quả chạy của SCAR-X.

## Những gì upstream chứng minh và giới hạn tín hiệu học

Factorization `dL/dw = sum_t learning_signal_j(t)*eligibility_ji(t)` có thể là
đẳng thức nếu learning signal là total derivative thích hợp. Script thứ nhất
dùng `tf.gradients(loss, spikes[t])` để lấy tín hiệu chính xác ấy. Nó không chứng
minh mọi tín hiệu cục bộ xấp xỉ đều cho gradient chính xác. Script thứ hai kiểm
symmetric e-prop theo graph đã cắt các đường truyền spike cụ thể.

SCAR-X tiếp tục dùng tín hiệu trước cập nhật `Wout.T @ error`, mask saturation
và normalization source. Readout/statistics/optimizer được giữ để so sánh công
bằng. Đây là tín hiệu xấp xỉ: không có total derivative của các trạng thái
neuron khác, future learning hoặc cập nhật normalizer phụ thuộc feature.
Oracle dùng frozen parameters/statistics và surrogate được chỉ định; không
phải finite difference của hàm hard-Heaviside vốn hầu hết có đạo hàm bằng0.

## Phương trình source và chuyển đổi tối thiểu

Upstream `models.py:291–323` cập nhật adaptation bằng spike local bước trước,
reset bằng **ngưỡng hằng** nhân spike truyền, rồi tạo spike mới. Spike truyền và
reset có thể stop-gradient; spike local cho adaptation vẫn giữ sensitivity.
Upstream có refractory và triangular surrogate theo threshold, khác SCAR-X.

Ở `models.py:351–366`, với cách đánh chỉ số trước spike hiện tại:

```text
epsilon_v(t) = leak*epsilon_v(t-1) + pre(t)
e(t)         = psi(t)*(epsilon_v(t) - beta*epsilon_a(t))
epsilon_a(t+1) = rho*epsilon_a(t) + e(t)
```

Tương đương recurrence dùng previous psi ở `epsilon_a(t)`. Phần đọc đầu ra
lọc e bằng `decay_out` tại source392–395; không mặc định hetero gamma độc lập.

SCAR-X forward giữ nguyên:

```text
theta = threshold + beta*a_old
u     = leak*v_old + input_current + recurrent_current
z     = hard(u >= theta)
v_new = u - z*theta
a_new = rho*a_old + z
f_new = kappa*f_old + (1-kappa)*z
```

Đặt `ue=leak*e_v+pre`, `ze=psi*(ue-beta*e_a_old)`. Bảng dưới là sensitivity
post-reset, không phải thay đổi forward:

| Estimator | `e_v_new` | Adaptation trace |
|---|---|---|
| SCAR-X own-reset | `ue-theta*ze-z*beta*e_a_old` | `rho*e_a_old+ze` |
| SCAR-X stopped-reset | `ue` | `rho*e_a_old+ze` |
| E-prop chuyển đổi cho SCAR-X | `ue-z*beta*e_a_old` | `rho*e_a_old+ze` |

Dòng cuối cắt derivative qua **reset spike**, nhưng giữ derivative của adaptive
theta. Stopped-reset hiện tại bỏ cả hai thành phần, vì vậy không đồng nhất với
e-prop chuyển đổi khi beta khác0. Khi beta=0, adapter matched và stopped-reset
khớp bitwise qua nhiều cập nhật fixture.

Hai variant cố định trước DEV:

- `eprop_matched`: thay reset Jacobian như bảng; giữ hetero gamma.25→.92,
  sensory gate và tất cả thành phần khác của baseline.
- `eprop_aligned`: cùng Jacobian, bỏ sensory gate và dùng kappa thực của forward
  cho filter eligibility. Đây kiểm đồng thời hai thay đổi filter/gate; nếu thắng
  chưa được quy nguyên nhân duy nhất cho reset.

Cả hai giữ logistic surrogate SCAR-X để tránh thêm confound. Helper triangular
upstream chỉ phục vụ kiểm toán số học. Không gọi hai variant là bản chạy nguyên
gốc paper, không chuyển công bố accuracy/energy của paper sang H48 này.

## Kiểm toán chạy thật

Chạy tại worktree literature:

```powershell
python -X utf8 -m pytest -q -p no:cacheprovider tests/test_literature_eprop.py tests/test_literature_braintrace.py tests/test_literature_snapshots.py
```

**42 test PASS** ở lần kiểm reviewer (chỉ fixture kỹ thuật, không DEV):

- AST của đúng hàm upstream `compute_eligibility_traces` được thực thi nguyên
  vẹn bằng NumPy primitive shim; output e/epsilon_v/epsilon_a/psi khớp adapter
  scan ở tolerance1e-14 trên fixture no-refractory/no-diagonal-mask. Không phải
  đã chạy TensorFlow/autodiff hoặc toàn bộ numerical suites upstream.
- Scalar recurrence oracle, adaptive-threshold reset Jacobian finite difference
  của detached smooth graph, forward bitwise và đúng một `core.step` mỗi tick.
- Source readout/optimizer/RNG giữ nguyên khi cung cấp eligibility giống nhau;
  beta0 endpoint exact; khác biệt trace có thật khi beta>0.
- 12 tổ hợp model/condition giữ full state, RNG, mid-episode resume và cập nhật
  tiếp khớp byte; 3 cold subprocess resume thật cũng khớp SHA cuối.
- Rollback, policy/filter/mechanism hash, float64 và cross-kind/variant tamper
  đều bị từ chối; template không bị thay đổi khi loader thất bại.
- BrainTrace isolated full-surrogate oracle được kiểm bằng một recurrence
  dense4state Jacobian độc lập trên topology nhỏ có recurrent mixing; không
  dùng full Jacobian đó làm candidate hoặc tạo chất lượng H48 giả.

Mảng persistent của e-prop reference không tăng so MatchedLocalCredit: vẫn ba
eligibility matrices48x16float32. Việc kiểm finite arrays và snapshot có buffer
tạm; CPU/RSS/commit cần runner đo riêng, không suy từ số byte mảng.

## Kết luận tái sử dụng

`ALREADY_SOLVED_IN_LITERATURE`: factorization, ALIF traces, distinction giữa
exact và approximate learning signals, numerical verification strategy.
`REPRODUCED_IN_SCARX` ở mức kỹ thuật: pinned upstream trace scan và transformed
reset/filter adapters có tests và cold resume. `UNRESOLVED`: thắng matched
DEV/OOD, retention lâu dài, language/vision/general skills. Không coi adapter
mới là đóng góp khoa học chỉ vì khác source; cần kết quả và đối chứng tiếp theo.
