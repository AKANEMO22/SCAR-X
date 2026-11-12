# R7 — CROSS-SNN MECHANISM TRANSFER SURVEY
## Bảng so sánh 11 cơ chế + đề xuất prototype

**Branch:** `research/deepseek/r7-causal-hidden` · **Base SHA:** `4f1c902b7948af1513dc127555bbd3ef80412f32`
**Production `src/scarx/neural/` modified:** no · **New runs:** 0

**Phần này là khảo sát + thiết kế, không phải kết quả thực nghiệm.** Không có PASS nào được tuyên bố.

---

## 0. Cảnh báo giấy phép — đọc trước

| nguồn | license | dùng được? |
|---|---|---|
| **SpikingLM** (`hamings1/SpikingLM`) | **`license: null`** (GitHub API, kiểm 2026-11-12) | **KHÔNG chép code.** Chỉ đọc nguyên lý từ README/paper. Không có giấy phép = không có quyền tái sử dụng. |
| SpikeLLM | repo chỉ có 1 dòng README, không có code | không có gì để chép |
| SpikeGPT | `README.md` không tồn tại ở `main` lẫn `master` (HTTP 404) | không xác minh được |
| SpikingBrain2.0 | README công khai, mô hình 5B | chỉ tham khảo |
| DECOLLE | Frontiers, CC-BY | nguyên lý công khai, tự cài đặt |
| e-prop | Nature Comms | nguyên lý công khai, tự cài đặt |
| Nengo-SPA | MIT (nengo-spa) | tham khảo toán học |

**Không dòng mã nào từ các nguồn trên được đưa vào SCAR-X.** Mọi cơ chế dưới đây là **tự cài đặt lại từ phương trình trong paper**, không port code.

---

## 1. Bảng so sánh 11 cơ chế/ý tưởng

Xếp hạng theo: khả năng sửa P1/P2 đã chứng minh · tương thích kiến trúc · chi phí · độ rõ nhân quả · rủi ro protected tasks · khác biệt với thuật toán đã STOP.

| # | Cơ chế | Nguồn | Sửa P1 (hidden rep)? | Sửa P2 (retention)? | Chi phí | Nhân quả rõ? | Rủi ro | Trạng thái trong SCAR-X | Hạng |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **e-prop / LSNN** (neuron-specific eligibility + learning signal) | Bellec 2020 | có lý thuyết | không | 1 trace/synapse | cao | **trung bình** | **ĐÃ STOP** — `STOP_DIRECT_READOUT_ELIGIBILITY_V1`, và tôi đã tự đo trace chính xác: cosine +0.9452 **vẫn zero gain** (R11) | **loại** |
| 2 | **DECOLLE local error qua ma trận G cố định** | Kaiser 2020 | **có** | không | O(N) | **cao** | thấp | **chưa thử** | **A** |
| 3 | **Distribution-aware Scaling** (learnable λ cho Q/K/V) | SpikingLM | có (dead-neuron/saturation) | không | 1 scalar/lớp | **cao** | **thấp** | **chưa thử** | **B** |
| 4 | **Benna–Fusi multi-timescale synaptic state** | Benna 2016 | không | **có** | **n_state × synapse** | cao | trung bình | **chưa thử trực tiếp**; nhưng "importance decay" + "metaplasticity" **đã chạy và fail** (303 trial) | **C** |
| 5 | **Synaptic Intelligence** (online importance) | Zenke 2017 | không | có | 1 scalar/synapse | cao | trung bình | **ĐÃ THỬ VÀ FAIL** — "SI tại readout" trong 303 trial, accuracy ~50% | **loại** |
| 6 | **Spike2Max attention** (max-subtraction + exp2) | SpikingLM | có (token competition) | không | thấp | cao | **cao** — cần attention, không có trong SCAR-X | không áp dụng được | **loại** |
| 7 | **Dual-Path Activation Coding** (FP8 + INT8 spiking) | SpikingBrain2.0 | không | không | thấp | thấp | thấp | inference-only, không phải learning | **loại** |
| 8 | **Saliency-based spiking** (chọn neuron theo saliency) | SpikeLLM | không rõ | không | thấp | thấp | trung bình | không có code để kiểm | **loại** |
| 9 | **Nengo-SPA circular convolution binding** | Eliasmith | **có** (binding) | không | O(N²) hoặc O(N log N) | cao | **cao** | **CẤM** — symbolic binding cài sẵn; brief mục B nói rõ không được dùng để trả lời thay neural core | **loại** |
| 10 | **VTB (vector-derived transformation binding)** | nengo-spa | có (binding sâu) | không | thấp hơn CC | cao | **cao** | như #9, và là phép đại số cài sẵn | **loại** |
| 11 | **Adaptive threshold / homeostatic scaling** | chung | có | không | 1 scalar/neuron | trung bình | thấp | **ĐÃ THỬ VÀ VOID** (R13, R13b) | **loại** |

### Vì sao 6 trong 11 bị loại ngay

- **#1 e-prop** — không chỉ vì brief nói đã STOP, mà vì **tôi đã đo trực tiếp** rằng ngay cả trace **chính xác về mặt đạo hàm** (cosine +0.9452 so với null −0.0005, kiểm bằng sai phân hữu hạn +0.9979) vẫn cho **zero representation gain** ở mọi eta. e-prop là một xấp xỉ của đúng tín hiệu đó. Nếu bản chính xác không giúp, bản xấp xỉ không thể giúp. **Đây là lập luận loại trừ, không phải ý kiến.**
- **#5 SI, #4 một phần** — SI đã chạy trong 303 trial và fail (accuracy ~50%, metaplasticity giảm forgetting nhưng accuracy sụp). Chạy lại là lặp lại thí nghiệm đã biết kết quả.
- **#6, #7, #8** — cần Transformer attention hoặc là inference-only; không phải local learning.
- **#9, #10** — brief mục B cấm rõ ràng.
- **#11** — đã VOID ở R13.

---

## 2. Ba ứng viên còn lại

### Ứng viên A — DECOLLE local error qua ma trận chiếu cố định

**Vì sao khác mọi thứ đã STOP.** SCAR-X hiện dùng **một scalar advantage** broadcast cho mọi synapse:

```
advantage = clip(p[target] - reward_mean, -1, 1)
delta     = hidden_lr * advantage * trace - anchor
```

`advantage` là **một số**. Mọi synapse nhận cùng một hệ số. e-prop (đã STOP) thay đổi **trace**, không thay đổi **bản chất tín hiệu** — nó vẫn nhân với một learning signal vô hướng.

DECOLLE thay đổi đúng chỗ đó: mỗi neuron nhận một **error riêng**, lấy từ một readout cục bộ:

```
error_i = Σ_k G_ik (Y_k - Ŷ_k)        # G cố định, ngẫu nhiên, không huấn luyện
ΔW_ij   = -η · error_i · σ'(U_i) · P_j
```

Trong đó `P_j` là trace tiền synapse (đã có trong SCAR-X), `σ'` là surrogate gradient (đã có). **Điểm mới duy nhất là `error_i` thay cho `advantage`.**

Đây là điều tôi **chưa từng thử** trong 42 vòng, và nó trả lời trực tiếp câu hỏi trung tâm: *"hidden có thể học nhưng learning signal sai?"* — nếu tín hiệu vô hướng là sai, tín hiệu theo neuron phải khác.

**Chi phí:** `G` là ma trận H×2 cố định = 96 số. Trace `P` đã tồn tại. **Không thêm bộ nhớ đáng kể.**

### Ứng viên B — Distribution-aware Scaling

**Vì sao.** README SpikingLM nói rõ hai trở ngại: *"gradient degradation from dead neurons"* và *"weak token selectivity"*. Cơ chế của họ cho vấn đề thứ nhất là **learnable scalar λ khởi tạo bằng 7** nhân vào output của projection. Ý tưởng: đưa activation vào vùng mà neuron bắn, thay vì kẹt dưới ngưỡng.

SCAR-X có **đúng triệu chứng đó**: R12 đo **1,7/48 neuron chết**, và R14 xác nhận lại sau khi sửa lỗi đo.

**Chuyển sang SCAR-X:**
```
v[t] = leak·v[t-1] + λ_in · (W_in x[t]) + λ_rec · (W_rec spikes[t-1])
λ_in, λ_rec  học online, khởi tạo 1.0
λ ← clip(λ + η_λ · (firing_rate - target_rate), λ_min, λ_max)
```

**Đây KHÔNG phải tăng neuron hay topology.** Chỉ là hai scalar nhân vào input hiện có. `fan_in=8`, `H=48` giữ nguyên.

**Rủi ro:** đây là homeostatic scaling, và R13 đã VOID HTA. **Khác biệt:** HTA chỉnh **threshold**; ứng viên này chỉnh **gain đầu vào**. Nhưng tôi phải nói thẳng: **hai cơ chế rất gần nhau, và R13 thất bại là một tiên nghiệm xấu.** Tôi xếp nó hạng B vì lý do đó, không phải vì nó hứa hẹn hơn.

### Ứng viên C — Benna–Fusi multi-timescale synapse

Mỗi synapse giữ `n` biến với hằng số thời gian log-space; tổng hợp cho một bộ nhớ dài hạn mà không cần task boundary.

**Vì sao xếp cuối:** chi phí bộ nhớ là **n × số synapse**. Với H48, `plastic_rows=12`, `fan_in=8`, đó là ~4600 synapse — nhân n=5 là 23k float, vẫn dưới 2GB, nên **khả thi về tài nguyên**. Nhưng "importance decay" và "metaplasticity" đã fail trong 303 trial, và Benna–Fusi về bản chất là một dạng metaplasticity. **Tiên nghiệm xấu thứ hai.**

---

## 3. Protocol DEV tối thiểu cho ứng viên A (hạng cao nhất)

**Câu hỏi:** tín hiệu lỗi **theo từng neuron** có tạo representation gain mà tín hiệu **vô hướng** không tạo được không?

**Thiết kế bắt buộc — mỗi arm chỉ đổi ĐÚNG MỘT yếu tố:**

| arm | tín hiệu học | kiểm soát cái gì |
|---|---|---|
| `frozen` | không có | sàn — hidden không đổi |
| `scalar` | `advantage` vô hướng (production) | baseline hiện tại |
| **`perneuron`** | `error_i` qua G cố định | **ứng viên** |
| `perneuron_zero` | `error_i × 0` | loại trừ khả năng "chỉ cần thêm update" |
| `perneuron_shuffled` | `error_i` hoán vị giữa neuron | loại trừ khả năng "chỉ cần thêm nhiễu" |

**Bốn arm kiểm soát là bắt buộc.** R42 (đang chạy) đã dạy rằng `learned` vs `frozen` một mình không đủ để kết luận.

**Đo:** accuracy trực tiếp, common-refit probe (ridge), đóng góp `w_in`/`w_rec` (zero-intervention), per-seed, CI paired. Seeds mới, chưa dùng.

**Falsifier:** nếu `perneuron` không vượt `scalar` ≥ 2pp trên probe **và** `perneuron_shuffled` không tệ hơn `perneuron` ≥ 2pp, cơ chế bị bác bỏ.

**Ngân sách:** 15 seed × 5 arm × 3 task — trong phạm vi DEV đã duyệt.

---

## 4. Những gì tôi KHÔNG làm

- **Không** chép code từ bất kỳ repo nào, đặc biệt SpikingLM (không license).
- **Không** mở lại e-prop/readout-eligibility, SI, metaplasticity, importance decay, HTA — đều đã STOP hoặc VOID **có bằng chứng**.
- **Không** dùng Nengo-SPA binding (brief cấm rõ).
- **Không** chạy formal confirmation, không đổi architecture, không tăng neuron.
- **Không** tuyên bố PASS cho bất kỳ cơ chế nào — **chưa có thí nghiệm nào chạy.**

---

## 5. Trạng thái trung thực

| hạng mục | trạng thái |
|---|---|
| Khảo sát nguồn | **xong** — 9 nguồn, license ghi rõ, 2 nguồn không truy cập được |
| Bảng so sánh | **xong** — 11 cơ chế, 6 loại có lý do |
| Ứng viên đề xuất | **3** (A/B/C), A xếp cao nhất |
| Prototype | **chưa** — chờ duyệt protocol |
| Bằng chứng thực nghiệm | **chưa có** |

**Mục tiêu 1 (hidden representation) vẫn CHƯA CHỨNG MINH** sau 42 vòng. Ứng viên A là lần đầu tiên tôi đề xuất thay đổi **bản chất** tín hiệu học thay vì trace hay learning rate — và nó chưa được chạy.
