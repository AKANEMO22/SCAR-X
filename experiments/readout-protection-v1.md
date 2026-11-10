# Protocol A — Readout Protection (đăng ký trước)

**Trạng thái: chỉ là thiết kế. CHƯA được duyệt để code. Chưa đổi learning rule nào.**

Ngày đăng ký: 2026-11-10. Người đăng ký: phiên Codex.
Căn cứ: chẩn đoán round 24 + bằng chứng hardened round 26.

---

## 1. Giả thuyết

Readout hiện tại ghi đè lên các cột đã phục vụ cặp khác, vì mọi cặp dùng chung
không gian cột và `readout_lr` (0,025) lớn hơn `hidden_lr` (0,0001) **250 lần**.
Nếu giảm tốc độ cập nhật cho những cột **đang đóng góp** vào kỹ năng đã xác minh,
thì học cặp mới sẽ ít phá cặp cũ hơn.

### Bằng chứng hiện có (và giới hạn của nó)

| Quan sát | Số đo | Giới hạn |
|---|---|---|
| Interference tái tạo bằng transplant **chỉ readout** | 1 cặp / 4 bài | **mẫu rất nhỏ**, không đủ kết luận thống kê |
| Transplant `hidden_only` / `bias_only` không tái tạo | 0 cặp | ủng hộ readout là cơ chế |
| Hidden đóng băng vẫn quên | P2: 27–32 điểm | bằng chứng mạnh từ nghiên cứu trước |
| Head-only drop vs hidden-only | P2: 5,04đ vs 1,47đ / window | tương tác, không cộng thành tỷ lệ |

**Không** dùng LR ratio hay norm như bằng chứng nhân quả. Can thiệp transplant là
bằng chứng chính; nó chỉ cho 1 sự kiện.

---

## 2. Định nghĩa cơ chế A

### 2.1 Synapse nào được coi là quan trọng

Với readout, "synapse" là phần tử `W_out[token, neuron]` — cột `neuron` đóng góp
cho token đó. Importance phải suy từ **tín hiệu cục bộ đã có**, không từ held-out.

Đề xuất dùng đúng đại lượng mà learning rule đã dùng (không thêm module mới):

```
importance[token, neuron] = EMA của |error[token] * feature[neuron]|
```

trong đó `error` và `feature` là chính các mảng `learn()` đã tính. Đây là
`|Δw|` kỳ vọng của chính synapse đó — không cần forward mới, không cần nhãn.

EMA hệ số `ρ` (đăng ký trước, xem §4).

### 2.2 Cách giảm learning rate

```
lr_eff[token, neuron] = readout_lr * g(importance[token, neuron])
g = 1 / (1 + β * importance)      # giảm đơn điệu, không bao giờ về 0
```

- `g ∈ (0,1]`: **không có synapse nào bị đóng băng tuyệt đối**
- `β` là tham số duy nhất cần quét

### 2.3 Trả lời câu hỏi "có làm mọi cột bị đóng băng không"

Với công thức trên: **không**, vì `g > 0` luôn. Nhưng nếu `importance` tăng theo
số update, `g` tiến về 0 theo thời gian → **có nguy cơ đóng băng dần**.

Do đó phải đo trực tiếp:
- phân bố `g` sau chuỗi bài (min/median/max)
- số synapse có `g < 0,1` (coi là gần đóng băng)
- **nếu >80% synapse có `g < 0,1` thì STOP** vì cơ chế đã thành đóng băng

Đây là tiêu chí STOP **đăng ký trước**, không phải điều chỉnh sau.

---

## 3. Dữ liệu và cách chia

| Tập | Nội dung | Dùng để |
|---|---|---|
| **Train** | 5 cặp dạy, thứ tự cố định đăng ký | acquisition |
| **Dev** | 8 câu sản phẩm hiện có (`CASES`) | chọn `β` |
| **Held-out** | 6 câu `HELDOUT` chưa từng dùng để tune | **chỉ chấm cuối** |
| **Guard** | 11 cặp `_guard_pairs` | retention |

Thứ tự dạy (khóa trước):
`chào` → `cảm ơn` → `bạn khỏe không` → `tôi muốn dạy bạn` → `bạn làm được gì`

Không dùng held-out để chọn `β`, seed, hay ngưỡng.

---

## 4. Tham số đăng ký trước

| Tham số | Giá trị khóa |
|---|---|
| `ρ` (EMA importance) | **0,95** |
| `β` quét | **{0, 0.5, 1, 2, 4}** — chọn trên dev |
| Số bài dạy | **5** |
| `repeats` mỗi bài | **8** |
| Điểm xuất phát | checkpoint `b305479afb0e40c1ba5a04acd0139658` |
| Số điều kiện | **6** (xem §5) |
| Số lần lặp | **3** điểm xuất phát độc lập: checkpoint tốt + 2 checkpoint khác còn 4/4 |
| Tổng trial | 6 × 5 β-không-áp-dụng… xem ghi chú |

Ghi chú: baseline/global-LR/readout-only/random có **1 cấu hình mỗi điều kiện**
(β không áp dụng). Chỉ A có β. Tổng = (5 điều kiện đối chứng × 3 điểm xuất phát)
+ (5 β × 3 điểm xuất phát) = 15 + 15 = **30 trial**.

---

## 5. Đối chứng (đăng ký trước)

| # | Điều kiện | Mục đích |
|---|---|---|
| C1 | **Baseline**: rule hiện tại | mốc so sánh |
| C2 | **Giảm LR toàn cục** `readout_lr/4` | tách "học chậm" khỏi "bảo vệ chọn lọc" |
| C3 | **Readout-only** (hidden đóng băng) | tái lập P2 |
| C4 | **Bảo vệ ngẫu nhiên** cùng phân bố `g`, cùng ngân sách | tách lợi ích của *chọn lọc* khỏi *giảm cập nhật* |
| C5 | **A** với β đã chọn trên dev | giả thuyết chính |
| C6 | **A + C2** (vừa bảo vệ vừa giảm LR) | kiểm tương tác |

**C4 là đối chứng quan trọng nhất.** Nó trả lời: nếu giảm cập nhật ở *cột ngẫu
nhiên* cũng giúp, thì "chọn lọc" không có giá trị — chỉ là học chậm.

Cùng điều kiện bắt buộc: cùng checkpoint, cùng dữ liệu, cùng thứ tự, cùng số
update, cùng seed khởi tạo.

---

## 6. Phép đo bắt buộc

Mỗi trial ghi:

| # | Phép đo | Cách đo |
|---|---|---|
| 1 | **Newly acquired skills** | cặp nào **từ sai thành đúng** (không phải loss) |
| 2 | **Previously correct lost** | cặp nào **từ đúng thành sai** |
| 3 | **Final accuracy** | sau toàn chuỗi, trên dev |
| 4 | **Forgetting** | trung bình + tệ nhất trên guard |
| 5 | **Free generation** | `generate()` thật, không teacher-forced |
| 6 | **Hidden contribution** | so full vs readout-only |
| 7 | **Reset + continuous state** | cả hai, công bố riêng |
| 8 | **RAM / CPU / persistence** | RSS, giây, restart identity |
| 9 | **Chi phí importance** | đo riêng thời gian tính EMA |

---

## 7. Ngưỡng GO / STOP (khóa trước)

### GO nếu **đồng thời**

1. `newly_acquired ≥ 2` cặp (không chỉ 1), trên **cả 3** điểm xuất phát
2. `previously_correct_lost = 0` trên mọi trial
3. `forgetting_trung_bình ≤ 3 điểm` **và** `tệ nhất ≤ 5 điểm` (ngưỡng P2)
4. A tốt hơn **C4 (bảo vệ ngẫu nhiên)** với CI95% loại trừ 0
5. `<80%` synapse có `g < 0,1`

### STOP nếu bất kỳ

1. `newly_acquired = 0` trên mọi β → bảo vệ chặn học hoàn toàn
2. `forgetting` không tốt hơn baseline → không có tác dụng
3. A **không** tốt hơn C4 → chọn lọc vô ích, chỉ là học chậm
4. `>80%` synapse `g < 0,1` → đóng băng thực chất
5. `newly_acquired` chỉ đạt 1 cặp → mẫu quá nhỏ, không kết luận

### PIVOT nếu

A tốt hơn C4 nhưng `newly_acquired < 2`: cơ chế có tác dụng nhưng acquisition
chưa đủ → ghi nhận, **không** tự chuyển sang replay.

---

## 8. Dự đoán kết quả (đăng ký trước)

### Nếu giả thuyết ĐÚNG, tôi dự đoán

- A giảm forgetting so với baseline **nhưng** `newly_acquired` thấp (0–1 cặp)
- C4 (ngẫu nhiên) kém hơn A rõ rệt → chứng minh chọn lọc có giá trị
- `g` phân bố lệch: một số cột `g` thấp, phần lớn `g` cao
- **Kết quả ròng: PIVOT**, không GO

### Nếu giả thuyết SAI, tôi dự đoán

- A ≈ C4 → chỉ là học chậm, chọn lọc vô ích
- hoặc A chặn học hoàn toàn → STOP ở tiêu chí 1

### Dự đoán quan trọng nhất — rủi ro đã biết từ trước

**P2 (báo cáo 21) đã chỉ ra: "bảo vệ head không tự tạo feature thiếu".**
Cả offline-joint và refit head trên hidden features đều **dưới 65%**.

Nghĩa là: A có thể giảm quên **mà không tạo được năng lực mới**, vì hidden
representation thiếu thông tin. Nếu vậy, A sẽ **PIVOT hoặc STOP**, và vấn đề thật
nằm ở **representation**, không ở readout.

Tôi dự đoán **A KHÔNG đạt GO** với xác suất cao, và kết quả có giá trị là **loại
trừ readout là nút thắt**, chuyển trọng tâm sang representation.

---

## 9. Vì sao A khác các trial đã STOP

| Trial đã STOP | Vì sao khác |
|---|---|
| **P2 retention (21)** | P2 dùng replay/reservoir để giữ mẫu, **không** bảo vệ synapse theo importance tính từ tín hiệu cục bộ. P2 cũng không có đối chứng "bảo vệ ngẫu nhiên". |
| **P2 readout conditioning** | Chuẩn hóa đầu ra, không giảm LR theo importance. |
| **Word Bridge 36** | Khác bài (word sequence), dùng eligibility 35 + head riêng; A không thêm signal mới. |
| **Audit 39** | Chỉ đo hướng tín hiệu, không thử bảo vệ. |
| **Direct readout eligibility 35** | Đổi **signal** (readout error feedback). A giữ nguyên signal, chỉ đổi **tốc độ** cập nhật theo importance. |

**A không mở lại family đã STOP**: không replay, không signal mới, không topology,
không tăng model. Nó chỉ thêm một hệ số `g` vào bước cập nhật readout đã có.

---

## 10. Ranh giới

- Một lõi ALIF SNN, local synaptic learning, CPU-first ≤2 GB
- Không Transformer / pretrained LLM / LLM proxy / RAG / BPTT / autograd
- Không hardcode prompt; không module trả lời thay
- Không dùng held-out để tạo importance hoặc chọn tham số
- **Không sửa checkpoint live.** Mọi trial chạy trên bản sao
- Không đưa trọng số thử nghiệm vào Hermes
- Không dùng final held-out để chọn tham số
- Tối đa **2 vòng** điều chỉnh cho cùng giả thuyết (theo RESEARCH_PLAN 7.2)

---

## 11. Việc phải làm sau khi protocol được duyệt

1. Code prototype trên **nhánh riêng** (`scarx/readout-importance-v1`), không đụng main
2. Chạy P0: kiểm `importance` không NaN, `g ∈ (0,1]`, forward parity với baseline khi `β=0`
3. Chạy 30 trial theo §4
4. Báo cáo đầy đủ, kể cả trial thất bại
5. Nếu GO: trình riêng để duyệt tích hợp. Nếu STOP/PIVOT: ghi vào STATUS, không tự chuyển replay.

---

**Chờ người dùng duyệt protocol này trước khi viết bất kỳ dòng code prototype nào.**
