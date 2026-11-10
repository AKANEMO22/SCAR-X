# Hoàn thiện reference chat/vision — data only

2026-11-10. Dựng/kiểm dữ liệu theo
[protocol](../../experiments/adaptive-curriculum-grounding-v1.md).
**Không huấn luyện, không pilot P2, không thay model Hermes.**
Reference file **không phải** kiến thức đã học.

Run: `runs/20261110T142721Z-adaptive-curriculum-grounding-v1-a7c646`.
Artifact: `artifacts/adaptive-curriculum-grounding.json`.
Quyết định của run: **`DATA_GROUNDED_NOT_TRAINING_READY`**.

---

## 1. Độ phủ — đã xác minh lại độc lập

Mọi con số dưới đây tôi tự tính lại từ `dataset.json`, không chép từ artifact.

| Hạng mục | Artifact | **Tôi xác minh** | Khớp |
|---|---:|---:|---|
| Chat reference | 95 | **95** | ✅ |
| Vision QA | 576 | **576** | ✅ |
| Ảnh RGB sở hữu | 36 | **36** | ✅ |
| Scene | 36 | **36** | ✅ |
| SHA-256 dataset | `1b914509…` | **`1b914509…`** | ✅ |

Phân bố:

| Chat theo family | | Chat theo ngôn ngữ | |
|---|---:|---|---:|
| bye | 16 | en | 48 |
| capability | 16 | vi | 47 |
| greet | 16 | | |
| identity | 15 | | |
| learning | 16 | | |
| thanks | 16 | | |

| Vision theo kind | | |
|---|---:|---|
| object_count | 288 | |
| single_color | 96 | |
| single_shape | 96 | |
| left_color | 96 | |

`semantic_target` cân bằng hoàn hảo (mỗi giá trị 16–96, không lệch).

---

## 2. Nguồn nhãn và tính nhất quán

- **Chat**: 95 reference do dự án tự viết, review theo `family`, có rubric
  (`exact_reference_required: false`, `unsupported_capability_claims_allowed: false`).
  `label_basis: "Projectauthored source-reviewed reference, never teacher/model output"`.
  6 nhóm `fact_sources` (identity, limited_abilities, learning_tab, guarded_updates,
  not_self_labels, persistence) đều trỏ tới file + anchor trong repo, kèm SHA-256.
- **Vision**: nhãn primitive sinh từ manifest, **đối chiếu oracle độc lập** đọc
  connected-components / màu / bounding-box / row-width trên pixel thật.
  **`pixel_manifest_disagreements: 0/36`** — tôi xác minh lại, khớp.
- **Không rò rỉ annotation**: kiểm `input` của từng row — **0/576** vision và
  **0/95** chat chứa key lạ ngoài `prompt`/`image_file`.
- **`automatic_learning_allowed: false`** trên **cả 671 row**, và ở cấp dataset.

---

## 3. Vocabulary — con số bàn giao có sai lệch

Bàn giao ghi **388 token**. Artifact ghi **386**. Tôi tính lại: **386 đúng**.

| Cách tính | Kết quả |
|---|---:|
| Token chat (distinct) | **360** |
| Token vision (distinct) | **35** |
| Tổng thô 360 + 35 | 395 |
| Token chung (overlap) | **9** |
| **Union thật** | **386** |

`388` không khớp bất kỳ cách tính nào (không phải 386, không phải 395). Ghi nhận
sai lệch này; **kết luận định tính không đổi**: 386 > cap 256.

```
vocabulary cap trong code      : 256
vocabulary live hiện tại       :  24
token cần                      : 386
vượt cap                       : 130 token
token model CHƯA có            : 362/386
reference dài quá 24 token     : 0  (giới hạn độ dài không bị vượt)
```

**Hệ quả:** mọi reference đều cần ít nhất một token core chưa có →
**decoder không thể sinh câu nào nguyên văn**, kể cả sau huấn luyện hoàn hảo.

---

## 4. Va chạm thông tin khi nén RGB → 64 pixel

Đo bằng chính `PilotLab.pixels` hiện hành (chỉ tiền xử lý, không gọi core):

```
36 ảnh RGB 128×128  →  chỉ 18 input 64-pixel phân biệt
18 nhóm collision, mỗi nhóm đúng 2 ảnh
```

Mỗi nhóm là **cùng geometry, khác màu** → màu bị xoá hoàn toàn.

### Bước nào xoá màu — cô lập từng bước

Nguồn: `src/scarx/app/pilots.py :: PilotLab.pixels`

```python
gray = image.convert('L')                                   # <-- MÀU BỊ XOÁ TẠI ĐÂY
small = gray.resize((64,64))
pixels = np.asarray(gray.resize((8,8)),np.float32)/255*16    # 8×8 = 64 số
```

Đo trên 36 ảnh thật, từng bước:

| Bước | Ảnh phân biệt được | Cặp chỉ khác màu bị gộp |
|---|---:|---:|
| S0 — RGB gốc 128×128×3 | **36/36** | 0/18 |
| **S1 — sau `convert('L')`** | **18/36** | **18/18** |
| S2 — grayscale 64×64 | 18/36 | 18/18 |
| S3 — grayscale 8×8 (input model) | 18/36 | 18/18 |
| *S3_rgb8 — RGB 8×8 (tham chiếu trần)* | *36/36* | *0/18* |

**Kết luận nhân quả:** mất màu xảy ra tại **`convert('L')`**, *trước* mọi bước
resize. Resize 64→8 **không** làm mất thêm (18 → 18).

### Tách "thiếu thông tin đầu vào" khỏi "neuron chưa học"

| Đo | Ảnh phân biệt |
|---|---:|
| Input thật (grayscale 8×8 = 64 số) | **18/36** |
| Tham chiếu giữ màu (RGB 8×8 = 192 số) | **36/36** |

18 cặp chỉ khác màu cho **cùng một input**. Đây là **trần thông tin đầu vào**:
không learning rule nào tách được điều không tồn tại trong số liệu nhận vào.

`S3_rgb8` **chỉ là tham chiếu trần**, không phải encoder đã duyệt, không thêm model.


---

## 5. Cái gì KHÔNG THỂ giải dù huấn luyện thêm

Đây là phần quan trọng nhất. Ba giới hạn **độc lập**, không gộp thành một nguyên nhân.

### Tầng 1 — Thiết kế câu hỏi (dataset)

Nếu chỉ nhìn câu hỏi mà không nhìn ảnh, đoán giá trị phổ biến nhất đã được:

| Kind | Trần "mù" |
|---|---:|
| left_color | **0,5000** |
| single_color | **0,5000** |
| object_count | 0,3333 |
| single_shape | 0,3333 |
| **Tổng** | **0,3889** (224/576) |

### Tầng 2 — Tiền xử lý ảnh (mất thông tin)

| Kind | Trần với 64 pixel | Mất |
|---|---:|---|
| **single_color** | **0,5000** | **50%** |
| **left_color** | **0,5000** | **50%** |
| object_count | 1,0000 | 0 |
| single_shape | 1,0000 | 0 |
| **Tổng** | **0,8333** (480/576) | **96 câu không thể giải** |

Câu hỏi về **màu** không thể trả lời: hai ảnh khác màu cho **cùng một input**.

### Tầng 3 — Decoder / vocabulary

386 token cần, cap 256, live vocab 24 → không câu nào sinh được nguyên văn.

### Phân biệt rõ bốn loại lỗi

| Loại | Bằng chứng | Cách chữa (khác nhau) |
|---|---|---|
| **Dataset** | trần mù 38,9% | đổi thiết kế câu hỏi, thêm biến thể |
| **Representation (input)** | trần 64px 83,3%, mất 50% câu màu | giữ màu trong encoding — **cần duyệt** |
| **Decoder/vocabulary** | 362/386 token chưa có | đổi tokenizer — **cần duyệt** |
| **Learning rule** | *chưa đo ở bước này* | P1/P2 — **cần duyệt** |

Tăng lượng dữ liệu **không** chữa được tầng 2 hay tầng 3.

---

## 6. Tương thích với core/encoder hiện tại

| Mục | Trạng thái |
|---|---|
| Chat reference vừa độ dài (≤24 token) | ✅ 0 vượt |
| Vision reference vừa độ dài | ✅ max 8 token |
| Token đã có trong vocab live | ❌ 24/386 |
| Vượt cap 256 | ❌ 386 > 256 |
| API vision hiện tại | `digits_only: true` — **không nhận RGB scene** |
| `automatic_learning_allowed` | **false** toàn bộ |

**Kết luận:** dữ liệu **nhất quán và đúng nhãn**, nhưng **không tương thích** với
decoder/encoder hiện tại. Đúng như artifact kết luận: `DATA_GROUNDED_NOT_TRAINING_READY`.

---

## 7. Trọng số và phiên đang chạy

`live_weights_unchanged_during_build: true`. Không huấn luyện, không gọi core.
Model Hermes giữ nguyên.

---

## 8. Giới hạn của báo cáo này

- Chưa đo **learning rule** — không huấn luyện nên không có số.
- Trần "mù" và trần 64-pixel là **phân tích thông tin**, không phải năng lực SCAR-X.
- Corpus 95 + 36 + 576 là **development**, không phải normal chat hay general vision.
- Không tuyên bố kiến thức đã vào synapse.
