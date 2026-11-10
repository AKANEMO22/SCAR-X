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

## 3. Vocabulary — hai đại lượng khác nhau, cả hai đều đúng

Bàn giao ghi **388**, artifact ghi **386**. Ở bản trước tôi kết luận 388 là sai.
**Đó là kết luận của tôi sai.** Hai con số đo hai thứ khác nhau.

Code xác nhận (`src/scarx/neural/language.py:31`, `src/scarx/neural/unified.py:24`):

```python
if self.vocabulary[:2] != ['<eos>', '<bos>'] or not 3 <= len(vocabulary) <= 256
```

Hai token điều khiển `<eos>` và `<bos>` **luôn chiếm 2 vị trí đầu**, và chúng
**không nằm trong** tập token nội dung (`'<eos>' in union → False`,
`'<bos>' in union → False`).

| Đại lượng | Công thức | Giá trị |
|---|---|---:|
| Token nội dung — chat | distinct | **360** |
| Token nội dung — vision | distinct | **35** |
| Giao nhau | intersection | **9** |
| **Token nội dung khác nhau** | union = 360+35−9 | **386** |
| `<eos>` + `<bos>` | hằng số | **2** |
| **Vị trí vocabulary tối thiểu** | 386 + 2 | **388** ✅ |

**388 là đúng** khi nói về *vị trí vocabulary cần cấp*.
**386 là đúng** khi nói về *số token nội dung khác nhau*.

Tổng thô `360+35 = 395` **sai** vì đếm trùng 9 token; `395−9 = 386` là union.

```
vocabulary cap trong code        : 256   (trần vị trí)
vị trí cần                       : 388
thiếu theo vị trí vocabulary     : 132   (388 − 256)
content token cần                : 386
content token vượt cap           : 130   (386 − 256)
reference dài quá 24 token       : 0
```

**Dùng số nào:** khi nói "vocabulary còn thiếu bao nhiêu" phải dùng **132 vị trí**,
vì vocabulary chứa cả `<eos>`/`<bos>`. `130` chỉ là số **content token** vượt cap
và **không** phải số slot còn thiếu.

**Hệ quả:** 670/671 reference cần ít nhất một token core chưa có (xem §3b) →
**decoder không thể sinh câu nào trong số đó nguyên văn**, kể cả sau huấn luyện
hoàn hảo.

### 3b. `fully_known_reference_count = 1` — giải thích chính xác

Artifact ghi `fully_known_reference_count: 1` cho chat và `0` cho vision. Điều này
**không** mâu thuẫn với kết luận trên; nó chỉ ra **một ngoại lệ**.

Đã truy đúng reference đó (không xuất vocabulary):

| Trường | Giá trị |
|---|---|
| `id` | `chat_identity_vi_07` |
| Ngôn ngữ / family | `vi` / `identity` |
| `reference_kind` | `project_behavior` |
| Số token | **4** — tất cả đều có trong live vocabulary |
| Độ dài | 14 ký tự |

Phân bố số token thiếu:

| Chat | | Vision | |
|---|---:|---|---:|
| fully known | **1**/95 | fully known | **0**/576 |
| thiếu 1–21 token | 94/95 | thiếu 1–6 token | 576/576 |

**Phát biểu đúng:** *1 trong 671 reference* (một câu identity tiếng Việt rất ngắn)
được vocabulary hiện tại phủ đầy đủ; **670/671 còn lại đều thiếu ít nhất một token**.
Docs trước ghi "mọi reference đều thiếu" là **sai** và đã sửa.


**Lưu ý phương pháp:** số token "chưa biết" **không** được tính bằng
`386 − kích thước vocabulary hiện tại`, vì phép trừ kích thước không cho biết tập
nào giao tập nào. Muốn con số đó phải tính **giao** giữa tập token cần và tập
token đang có. Báo cáo này **không xuất vocabulary thật** (chỉ số lượng tổng hợp).


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

Đây **không phải lỗi cần hạ xuống**. Đây là **optimal majority baseline** của tập
câu hỏi hiện tại: nếu model không nhìn ảnh, cách tốt nhất là đoán giá trị phổ biến
nhất cho mỗi câu hỏi. Không chiến lược mù nào làm tốt hơn.

Chance baseline tách riêng theo số lớp đáp án:

| Số lớp đáp án | Số câu | Majority baseline | Uniform chance |
|---|---:|---:|---:|
| **2 lớp** (binary: đỏ/xanh, tròn/vuông…) | 192 | **0,5000** | 0,5000 |
| **3 lớp** (three-class: 1/2/3 vật thể) | 384 | **0,3333** | 0,3333 |
| **Tổng** | 576 | **0,3889** | 0,3889 |

Trong tập này majority **bằng đúng** uniform, vì mỗi câu hỏi có số đáp án cân
bằng tuyệt đối (36 câu × 3 lớp = 12 mỗi lớp; 36 câu × 2 lớp = 18 mỗi lớp).

Theo từng kind:

| Kind | Trần "mù" |
|---|---:|
| left_color | **0,5000** |
| single_color | **0,5000** |
| object_count | 0,3333 |
| single_shape | 0,3333 |
| **Tổng** | **0,3889** (224/576) |

**Cách đọc đúng:** 38,89% là **mức tối ưu của một bộ dự đoán chỉ nhìn câu hỏi**
trên đúng tập này — tức đoán giá trị phổ biến nhất cho mỗi câu hỏi, không dùng
pixel. Đây là **một mốc so sánh**, **không phải** bằng chứng neural vision, và
**không** phải khuyết điểm của dataset; một tập cân bằng là thuộc tính tốt.

**Điều KHÔNG được suy ra:** không được nói "mọi accuracy cao hơn 38,89% đều có
thể đạt bằng cách không nhìn ảnh". Sai, vì:

- Con số này đo một **bộ dự đoán cố định** (majority per question), không đo mọi
  chiến lược mù có thể có.
- Một model có thể vượt 38,89% bằng **học thuộc câu hỏi → đáp án** mà vẫn không
  nhìn ảnh (vì mỗi prompt lặp 36 lần với 1 trong 2–3 đáp án).
- Ngược lại, vượt 38,89% **không** tự động chứng minh model dùng pixel.

Vì vậy 38,89% là **sàn cần vượt**, và để chứng minh vision thật cần thêm phép đo
tách biệt (ví dụ: cùng câu hỏi, đổi ảnh, kiểm tra đáp án đổi theo). Bước đó **chưa
làm** và **chưa có số**.

Nếu muốn một bộ thử thách khó hơn, đó là **dataset version mới**, phải trình
thiết kế riêng; **không sửa tập hiện tại**.

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
