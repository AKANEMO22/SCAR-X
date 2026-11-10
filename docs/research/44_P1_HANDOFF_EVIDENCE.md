# Bàn giao P1 — bằng chứng đã khóa cho Claude và Gemini

2026-11-10. Dành cho hai agent độc lập tiếp nhận phần nghiên cứu khó.
**Không phải** tuyên bố năng lực của SCAR-X.

- Claude: đề xuất thuật toán và kiến trúc **P1 (representation/encoding)**.
- Gemini: kiểm định dữ liệu và benchmark **độc lập**.

Người bàn giao **không được** tự triển khai đề xuất của hai agent này, không chạy
learner mới, không sửa encoder/tokenizer/vocabulary/learning rule khi chưa duyệt.

---

## 1. Giới hạn ĐÃ CHỨNG MINH (có số đo, tái lập được)

### 1.1 Mất thông tin màu — nhân quả, từng bước

Nguồn: `src/scarx/app/pilots.py :: PilotLab.pixels`

```python
gray = image.convert('L')                                 # <-- MÀU BỊ XOÁ
small = gray.resize((64,64))
pixels = np.asarray(gray.resize((8,8)),np.float32)/255*16  # 64 số
```

| Bước | Ảnh phân biệt | Cặp khác màu bị gộp |
|---|---:|---:|
| S0 — RGB gốc 128×128×3 | **36/36** | 0/18 |
| **S1 — sau `convert('L')`** | **18/36** | **18/18** |
| S2 — grayscale 64×64 | 18/36 | 18/18 |
| S3 — grayscale 8×8 (input model) | 18/36 | 18/18 |
| *S3_rgb8 — RGB 8×8 (tham chiếu trần)* | *36/36* | *0/18* |

**Kết luận:** mất màu tại `convert('L')`, *trước* resize. Resize 64→8 không mất thêm.

Tái lập: `python scripts\audit_color_loss_stages.py`

### 1.2 Trần thông tin của pipeline chữ số hiện tại

Với input 64 số grayscale: **18/36 scene phân biệt được**. 18 cặp chỉ khác màu
cho **cùng một vector input**.

Vì vậy câu hỏi về màu có trần **0,5000** — không learning rule nào vượt được.

| Kind | Trần 64-pixel |
|---|---:|
| single_color | **0,5000** |
| left_color | **0,5000** |
| object_count | 1,0000 |
| single_shape | 1,0000 |
| **Tổng** | **0,8333** (480/576) |

### 1.3 Trần tham chiếu RGB 8×8

Giữ màu ở 8×8 mỗi kênh (**192 số**) phân biệt **36/36** scene, trần **1,0000** cho
mọi kind.

**Đây chỉ là trần tham chiếu.** Không phải encoder được duyệt, không thêm model,
không phải kiến trúc đề xuất.

### 1.4 Vocabulary — hai đại lượng khác nhau

Code (`language.py:31`, `unified.py:24`) buộc `vocabulary[:2] == ['<eos>','<bos>']`
và `3 <= len(vocabulary) <= 256`.

| Đại lượng | Giá trị |
|---|---:|
| Token nội dung chat (distinct) | 360 |
| Token nội dung vision (distinct) | 35 |
| Giao | 9 |
| **Token nội dung khác nhau (union)** | **386** |
| `<eos>` + `<bos>` | 2 |
| **Vị trí vocabulary tối thiểu** | **388** |
| Cap trong code | 256 |
| **Thiếu theo vị trí** | **132** |
| Thiếu theo token nội dung | 130 |

`<eos>`/`<bos>` **không** nằm trong tập token nội dung (đã kiểm).

**Hệ quả:** mọi reference cần ít nhất một token core chưa có → decoder **không thể
sinh câu nào nguyên văn**, kể cả sau huấn luyện hoàn hảo.

### 1.5 Baseline "mù" của tập câu hỏi

Không nhìn ảnh, đoán giá trị phổ biến nhất mỗi câu hỏi:

| Số lớp | Số câu | Majority | Uniform |
|---|---:|---:|---:|
| 2 (binary) | 192 | **0,5000** | 0,5000 |
| 3 (three-class) | 384 | **0,3333** | 0,3333 |
| **Tổng** | 576 | **0,3889** | 0,3889 |

Majority = uniform vì mỗi câu hỏi cân bằng tuyệt đối. Đây là **optimal majority
baseline**, **không phải lỗi cần hạ xuống**. Mọi accuracy ≥ 38,89% có thể đạt mà
không cần nhìn ảnh.

---

## 2. Giới hạn CHƯA CHỨNG MINH (đừng suy diễn)

| Điều | Trạng thái |
|---|---|
| **Accuracy thực của một neural model trên 576 QA** | **CHƯA ĐO** |
| Hidden neuron có học được representation hữu ích (P1) | **CHƯA ĐO** |
| Learning rule có giữ kỹ năng cũ khi học mới (P2) | P0 pass, **pilot P2 chưa duyệt** |
| RGB descriptor có giúp model học tốt hơn không | **CHƯA ĐO** — chỉ là trần thông tin |
| Corpus 95+36+576 có dạy được normal chat | **CHƯA** — là development |
| 64 pixel có đủ cho chữ số viết tay thật | **CHƯA ĐO** |

**Không** được biến trần thông tin thành "model sẽ đạt X%".

---

## 3. Vì sao tăng dữ liệu KHÔNG giải được

Ba tầng chặn **độc lập**; thêm dữ liệu không chạm tầng nào:

1. **Thông tin đầu vào:** 18 cặp khác màu → cùng input. Thêm 1000 ảnh cùng loại
   vẫn cho cùng vector. Cần **đổi encoding**.
2. **Vocabulary:** 388 vị trí > cap 256. Thêm dữ liệu làm union **tăng**, tệ hơn.
3. **Baseline mù:** mọi accuracy phải vượt 38,89% mới chứng minh có nhìn ảnh.
   Thêm dữ liệu cân bằng không đổi baseline này.

Tăng dữ liệu **chỉ** giúp nếu cả ba tầng đã thông.

---

## 4. Điều kiện tối thiểu của một protocol P1

Protocol P1 phải có **đủ** các mục sau trước khi được duyệt:

1. **Câu hỏi nghiên cứu tách bạch:** P1 hỏi *"hidden neuron có học được
   representation hữu ích?"* — **không** phải *"readout có giữ kỹ năng?"* (P2).
2. **Encoding cụ thể:** mô tả chính xác vector đầu vào, số chiều, cách sinh từ
   pixel. Không được là "thêm feature màu" chung chung.
3. **Vẫn một lõi ALIF:** không CNN, không ViT, không BLIP, không Qwen-VL, không
   model vision thứ hai.
4. **Baseline đối chứng:**
   - Encoding hiện tại (64 số grayscale)
   - Encoding mới
   - Cùng checkpoint, cùng seed, cùng thứ tự, cùng số update
5. **Chỉ số phải vượt 38,89%** — nếu không, model không dùng pixel.
6. **Đo riêng 4 tầng:** dataset / input-information / hidden representation /
   readout retention. Không gộp.
7. **GO/STOP đăng ký trước**, gồm tiêu chí điện toán và thời gian.
8. **Không dùng final test để chọn tham số.**
9. **Ngân sách:** 1 thread, ≤2 GB, thời gian tường minh.
10. **Tương thích checkpoint:** nêu rõ encoding mới có làm hỏng weight hiện tại không.

---

## 5. Mã liên quan nhưng CHƯA ĐƯỢC SỬA

| File | Vai trò | Trạng thái |
|---|---|---|
| `src/scarx/app/pilots.py` | `PilotLab.pixels` — nơi mất màu | **KHÔNG sửa** |
| `src/scarx/neural/language.py` | vocabulary contract, cap 256 | **KHÔNG sửa** |
| `src/scarx/neural/unified.py` | core ALIF, `readout_lr`, `hidden_lr` | **KHÔNG sửa** |
| `src/scarx/app/generation.py` | guard, teach, restore | **KHÔNG sửa** |
| `artifacts/adaptive-curriculum-*.json` | dữ liệu gốc | **KHÔNG sửa** |
| `data/candidates/.../dataset.json` | 95 + 576 + 36 | **KHÔNG sửa** |

Mọi thay đổi cần protocol riêng và phê duyệt.

---

## 6. Tái lập

```powershell
cd 'C:\Users\hachimi\Downloads\model agent\scar-x'
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'

python scripts\verify_adaptive_grounding.py     # 95/576/36, SHA, 386 token
python scripts\audit_information_ceiling.py     # 3 trần độc lập
python scripts\audit_color_loss_stages.py       # mất màu từng bước
python -m pytest -q                             # 287 passed
```

Mọi script **read-only**. Không huấn luyện, không gọi model, không mạng.

---

## 7. Trạng thái hệ thống lúc bàn giao

| Mục | Giá trị |
|---|---|
| Neural weights | `updates=241129`, `hidden=64` — không đổi |
| Vocabulary | 24 token — không đổi |
| Topology | dense input + fan-in 8 + shared readout |
| `torch` trong core | `False` |
| RSS | 57,2 MB / 2 GB |
| Tests | **287 passed** |
| Hermes | không đụng |
| Dataset | không sửa |
