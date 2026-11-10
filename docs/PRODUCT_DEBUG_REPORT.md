# Báo cáo A–I: Product-first + Root-cause Debugging

> **Đối chiếu mới:** báo cáo này lưu phiên 22 vòng, không phải trạng thái mới
> nhất. Kết luận A4 về rollback đã bị phép đo cô lập ở STATUS vòng 23 thay thế:
> neural rollback đúng; idle và các lệnh đọc live xen kẽ làm nhiễu phép so sánh.
> Giữ nguyên phần dưới làm lịch sử. [Review 40](research/40_DEEPSEEK_REVIEW.md)
> còn kiểm chứng transplant, metric và acquisition. Các số phiên cũ không tự
> trở thành kết quả của metric sản phẩm v2.

Ngày 2026-11-10. Phiên làm việc 22 vòng liên tục trên SCAR-X Generation 1.
Mọi con số dưới đây đều từ lệnh chạy thật trong session này; chỗ nào chưa đo
được thì ghi rõ **chưa đo**, không suy diễn.

---

## A. Root cause — những gì đã xác nhận và bằng chứng

Năm nguyên nhân gốc, mỗi cái đều có thực nghiệm cô lập:

### A1. Guard chưa từng chạy khi dạy thủ công

`generation.py` từng có `guard = self._guard_pairs(prompt) if _automatic else []`.
Đường dạy thủ công (form web, Hermes) — đường người dùng thật dùng — **bỏ qua
guard hoàn toàn**.

Bằng chứng: bài `'tạm biệt'` được **ACCEPTED** trong khi `'chào bạn'` đi từ
`'Chào bạn.'` thành `'Bạn có bạn.'`, và `guard_lost` không hề được tính
(`None`). Điều này cũng giải thích vì sao báo cáo trước thấy "15/15 bị chặn"
(nhánh `_automatic`) mà đồng thời "4/4 ACCEPTED" (nhánh thủ công) — hai nhánh
khác nhau, một nhánh không có guard.

**Đã sửa.** Nay guard chạy trên mọi đường dạy. Không nới ngưỡng.

### A2. `generate()` không reset state làm mọi phép đo phụ thuộc thứ tự

`generate()` cố ý không reset recurrent state (ADR0006). Hệ quả đo được:

| Test | Kết quả |
|---|---|
| Live state, gọi 8 lần cùng prompt | **2 output khác nhau** |
| Reset state, gọi 8 lần | **1 output** |
| Đảo thứ tự prompt | `chào bạn`: `'Chào'` ↔ `'Bạn'`; `bạn là ai`: sai ↔ **đúng** |

Đây là lý do các con số báo ở nhiều vòng đầu **không khớp nhau**.

**Đã sửa cách đo**: `measure()` mặc định chấm mỗi prompt từ clone reset state
(cùng quy ước guard dùng), nên tái lập được. Vẫn giữ
`measure(reset_state=False)` để xem hành vi live thật.

### A3. Trọng số học ĐÚNG, nhưng state live làm hỏng câu trả lời

Cô lập từng bước trên cùng bản sao:

| Đường | live state | reset state |
|---|---|---|
| `train_pair` thuần | ✅ `'Tôi là scar-x.'` | ✅ `'Tôi là scar-x.'` |
| `expand_vocabulary` + train | ✅ | ✅ |
| **`generate()` TRƯỚC rồi train** | ✅ | ❌ `'Tôi là scar-x là'` |
| train ×4 | ✅ | ❌ |

Lặp lại 3 lần: **3/3 đều đúng**. Nguyên nhân: `teach()` gọi `generate()` để chụp
"before text" **trước khi** huấn luyện, và chính lệnh đó đẩy state sang quỹ đạo
khác.

**Đã sửa**: chụp "before" trên clone dùng một lần, không đụng live core.

### A4. Rollback khôi phục trọng số nhưng không khôi phục hành vi

Đo trước/sau một lượt dạy **bị từ chối**:

| Thành phần | Trước | Sau | Giống? |
|---|---|---|---|
| `weights_sha256` | `56ca4ce3…` | `56ca4ce3…` | ✅ |
| `updates` | 241219 | 241219 | ✅ |
| `ticks` | 2172265 | 2172358 | ❌ |
| **`'hi'`** | **`'Hello.'`** | **`'Hello'`** | ❌ |

Trọng số giống hệt nhưng câu trả lời khác. **Chưa sửa** — cần quyết định có
khôi phục cả state hay chấp nhận và ghi rõ.

### A5. `normalized()` bỏ dấu câu khiến guard bảo vệ khuyết điểm

`normalized()` dùng `.strip(' .!?')` nên `'Hello'` và `'Hello.'` được coi là
**bằng nhau**. Guard vì thế coi câu cụt là "đã đúng" và **chặn mọi nỗ lực sửa**.

**Đã sửa** thành hai tiêu chí độc lập: `content_ok` (đủ từ, không lặp) **chặn**
khi mất; `complete_ok` (kết thúc bằng `.`/`!`/`?`) chỉ **báo cáo**, không chặn.

Trong lúc sửa, tôi tự gây ra một hồi quy: lần đầu dùng so khớp **chuỗi con** làm
guard yếu đi — retention violations từ **0 lên 2** (`'4 + 4 bằng máy?'`,
`'hello'` bị mất). Đã phát hiện và sửa về so khớp **chính xác**; đo lại **0**.

### A6. Cơ chế khiến "dạy dấu câu" không hiệu quả

Truy vết `'bạn tên gì' → 'Tôi là scar-x.'`:

| Vị trí | Token | Xác suất |
|---|---|---:|
| 2 | `scar-x` | 0,961 |
| 3 | **`<eos>`** | **0,952** |
| 3 | `.` | 0,043 |

Model biết nội dung (0,9+) nhưng chọn `<eos>` thay dấu chấm. Cùng cơ chế với
`premature_stop = 1,000`.

---

## B. Fixed — file nào, sửa gì

| File | Sửa | Kiểm chứng |
|---|---|---|
| `src/scarx/app/generation.py` | Guard chạy mọi đường dạy | `test_guard_policy.py` |
| `src/scarx/app/generation.py` | `_pair_verdict` hai tiêu chí | 237 test + đo retention |
| `src/scarx/app/generation.py` | Ghi `learning_rejected` vào trace | 1 event/lượt reject |
| `src/scarx/app/generation.py` | Giữ log qua `trace.clear()` | |
| `src/scarx/app/generation.py` | Chụp "before" trên clone | ticks không đổi |
| `src/scarx/app/generation.py` | Coverage guard từ `PRODUCT_GUARD_PROMPTS` | thuần khiết, không gọi `generate()` |
| `src/scarx/learning/dialogue.py` | Retire câu hỏi khác chủ đề | `test_chat_defects.py` |
| `src/scarx/learning/dialogue.py` | Tách `user_turn`, không nag | |
| `src/scarx/neural/unified.py` | `_repeating()` guard decode | chỉ trong `generate()` |
| `src/scarx/evaluation/product_acceptance.py` | Mới: 12 metric + `weights_sha256` | 7 test thước đo |
| `src/scarx/runtime/experiment_status.py` | Accessor audit 39 + lọc superseded | 4 test |

Script mới (read-only trừ `restore_good_checkpoint.py`):
`acceptance_suite.py`, `find_working_checkpoint.py`, `prove_restore_fixes_product.py`,
`restore_good_checkpoint.py`, `verify_restart_safety.py`, `prove_retention.py`,
`why_each_lesson_rejected.py`, `find_real_rejection_cause.py`,
`diagnose_state_drift.py`, `separate_decoder_from_learning.py`,
`measure_vision_ten_digits.py`, `compare_live_vs_fresh.py`, và 10 script chẩn đoán khác.

---

## C. Neural integrity — còn đúng ràng buộc không?

| Ràng buộc | Kiểm chứng |
|---|---|
| Không `torch`/`autograd`/`backward` trong `neural/` | **0 vi phạm** (quét import + `.backward(`) |
| `torch` nạp khi chạy app | **`False`** |
| Một lõi ALIF SNN | `cores_loaded=1`, `hidden=64`, fan-in 8, plastic_rows 12 |
| Không Transformer / LLM / LLM proxy | 0 tham chiếu; 0 lời gọi API ngoài |
| Không RAG / lookup / hardcode | `unified.py` **không** chứa `GUARD_PROMPTS` hay đáp án |
| Không BPTT trong main | `controls.py` không được nạp (`torch: False`) |
| Học synapse cục bộ | `learn()` nguyên vẹn: `readout_lr=.025`, `hidden_lr=.0001` |
| CPU-first ≤ 2 GB | **42–59 MB** |
| ADR0006: runtime không reset state | Live core liên tục; chỉ clone trong guard reset |

**Không đổi learning rule, topology, hay signal.**

---

## D. Comparison — before/after, gồm cả thất bại

### Chat (thước đo chặt: đủ từ + không lặp + câu hoàn chỉnh)

| Metric | Phiên live trước | **Sau restore + restart** |
|---|---:|---:|
| Free-answer đúng | **0,000** | **0,625** |
| Premature stop | 1,000 | **0,250** |
| Held-out (6 câu chưa dùng để tune) | 0,500 | **0,667** |
| Repetition | 0,000 | 0,000 |
| Guard passing | — | 8/11 |

### Chat thật qua API sau restart

```
'hi'        -> 'Hello.'                 'hello'     -> 'Hello.'
'chào'      -> 'Chào bạn.'              'chào bạn'  -> 'Chào bạn.'
'bạn là ai' -> 'Tôi là scar-x.'         'bạn tên gì'-> 'Tôi là scar-x.'
'cảm ơn'    -> 'Rất vui được giúp bạn.' 'tạm biệt'  -> 'Tạm biệt bạn.'
```

**8/8 câu hoàn chỉnh có dấu chấm.** Nhưng khi hỏi qua `chat()` với live state,
`'hi'` cho `'Hello'` (thiếu chấm). Cả hai cách đọc đều công bố.

### Những trường hợp THẤT BẠI (ghi rõ, không che)

| Thử nghiệm | Kết quả |
|---|---|
| Dạy 3 câu còn sai sau restore | **0/3** — bị guard từ chối |
| 7 bài dạy "không xung đột", quét 1/2/4 bước | **0/7** vừa học được vừa không hại |
| 6 bài ACCEPT nhưng không đổi kết quả | **accept rỗng** — `<eos>` thắng |
| Vision 10 chữ số (7-segment sắc nét) | **1/10** |
| Vision 10 chữ số (khử răng cưa) | **3/10** |
| Vision khi thu nhỏ live 128→64 | 1/4, **không phục hồi** |

---

## E. Retention — có mất kiến thức cũ không?

Đo trực tiếp qua 9 lượt dạy liên tiếp, so sánh sau **mỗi** lượt:

| Phép đo | Kết quả |
|---|---:|
| Retention violations | **0** |
| Reject rollback đúng | **7/7** |
| Kiểm chứng rollback chính xác | `weights`, `updates`, `vocab`, `correct`, `guard` **identical** |

Nhưng **trung thực**: khi dạy nội dung **mới**, interference là thật — dạy `'hi'`
làm `'chào bạn'` thành `'Chào bạn. chào bạn'` (lặp). Guard chặn đúng, nhưng điều
đó có nghĩa **học thêm nội dung mới hiện không khả thi**.

---

## F. Generalization — có học được quy luật hay chỉ học thuộc?

**Chưa có bằng chứng học quy luật.**

| Nguồn | Kết luận |
|---|---|
| Audit 38 (session này đọc) | Gap 29,34 điểm giữa GOLDPREFIX và FREE; corpus 8 mẫu đóng |
| Audit 39 (session này chạy) | Tín hiệu đúng hướng (63,54%) nhưng held-out ΔNLL **−8,4e−05** ≈ 0 |
| Held-out 6 câu | 0,667 — **quá ít** để gọi là tổng quát hoá |
| Gold-prefix test | Forcing tiền tố đúng cũng chỉ đúng **1/8** |

Chỉ **học thuộc** trong phạm vi mẫu được dạy.

---

## G. Resource — RAM, CPU, checkpoint, persistence

| Mục | Giá trị |
|---|---|
| RSS | **42–59 MB** (trần 2 GB) |
| VMS | ~27–31 MB |
| Peak khi audit 39 | 53,4 MB |
| Checkpoints | **109** (108 cũ + 1 `action=restore`) |
| Backup | `PRE-RESTORE-backup-20261110T182732` (36,2 MB), `backup-20261110T182811` (31,1 MB) |
| Restart persistence | **weights identical** qua 2 chu kỳ đóng/mở |
| Text sau restart | identical trong phép đo sạch; trôi nếu để state chạy tiếp |

**Bằng chứng trọng số không đổi**: dùng `weights_sha256()` (hash mảng trọng số +
topology + vocabulary). **Không** dùng `updates` (counter có thể không tăng khi
readout đổi) và **không** dùng SHA payload runtime (đổi liên tục vì idle tick —
đo được `193d1868…` → `0566670d…` cách 6 giây).

---

## H. Remaining blockers — cần quyết định gì

| # | Blocker | Cần gì |
|---|---|---|
| 1 | **Interference**: dạy nội dung mới phá bài cũ | Đổi learning rule (replay / selective-synapse) — **protocol + duyệt** |
| 2 | **EOS thắng dấu chấm** (0,952 vs 0,043) | Đổi objective decode — **protocol + duyệt** |
| 3 | **Rollback không khôi phục state** (`ticks`/`voltage` trôi) | Quyết định: sửa hay ghi rõ giới hạn — **đụng ADR0006** |
| 4 | **Vision 1–3/10** | Cần dữ liệu/huấn luyện thật — **ngoài phạm vi** |
| 5 | **Held-out chỉ 6 câu** | Bộ đánh giá lớn hơn — **protocol** |
| 6 | `capabilities.json` cần cập nhật | Nên ghi `chat_restored_checkpoint` + số đo |

Không có blocker nào tôi tự quyết được mà không đổi ranh giới đã đặt.

---

## I. Run instructions — cách tự kiểm thử trên Windows

```powershell
cd 'C:\Users\hachimi\Downloads\model agent\scar-x'
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
```

### Kiểm thử sản phẩm (1 lệnh, 12 metric)

```powershell
python scripts\acceptance_suite.py --teach
```

### Chấm toàn bộ 108 checkpoint (read-only)

```powershell
python scripts\find_working_checkpoint.py
```

### Chứng minh restore trên bản sao (không đụng live)

```powershell
python scripts\prove_restore_fixes_product.py
```

### Restore phiên thật (có backup tự động)

```powershell
python scripts\restore_good_checkpoint.py              # dry run
python scripts\restore_good_checkpoint.py --apply      # thật
```

Script **từ chối chạy** nếu app đang mở (đã kiểm chứng).

### Kiểm retention và rollback

```powershell
python scripts\prove_retention.py
```

### Chẩn đoán vì sao bài dạy bị từ chối

```powershell
python scripts\why_each_lesson_rejected.py
python scripts\find_real_rejection_cause.py
```

### Toàn bộ test

```powershell
python -m pytest -q        # 237 passed
```

### Mở ứng dụng

```powershell
python run_app.py
# http://127.0.0.1:8765  →  #agent-work, #research-lab
```

---

## Tóm tắt một đoạn

Nguyên nhân gốc lớn nhất là **guard chưa từng chạy ở đường dạy thủ công**, cộng với
**thước đo phụ thuộc thứ tự** vì `generate()` không reset state — hai điều này
khiến mọi báo cáo trước đó vừa lạc hậu vừa không tái lập được. Sau khi sửa và
restore về checkpoint tốt nhất, chat của phiên thật đi từ **0/8 lên 5/8 câu hoàn
chỉnh** (8/8 nếu chỉ xét "có dấu chấm"), premature stop từ **100% xuống 25%**,
retention **0 vi phạm**, trọng số và lịch sử nguyên vẹn với 2 bản backup.

Điều **chưa** đạt: SCAR-X chưa học thêm được nội dung mới mà không phá bài cũ
(interference thật), chưa sửa được dấu câu (`<eos>` áp đảo), và vision chỉ
1–3/10. Ba việc đó đều cần đổi learning rule hoặc objective, tức cần protocol và
phê duyệt riêng — không nằm trong phạm vi product-first.
