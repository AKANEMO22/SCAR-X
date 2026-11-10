# Trạng thái SCAR-X

## Hoàn tất đối chiếu DeepSeek — review 40

[Review40](research/40_DEEPSEEK_REVIEW.md), master1.30 và
`artifacts/deepseek-review-delivery.json` ghi đầy đủ thay đổi/bằng chứng.
**258/258 tests qua**, gồm rollback/RNG/checkpoint và CPU/native parity;
JavaScript syntax và bảng localhost đã kiểm. Bản sửa đã nạp vào server.

Giữ sửa guard thủ công/before-clone/rollback isolation của DeepSeek. Sửa lỗi
transplant nhóm AFTER, fallback clone lỗi chạm live, snapshot SQLite/WAL,
metric substring, acquisition bị đồng nhất ACCEPTED và trace cursor sau reject.
Metric canonical v2 tách khỏi v1; sáu paraphrase cũ là development đã xem.

24 can thiệp trên 3 trạng thái ×4 bài ×2 nhánh, mỗi nhánh6bước: chỉ snapshot
hiện tại, bài hi, hai nhánh full/readout mất1cặp đã biết từ bước5. Chỉ w_out
AFTER đủ tái tạo lỗi này; hidden/bias riêng không tái tạo. Không suy rộng thành
hidden=0 hoặc mọi forgetting do readout; checkpoint không là seed độc lập.
Same24outputs chấm v1/v2 cùng14pass, không nhận đây là cải thiện model.

Acceptance trên copy:5/8canonical,3/6development;2/6 giao dịch giữ, chỉ1/3target
vốn chưa đúng chuyển thành đúng và còn đúng sau chuỗi. Không mất8probe cũ;
không là bằng chứng solvedP2/20context/skillgeneralization. Cả RSS và commit
đều đo dưới2GB ở các lượt nhỏ. Main không học/restore trong review; trước/sau
restart weights/topology/vocabulary/encoder giữ nguyên,64ALIF đã cấp.

**PIVOT learning mechanism/no scale** giữ nguyên. Readout Protection mới vẫn
CHƯA DUYỆT, không code/chạy. Các vòng lịch sử phía dưới giữ nguyên; nhận định
rollback hỏng A4 được kiểm tra cô lập mới thay thế. Norm/LR và một bài không đủ
cho kết luận nhân quả “hidden gần như 0” của vòng24.

## Vòng 24: interference là READOUT OVERWRITE, không phải hidden

Chẩn đoán cơ chế theo đúng ưu tiên 4 của người dùng (readout interference,
hidden contribution, từng bước cập nhật). Dạy `'hi' -> 'Hello.'` từng bước, theo
dõi cặp nào mất và trọng số nào đổi:

| Chế độ | Bước mất `'chào bạn'` | `\|Δw_out\|` | `\|Δw_in\|` | `\|Δw_rec\|` |
|---|---:|---:|---:|---:|
| Full local | **5** | 0,392 | 0,00161 | 0,00093 |
| **Readout only** (hidden đóng băng) | **4** | **0,519** | 0 | 0 |

**`w_out` thay đổi lớn hơn `w_in` 244 lần và `w_rec` 422 lần.**

Hai kết luận có bằng chứng:

1. **Readout-only còn tệ hơn** — mất sớm hơn một bước và `Δw_out` lớn hơn. Vậy
   interference **không** đến từ hidden representation.
2. **Hidden plasticity đóng góp gần như bằng 0** vào cả việc học lẫn việc phá.

Nói cách khác: readout học rất nhanh (`readout_lr=.025`) trong khi hidden học rất
chậm (`hidden_lr=.0001`) — chênh **250 lần**. Readout ghi đè lên chính các cột nó
đã dùng cho cặp khác, và hidden không kịp hình thành biểu diễn tách biệt để giúp.

Khớp với: audit 39 (tín hiệu đúng hướng nhưng held-out ≈ 0), `full_local` ≡
`readout_only` ở round 13, và "hidden contribution: không đo được" ở vòng trước.

**Chưa sửa.** Đây là đặc tính của learning rule hiện tại; thay đổi `readout_lr`,
`hidden_lr`, selective-synapse hay replay đều là **đổi learning rule** — cần
protocol và phê duyệt riêng, đúng như người dùng yêu cầu.

## Vòng 23: rollback ĐÃ ĐÚNG — không sửa; và guard đã chặt hơn

### Mục 2 (rollback toàn bộ state): KHÔNG có bug

Thí nghiệm cô lập, `background=False` (không idle thread), không `generate()` xen
vào, từ chối được tiêm xác định qua `_online_microstep`:

| Thành phần | Kết quả |
|---|---|
| `w_in`, `w_rec`, `w_out`, `bias` | **identical** |
| `stable_in/rec/out/bias` | **identical** |
| `voltage`, `adaptation`, `spikes` | **identical** |
| `pre`, `post`, `features` | **identical** |
| `elig_in`, `elig_rec`, `rows` | **identical** |
| `indices` (topology) | **identical** |
| `vocabulary` | **identical** |
| `ticks`, `updates` | **identical** |
| `capacity_rng`, `app_rng` | **identical** |
| `probe`, `replay_cursor`, `memories` | **identical** |
| `weights_sha256` | `26e597a4b8373807` → `26e597a4b8373807` |
| Hành vi sinh câu (4 prompt) | **identical 4/4** |

Chỉ `sequence` (bộ đếm trace) và `dialogue` (đếm `rejected_updates`) thay đổi —
**bookkeeping sự kiện**, không phải trạng thái neural.

**Kết luận: rollback khôi phục chính xác toàn bộ neural state. Không sửa gì**,
đúng chỉ dẫn *"Nếu rollback đã đúng, không sửa vô ích."*

### Vì sao trước đây tưởng rollback hỏng

Đo lại: `fresh` (reset clone) **ổn định** qua mọi lần; `live` **trôi**:

```
fresh('hi') trước : 'Hello'
live('hi')  lần 1 : 'Hello.'
live('hi')  lần 2 : 'Hello'
live('hi')  lần 3 : 'Hello'
fresh('hi') sau   : 'Hello'
```

Sự "trôi" đến từ **`generate()` trên live state**, đúng ADR0006, **không** phải
rollback. Đây là lần thứ 3 một phép đo của tôi bị nhiễu bởi chính state.

### Mục 1 (guard): ba lỗi đã sửa, guard CHẶT HƠN

| Lỗi | Trước | Sau |
|---|---|---|
| `complete_ok` không cần content | `'Lời.'` → complete=**True** | complete=**False** |
| Target ngắn là chuỗi con | `'bạn tên gì'` lưu `'scar-x'`, không bao giờ khớp | `'Tôi là SCAR-X.'` khớp đúng |
| Dashboard lệch guard | `guard_report` bỏ dấu câu | dùng chung `_pair_verdict` |

`content_ok` so **chính xác** (`norm==tgt_norm`), không substring. Threshold
`if metrics['guard_lost']: raise` **nguyên vẹn**. Câu chứa đáp án + thừa vẫn bị
chặn (`'Hello world.'` vs `'Hello.'` → `content=False`).

## Vòng 22: audit guard phát hiện 3 lỗi

Sau restore (5/8), đã thử **7 bài dạy không xung đột** (prompt chưa đúng, đáp án
ngắn, quét 1/2/4 bước). Kết quả: **0/7 học được mà không gây hại**.

Nhưng phát hiện quan trọng hơn: **6 bài được ACCEPT mà không có tác dụng**
(`target=False`). Truy vết một bài:

```
dạy 'bạn tên gì' -> 'Tôi là scar-x.'   (ACCEPT, loss 0,669 -> 0,057)
kết quả vẫn: 'Tôi là scar-x'            (thiếu dấu chấm)
```

Xác suất tại chỗ dừng:

| Vị trí | Token | Xác suất |
|---|---|---:|
| 0 | `tôi` | 0,906 |
| 1 | `là` | 0,934 |
| 2 | `scar-x` | 0,961 |
| 3 | **`<eos>`** | **0,952** |
| 3 | `.` | 0,043 |

**Model biết nội dung đúng (0,9+) nhưng chọn `<eos>` thay dấu chấm.** Đây là cùng
một cơ chế với `premature_stop`.

Và `_pair_verdict` xử lý đúng như đã thiết kế nhưng **hệ quả là guard không sửa
được**: `content_ok=True`, `complete_ok=False` → bài dạy được chấp nhận (vì nội
dung không đổi) nhưng **không tạo ra dấu chấm**. Guard ghi nhận thiếu sót mà
không có quyền chặn.

Nói cách khác: **dạy dấu câu không hiệu quả vì EOS đã thắng áp đảo**, và guard
đúng khi không chặn (không mất nội dung) nhưng cũng không thể sửa.

### Kết luận cho interference

| Thử nghiệm | Kết quả |
|---|---|
| Bài xung đột (`hi`, `cảm ơn`) | Bị chặn — **interference thật** |
| Bài không xung đột | **ACCEPT nhưng không đổi kết quả** |
| Quét bước 1/2/4 | Không bài nào vừa học được vừa không hại |

Cả hai đường đều bế tắc trong phạm vi **không đổi learning rule**:
- Muốn học nội dung mới → phá bài cũ (interference)
- Muốn sửa hình thức → EOS thắng, bài dạy không tạo được dấu chấm

Đây là hai vấn đề **khác nhau** cần hai hướng khác nhau, đều nằm ngoài quyền
product-first.

## Vòng 20: ĐÃ RESTORE PHIÊN THẬT — chat từ 0/8 lên 5/8

Người dùng đã duyệt. Quy trình thực hiện đúng thứ tự yêu cầu:

1. **Backup trước**: `local/PRE-RESTORE-backup-20261110T182732` (36,2 MB, 4 file)
2. Kiểm chứng restart an toàn trên **bản sao** với mã mới: 2 chu kỳ đóng/mở,
   `weights_sha256` và texts **identical** ở cả hai
3. Dừng app (PID 12436, đang chạy mã cũ từ 15:41), xác nhận port 8765 free
4. Restore `b305479afb0e40c1ba5a04acd0139658` — backup thứ hai tự động
   `local/backup-20261110T182811` (31,1 MB)
5. Khởi động lại với mã mới

**Kết quả đo trên phiên THẬT (không phải bản sao):**

| Metric | Trước restore | **Sau restore + restart** |
|---|---:|---:|
| Free-answer đúng (hoàn chỉnh) | 0,000 | **0,625** |
| Premature stop | 1,000 | **0,250** |
| Held-out | 0,500 | **0,667** |
| Guard passing | — | 8/11 |
| Repetition | — | 0,000 |
| RSS | — | 42,3 MB (dưới 2 GB) |

Chat thật qua API `/v1/chat/completions`:

```
'hi'        -> 'Hello.'                 'hello'      -> 'Hello.'
'chào'      -> 'Chào bạn.'              'chào bạn'   -> 'Chào bạn.'
'bạn là ai' -> 'Tôi là scar-x.'         'bạn tên gì' -> 'Tôi là scar-x.'
'cảm ơn'    -> 'Rất vui được giúp bạn.' 'tạm biệt'   -> 'Tạm biệt bạn.'
```

**8/8 câu trả lời hoàn chỉnh, có dấu chấm.** (Thước đo chấm 5/8 vì ba câu được
đánh giá theo `require_all` chặt hơn — `'Rất vui được giúp bạn.'` không chứa
`'không'`, và `'Tôi là scar-x.'` đúng nhưng `'hi'` thiếu so với kỳ vọng riêng.
Cả hai cách đọc đều công bố ở trên, không chọn cách có lợi.)

**Trọng số sau restore:** `26e597a4b8373807`, `updates=241129`, `hidden=64`,
`active=48`. Không đổi learning rule/topology.

**Rollback:** copy `local/PRE-RESTORE-backup-20261110T182732` đè lại
`local/generation1`. Lịch sử checkpoint cũ **không bị xóa** (108 checkpoint
vẫn còn nguyên).

## Vòng 18: guard đang BẢO VỆ khuyết điểm, không chỉ bảo vệ kỹ năng

Sau restore (5/8), ba câu còn sai đều **gần đúng**: `'Hello'` thiếu dấu chấm,
`'Rất vui được giúp bạn.'` thiếu dấu chấm, `'Tôi là'` bị cụt. Đã thử dạy cả ba —
**cả 3 đều bị guard từ chối**, 0 thay đổi.

Hỏi thẳng guard xem nó nghĩ gì về từng cặp:

| Prompt | Target lưu | Model sinh | Guard nói |
|---|---|---|---|
| `'hi'` | `'Hello.'` | `'Hello'` | **PASS (equal)** |
| `'cảm ơn'` | `'Rất vui được giúp bạn.'` | `'Rất vui được giúp bạn.'` | **PASS (equal)** |
| `'bạn tên gì'` | `'scar-x'` | `'Tôi là'` | fail |

Vì `normalized()` **bỏ dấu câu cuối**, `'Hello'` và `'Hello.'` được coi là **bằng
nhau**. Hệ quả:

1. Guard **bảo vệ dạng cụt** và từ chối mọi bài dạy sửa nó.
2. Sản phẩm **không thể cải thiện** ba câu này chừng nào guard còn bật.
3. Thước đo sản phẩm (đã sửa) lại coi `'Hello'` là **sai** → hai định nghĩa
   "đúng" xung đột, đúng như round 15 kết luận.

Đây là **cùng một nguyên nhân** với round 15 nhưng có hệ quả nặng hơn: không chỉ
là bỏ sót, mà là **chủ động chặn việc sửa**.

### Cần người dùng quyết (không tự sửa được)

Sửa `normalized()` là **đổi định nghĩa bảo vệ** — ngoài phạm vi product-first:

1. **Giữ dấu câu trong so sánh** → guard bảo vệ đúng câu hoàn chỉnh; nhưng sẽ từ
   chối nhiều bài hơn nữa (hiện đã chỉ nhận 16,7%).
2. **Giữ nguyên** và ghi rõ guard chỉ bảo vệ nội dung từ, không bảo vệ hình thức.
3. **Tách hai mức**: bảo vệ nội dung (chặn) + cảnh báo hình thức (không chặn).

## Vòng 17: chứng minh RESTORE sửa được sản phẩm — 0/8 lên 5/8

Đã chứng minh end-to-end trên **bản sao** của phiên thật (không ghi gì vào live):

| Metric | Trước restore | **Sau restore** |
|---|---:|---:|
| Free-answer đúng (hoàn chỉnh) | 0,000 | **0,625** |
| Premature stop | 1,000 | **0,250** |
| `weights_sha256` | `ebf05a78…` | `26e597a4…` |

Câu trả lời sau restore:

```
'hello'     -> 'Hello.'                    ✅
'chào'      -> 'Chào bạn.'                 ✅
'chào bạn'  -> 'Chào bạn.'                 ✅
'bạn là ai' -> 'Tôi là scar-x.'            ✅
'tạm biệt'  -> 'Tạm biệt bạn.'             ✅
'hi'        -> 'Hello'                     (thiếu dấu chấm)
'bạn tên gì'-> 'Tôi là'
'cảm ơn'    -> 'Rất vui được giúp bạn.'    (thiếu dấu chấm)
```

**Đích restore:** checkpoint `b305479afb0e40c1ba5a04acd0139658`
(`2026-11-09T07:00:26`). API: `Generation.restore(id)`.

### Trọng số đã hỏng thật, không phải topology

| Trạng thái | Điểm |
|---|---:|
| Checkpoint tốt, hidden 64 | **4/4** |
| Lớn lên 96 | **4/4** |
| Lớn lên **128** (bằng live) | **4/4** |
| Live hiện tại (hidden 128) | 1/4 |
| Live **thu nhỏ** về 64 | 1/4 |

Thu nhỏ **không** phục hồi → trọng số đã bị phá, không phải topology. Đây là lý do
restore là cách sửa đúng, còn đổi kích thước thì không.

### Cách tự chạy

```powershell
cd 'C:\Users\hachimi\Downloads\model agent\scar-x'
python scripts\find_working_checkpoint.py          # chấm cả 108 checkpoint
python scripts\prove_restore_fixes_product.py      # chứng minh restore trên bản sao
```

Cả hai chỉ đọc; restore trong script chạy trên thư mục tạm.

## Vòng 16: tìm được checkpoint còn trả lời đúng — có đường khôi phục cụ thể

Chấm toàn bộ **108 checkpoint** trong phiên live (đọc read-only, chấm 4 câu mẫu
từ reset state):

| Kết quả | Số checkpoint |
|---|---:|
| **4/4 đúng hoàn chỉnh** | **10** |
| 1/4 (đã hỏng) | 98 |

**Mốc hỏng xác định được:** checkpoint tốt cuối là `b305479afb0e`
(`2026-11-09T07:00:26`); checkpoint hỏng đầu là `2026-11-09T07:09:44` — cách
**~9 phút**. Sau mốc đó toàn bộ session hỏng.

**Tăng capacity KHÔNG phải nguyên nhân.** Checkpoint `capacity-grow`
(`5b54a081ec0c`, 04:40:13, hidden 64) vẫn đạt **4/4 đúng**. Các checkpoint tốt
đều ở `hidden=64, active=48`. Vậy growth trong sạch, đúng như round 14 kết luận.

**Checkpoint tốt nhất còn lại:**

```
id      : b305479afb0e
created : 2026-11-09T07:00:26
hidden  : 64   active: 48   updates: 241129
  'hi'        -> 'Hello.'
  'chào bạn'  -> 'Chào bạn.'
  'bạn là ai' -> 'Tôi là scar-x.'
  'tạm biệt'  -> 'Tạm biệt bạn.'
```

Đây là **đích khôi phục cụ thể** nếu người dùng muốn lấy lại chat hoạt động —
nhưng restore là thao tác trên phiên thật nên cần người dùng quyết định.

Công cụ: `scripts/find_working_checkpoint.py` (read-only). Kết quả:
`local/checkpoint-quality.json`.

**Lỗi đo tự bắt được (lần 5):** lần chạy đầu báo `scored: 0` với mọi checkpoint.
Nguyên nhân: cột `payload` chứa **archive session** (`PK…model.zip`), không phải
blob model thuần, nên `UnifiedCore.loads` trực tiếp luôn thất bại. Đã sửa để mở
đúng như `Generation._unpack`. Sau khi sửa: **108/108 chấm được**.

## Vòng 15: guard và sản phẩm dùng hai định nghĩa "đúng" khác nhau

Truy vết từng bước từ trọng số shipped (241123 updates) cho thấy mỗi bài dạy được
chấp nhận vẫn bào mòn câu trả lời cũ:

| Bước | Kết quả | `hi` | `tạm biệt` |
|---|---|---|---|
| shipped | — | `'Hello.'` | `'Tạm biệt bạn.'` |
| 1. `bạn khỏe không` | ACCEPT | mất `'.'` | mất `'bạn'` |
| 2. `tôi muốn dạy bạn` | ACCEPT | vẫn hỏng | vẫn hỏng |
| 3. `chào` | ACCEPT | vẫn hỏng | **hồi phục** |

Không có "một bước phá tất cả" — hư hại **tích lũy**.

### Nguyên nhân: guard bỏ dấu câu, sản phẩm thì không

`normalized()` định nghĩa `' '.join(tokenize(text)).strip(' .!?')`, nên guard coi
`'Hello'` và `'Hello.'` là **giống nhau**. Đo trực tiếp:

```
AFTER lesson: 'hi'  guard(reset)='Hello'  guard_pass=True   live='Hello.'
```

Guard báo **PASS** trong khi câu trả lời đã mất dấu chấm — đúng khuyết điểm
`premature_stop`. Vì vậy:

- Guard **không thể** bảo vệ câu hoàn chỉnh, vì nó không phân biệt hoàn chỉnh với cụt.
- Thước đo sản phẩm (đã sửa) coi `'Hello'` là **sai**, nên báo `premature_stop = 1,000`
  trong khi guard báo mọi thứ ổn.

Đây là **hai định nghĩa "đúng" khác nhau** trong cùng một hệ thống. Không phải
ngưỡng, không phải coverage — mà là **ngữ nghĩa so sánh**.

Ngoài ra `'chào bạn'` cho ra `'Chào bạn. chào bạn'` trong guard reset và
`'Chào bạn. bạn'` ở live: cùng trọng số, hai state, hai kết quả khác nhau.

### Cần người dùng quyết

Sửa `normalized()` cho guard phân biệt câu cụt là **thay đổi định nghĩa bảo vệ** —
ngoài phạm vi product-first. Ba lựa chọn:

1. Guard so khớp **nghiêm ngặt** (giữ dấu câu) → bảo vệ được câu hoàn chỉnh, nhưng
   sẽ từ chối nhiều bài dạy hơn nữa (hiện đã chỉ nhận 16,7%).
2. Giữ nguyên, và **ghi rõ** trong docs rằng guard chỉ bảo vệ nội dung từ, không bảo
   vệ hình thức câu.
3. Tách hai chế độ: guard nội dung + cảnh báo hình thức riêng.

## Vòng 14: trọng số SHIPPED trả lời ĐÚNG — session live đã bị làm hỏng

Đây là phát hiện quan trọng nhất của cả đợt. Đo trực tiếp hai session:

| Session | updates | hidden | `chào bạn` | `bạn là ai` |
|---|---:|---:|---|---|
| **Shipped** (trọng số kèm gói) | 241123 | 48 | `'Chào bạn.'` ✅ | `'Tôi là scar-x.'` ✅ |
| **Live** (`local/generation1`) | 241363 | 128 | `'Chào'` ❌ | `'Tôi là scar-x là'` ❌ |

Chỉ cách nhau **240 lượt cập nhật**, nhưng shipped trả lời **đúng hoàn chỉnh** cả
bốn câu mẫu (`hi`, `chào bạn`, `bạn là ai`, `tạm biệt`), còn live đã hỏng.

**Không phải do tăng capacity.** Đã kiểm: lớn shipped từ 48 → 64 → 96 → **128**
mà **không huấn luyện** thì câu trả lời vẫn đúng ở mọi mức. Vậy growth trong sạch.

### Nguyên nhân gốc #4: rollback khôi phục TRỌNG SỐ nhưng không khôi phục HÀNH VI

Đo trước/sau một lượt dạy **bị từ chối**:

| Thành phần | Trước | Sau | Giống? |
|---|---|---|---|
| `weights` | `56ca4ce3…` | `56ca4ce3…` | ✅ |
| `updates` | 241219 | 241219 | ✅ |
| `ticks` | 2172265 | 2172358 | ❌ |
| `voltage` | khác | khác | ❌ |
| **`'hi'`** | **`'Hello.'`** | **`'Hello'`** | ❌ |

**Trọng số giống hệt, `updates` không đổi, nhưng câu trả lời đã khác.** Nghĩa là
một bài dạy **bị từ chối** vẫn làm hỏng hành vi người dùng thấy, trong khi guard
báo "rollback sạch".

Đây là lý do `premature_stop = 1,000` và vì sao dấu chấm biến mất: state sóng
lan sang một quỹ đạo khác, và readout rất nhạy với state.

### Hai việc phải làm (đều cần người dùng duyệt)

1. **Khôi phục cả state khi rollback**, không chỉ trọng số — hoặc chấp nhận rằng
   rollback chỉ bảo vệ trọng số và ghi rõ trong tài liệu.
2. **Session live đang chạy mã cũ** (khởi động 15:41, mã sửa 17:31). Cần restart
   để nạp guard thật; đây là phiên thật của người dùng nên phải hỏi trước.

## Vòng root-cause 4: tìm ra guard CHƯA TỪNG chạy khi dạy thủ công — 2026-11-10

Người dùng chốt hướng Product-first + Root-cause. Đã trả lời 7 câu hỏi xác minh
bằng thực nghiệm trên **bản sao**.

**Về bằng chứng trọng số không đổi:** không dùng SHA của toàn payload runtime —
đo được nó **đổi liên tục** (`193d1868…` rồi `0566670d…` chỉ cách 6 giây) vì idle
vẫn tick và ghi state. `updates` cũng không đủ. Bằng chứng đúng là
`weights_sha256()` trong `evaluation/product_acceptance.py`: hash chỉ các mảng
trọng số + `indices` + topology + vocabulary. Mọi thí nghiệm dưới đây chạy trên
**bản sao** (`copy_session`), live session không bị mở để ghi.

### Nguyên nhân gốc lớn nhất (đã sửa)

`generation.py` dòng 289 cũ: **`guard = self._guard_pairs(prompt) if _automatic else []`**

Guard **chỉ chạy cho bài học tự động (idle)**. Đường dạy thủ công — form web và
Hermes, tức đường người dùng thật sự dùng — **bỏ qua guard hoàn toàn**. Hệ quả đo
được: bài `'tạm biệt'` được **ACCEPTED** trong khi `'chào bạn'` đi từ
`'Chào bạn.'` sang `'Bạn có bạn.'`, và `guard_lost` không hề được tính
(`guard_lost=None`). Điều này giải thích luôn vì sao báo cáo trước thấy "15/15
bị chặn" (nhánh `_automatic`) mà đồng thời "4/4 ACCEPTED" (nhánh thủ công):
**hai nhánh khác nhau, một nhánh không có guard**.

Đã sửa: guard chạy trên **mọi** đường dạy. Không nới ngưỡng. Cập nhật
`docs/OBSERVING.md` và test `test_passive_trace` để policy mới minh bạch.

### Log quan sát và chặn retry (mục tiêu #4)

**Hai bug quan sát được đã sửa** (phát sinh từ chính thay đổi guard ở trên):

1. Lượt dạy bị guard từ chối trả `rolled_back=None` và **không ghi gì vào trace**,
   nên trên localhost một bài bị từ chối **không thể phân biệt** với bài chưa
   từng chạy. Nay ghi event `learning_rejected` với `accepted=False`,
   `rolled_back=True`, `weight_delta` toàn 0 và lý do rõ ràng. Đã kiểm: trace có
   1 event, `weights_sha256` khôi phục đúng.
2. `except BaseException: self.trace.clear()` xoá luôn bản ghi vừa thêm. Nay giữ
   lại tối đa 4 event `learning_rejected` qua rollback.

**Chặn retry vô hạn: đã có sẵn và đúng, không cần sửa.** Đo 12 lượt liên tiếp:
dừng ở **lượt 2**, chuyển `needs_help`, `remaining=0`,
`stopped_identical_proposal=True`. Cơ chế chống lặp lại đề xuất bị từ chối y hệt
hoạt động đúng.

Thêm `tests/test_guard_policy.py` (4 test) khoá lại: guard chạy ở đường thủ công,
reject ghi `rolled_back` + weight_delta 0, reject xuất hiện trong trace, và retry
dừng sau 2 lần từ chối giống nhau. Tổng **230 test qua**.



Đo trên corpus có sẵn (`data/dialogue/generation1.json`, 8 family đều có đáp án
đầy đủ), so từng bước với vector xác suất đầy đủ:

| Prompt | Muốn | Sinh ra | Dừng ở | `<eos>` |
|---|---|---|---:|---:|
| `chào` | `Chào bạn.` | `chào` | p1 | 0,874 |
| `tạm biệt` | `Tạm biệt bạn.` | `tạm biệt` | p2 | 0,979 |
| `cảm ơn` | `Rất vui được giúp bạn.` | `rất vui được giúp` | p4 | 0,908 |
| `bạn làm được gì` | `Tôi đang học trả lời.` | `lời đang học trả lời.` | p6 | 1,000 |
| `tôi muốn dạy bạn` | `Bạn có thể dạy tôi.` | `bạn có thể` | p3 | 0,696 |

**Kết luận: 6/8 là lỗi LEARNING, không phải decoder.** Readout **tự chọn**
`<eos>` với xác suất 0,87–1,00 trong khi token đúng chỉ 0,01–0,30. Không decoder
nào cứu được khi chính readout muốn dừng. Chỉ 1/8 (`bạn khỏe không`) là nghi
decoder (lặp `hoạt động` không dừng).

**Và forcing tiền tố đúng cũng không cứu:** cho model đúng tiền tố, nó chỉ đồng ý
`1+0+0`, `1+1+0+0`, `1+1+1+1+0` … tức vài token đầu rồi lệch. Tổng: free 1/8,
gold-prefix 1/8. Vậy **ánh xạ prompt→answer yếu ngay cả khi có tiền tố**, không
chỉ là exposure gap.

**Lỗi đo tự bắt được (lần 2):** script đầu tiên in cột "gold-prefix" luôn bằng
đáp án corpus vì đó là token **bị ép**, rồi kết luận sai là "gold-prefix giúp".
Đã tách rõ: cột text chỉ là token ép; chỉ cột `+`/`0` và số argmax mới nói model
có đồng ý hay không. Sau khi sửa, kết luận đảo lại thành đúng.

Điều này khớp audit 38 (gap 29,34đ) và audit 39 (held-out ΔNLL ≈ −8,4e−05): vấn
đề gốc nằm ở **biểu diễn**, không ở decoder. Vì vậy **không sửa bằng decoder**.



`python scripts/acceptance_suite.py --teach` đo đủ 12 metric trên bản sao:

| Metric | Giá trị ĐÚNG |
|---|---:|
| 1 Free-answer accuracy | **0,000** |
| 2 Held-out generalization | 0,500 |
| 3 Learning acquisition | 0,167 (1/6) |
| 4 Retention | **0,000** (mẫu số bằng 0 vì chưa câu nào đúng) |
| 5 Guard acceptance | 0,167 |
| 6 Guard regression | 0 |
| 7 Guard passing | 4/13 |
| 8 Premature stop | **1,000** |
| 9 Repetition | 0,000 |
| 10 RSS / VMS | 44,6 MB / 30,0 MB (dưới 2 GB) |
| 11 Restart persistence | **True** (theo `weights_sha256`) |
| 11b Text identical sau restart | False — working state trôi, không phải mất học |
| 12 Bằng chứng trọng số | `weights_sha256`, không dùng `updates` |

### ĐÍNH CHÍNH: các số 0,500 / 4-8 / Retention 1,000 báo trước đó là SAI

Suốt các vòng trước STATUS ghi free-answer 0,500 và retention 1,000. **Sai.**
Thước đo `product_acceptance` khi đó quá lỏng, có ba lỗi:

1. `expect='scar-x'` dùng so khớp **chuỗi con**, nên `'Tôi là scar-x là'` được
   tính ĐÚNG dù cụt giữa câu và lặp.
2. `_cycles()` chỉ bắt lặp **khối liền kề y hệt**, không bắt kiểu `scar-x là`.
3. Không kiểm tra câu **hoàn chỉnh**, nên `'Hello'` (thiếu dấu chấm) được tính
   đúng.

Nay `correct` đòi đồng thời: đủ mọi từ bắt buộc (`require_all`) **và** không lặp
**và** kết thúc bằng `.`/`!`/`?` (hoặc khớp đúng câu tham chiếu). Kết quả thật:
**0/8 câu hoàn chỉnh** — khớp với `premature_stop = 1,000` và với README đã công
bố "chưa có general chat".

Thước đo lỏng làm hỏng chính mục đích của nó: nó che đúng khuyết điểm mà nó tồn
tại để phơi ra.




Đo 8 bài × 3 mức bước (4/8/16), trên bản sao:

| Phép đo | Kết quả |
|---|---:|
| Tỷ lệ chấp nhận | **3/24 = 12,5%** |
| Bài được nhận mà vẫn mất kiến thức cũ | **0** |
| Rollback đúng khi bị từ chối | **21/21** (weights SHA khớp lại) |

Vậy guard đang bảo vệ đúng, nhưng **interference chưa được giải quyết**: học bài
mới gần như luôn phá bài cũ. Đây là câu trả lời cho ưu tiên #1: **chưa đạt**.
Giảm số bước (adaptive step) **không** cứu được — đã thử 4/8/16, tất cả đều mất
cùng cặp guard.

### Sửa: quan sát không được làm nhiễu thứ được quan sát

`teach()` chụp "before text" bằng cách chạy `generate()` **trên live core**, và
chính lệnh đó đẩy recurrent state sang quỹ đạo khác. Đã đổi sang chụp trên
**bản clone dùng một lần** (`UnifiedCore.loads(self.model.dumps())`), giữ nguyên
tính liên tục của session theo ADR0006.

**Rollback đã được kiểm chứng là chính xác tuyệt đối.** So sánh trước/sau một
lượt bị từ chối: `weights` identical, `updates` identical, `vocab` identical,
`correct` identical, `guard` identical. Không có gì bị mất.

**Hai lỗi trong script retention của chính tôi, đã sửa:**
1. Báo `rollback_incomplete` cho **mọi** lượt reject vì so với baseline **đầu
   tiên** thay vì trạng thái ngay trước lượt đó — trong khi các lượt ACCEPT ở
   giữa đã thay đổi baseline một cách hợp lệ.
2. Cộng cả chênh lệch `ticks` (vốn tăng liên tục theo ADR0006) vào vi phạm.

Sau khi sửa: **0 vi phạm retention trên 9 lượt**, 7/7 reject rollback sạch.

### NGUYÊN NHÂN GỐC #3: trọng số học ĐÚNG, nhưng state live làm hỏng câu trả lời

Cô lập từng bước trên cùng một bản sao, đọc từ hai trạng thái:

| Đường | live state | reset state |
|---|---|---|
| A. `train_pair` thuần | ✅ `Tôi là scar-x.` | ✅ `Tôi là scar-x.` |
| B. `expand_vocabulary` + train | ✅ đúng | ✅ đúng |
| D. train, không expand | ✅ đúng | ✅ đúng |
| **C. `generate()` TRƯỚC rồi train** | ✅ đúng | ❌ **`Tôi là scar-x là`** |
| F. train ×4 | ✅ đúng | ❌ sai |

Lặp lại A ba lần: **3/3 đều đúng** — không phải may mắn.

**Cơ chế:** `teach()` gọi `generate()` để chụp "before text" **trước khi** huấn luyện.
Lệnh đó đẩy recurrent state sang một quỹ đạo khác, và **chính nó** làm phép đo
sau đó sai — không phải trọng số.

Sau một lần `teach()` thật:

| Đọc từ | Kết quả | Đúng? |
|---|---|---|
| reset clone | `'Tôi là scar-x.'` | **ĐÚNG** |
| live state | `'Tôi là scar-x là'` | sai |

**Kết luận:** trọng số **đã học đúng**. Câu trả lời hỏng là do **trạng thái suy
diễn**, không phải do learning. Điều này khớp audit 39 (tín hiệu đúng hướng,
held-out ≈ 0) và khớp `premature_stop = 1,000`.

Đây là hướng cải thiện khả thi nhất còn lại — và nó **không cần** đổi kiến trúc
hay learning rule, chỉ cần xử lý state trước khi sinh. Nhưng nó **đụng ADR0006**
("runtime never reset_state"), nên cần người dùng duyệt.

### NGUYÊN NHÂN GỐC #2: thước đo phụ thuộc thứ tự vì `generate()` không reset state

`generate()` **cố ý không reset** recurrent state (ADR0006: một state liên tục
cho cả phiên). Đúng cho hội thoại, nhưng làm **mọi phép đo không tái lập được**:

| Test | Kết quả |
|---|---|
| Live state, gọi 8 lần cùng prompt | **2 output khác nhau** (lần 1 khác lần 2–8) |
| Reset state, gọi 8 lần | **1 output** — ổn định |
| Đảo thứ tự prompt | `chào bạn`: `'Chào'` ↔ `'Bạn'`; `bạn là ai`: sai ↔ **đúng** |

Đây giải thích vì sao các con số tôi báo ở nhiều vòng **không khớp nhau**: mỗi
lần chạy, state vào khác nhau.

Đã sửa `measure()`: mặc định chấm **mỗi prompt từ clone reset state** (cùng quy
ước guard đang dùng), nên tái lập được — đo 2 lần liên tiếp cho cùng kết quả
`0,000`. Vẫn giữ `measure(reset_state=False)` để quan sát hành vi **live** mà
người dùng thật sự gặp (phụ thuộc thứ tự).

**Phát hiện kèm theo:** một `train_pair` **thật sự sửa được** câu trả lời —
`tôi là scar-x là scar-x` → `tôi là scar-x .` (đúng) — nhưng chỉ đọc đúng khi
chấm từ state sạch. Nghĩa là model **có** học được; vấn đề nằm ở **trạng thái
suy diễn**, không phải trọng số.

### Vision: đo lại đầy đủ 10 chữ số — chỉ 1/10 (sắc nét) và 3/10 (khử răng cưa)

Trước đây chỉ có **một mẫu** (nét đứng → số 1, conf 0,647) và nó được báo như
"vision đúng". Một mẫu không phân biệt được classifier thật với hàm hằng. Đã đo
lại toàn bộ 0–9 qua đúng đường sản phẩm, hai kiểu ảnh:

| Cách vẽ | Accuracy | Ghi chú |
|---|---:|---|
| 7-segment 8×8 sắc nét | **1/10** | chỉ đúng số 5 |
| 7-segment vẽ lớn rồi downsample bilinear | **3/10** | giống pipeline thật hơn |

Nhầm lẫn: `0→2`, `1→9`, `8→7`, và nhiều số cùng về `5`. Readout **có** phản ứng
với nội dung (không phải hằng số), nhưng phân loại kém.

**Đính chính:** số `89,17%` trong tài liệu là của **V0.2 LAB** (model khác, đã
retire theo ADR0006), không phải G1. Không được trích nó cho G1.

Trọng số digit của G1 **có** được huấn luyện (480/1280 phần tử `w_out` khác 0,
norm 25,09), nên đây không phải "chưa train" mà là **chất lượng thấp**. Muốn
vision tốt hơn cần dữ liệu/huấn luyện thật — ngoài phạm vi product-first.

### Hidden contribution — readout đang làm hết

So cùng một bài, `full_local` (hidden plastic) vs `readout_only` (hidden đóng
băng): **kết quả giống hệt nhau**, cùng số câu đúng, cùng `emits_target`. Vậy
phần cải thiện hiện tại **giải thích được bằng readout**, không phải biểu diễn
ẩn. Khớp với audit 39 (held-out ΔNLL ≈ −8,4e−05).

### Đã trả lời 7 câu hỏi

1. **15/15 vs guard_lost=0?** Khác nhánh: `_automatic` có guard (15/15 chặn),
   thủ công **không có guard**. Không phải mâu thuẫn.
2. **155 rejections?** Là **lượt thử cập nhật**, không phải bài học độc lập. Một
   bài có thể đốt 32 lượt; chỉ 10 bài được giữ.
3. **`updates` đủ chứng minh?** **Không.** `updates` là counter trong payload;
   thay readout/vocabulary không tăng nó. Nay dùng `weights_sha256()` trên chính
   mảng trọng số + topology + vocabulary.
4. **Câu đúng có từ neural core thật?** Có: `UnifiedCore.generate → word_step →
   probabilities(argmax)`, không target/lookup/retrieval.
5. **Guard bảo vệ gì?** Bảo vệ cặp đang PASS. Nhưng trước đây nó **không chạy**
   ở đường thủ công, và vẫn chấm trên **clone reset state** khác state sống.
6. **Readout ghi đè / vocab shift?** Có: vocabulary mở rộng đổi input dimension
   và làm hỏng cặp guard cũ (`hi → 'Hello.'` mất sau khi dạy bài khác).
7. **`_repeating()` cắt nhầm?** Có thể, nếu câu hợp lệ lặp y hệt một khối ≤3 từ.
   Đo trên live bench: **0/8** lần bị cắt, nên chưa phải nguyên nhân hiện tại.

### Còn lại, cần người dùng duyệt

- Guard chấm trên **clone reset** trong khi product trả lời bằng **state sống**;
  đo được 1/12 cặp bất đồng (`'4 + 4'`: guard `'8'` PASS, live `'8 là scar-x.'`).
- Truncation: `<eos>` học quá sớm (p=0,958 ở `chào`).
- Held-out 6 câu quá ít để gọi là tổng quát hoá.

## Vòng sản phẩm: chat sửa được 3 lỗi, nhưng guard chặn toàn bộ việc dạy — 2026-11-10

Sau yêu cầu làm chat/vision "ổn" trong khuôn khổ ràng buộc, đã đo bằng thước lặp
lại được `scripts/bench_product_chat.py` (8 câu cố định, chạy trên **bản sao**
phiên, không huấn luyện, không seed, không phải benchmark nghiên cứu).

**Đã sửa — 3 lỗi thật, có test hồi quy (`tests/test_chat_defects.py`):**

1. Câu hỏi tồn đọng của prompt **khác** chiếm phiên: hỏi `chào bạn` lại trả về
   câu hỏi cũ về «ê». Nay câu hỏi khác chủ đề bị retire, giữ đúng "1 câu mở mỗi
   phiên" của ADR0009.
2. Vòng lặp từ vô hạn: `Tôi là scar-x là scar-x…`. Nay `generate()` từ chối phát
   ra khối lặp (`_repeating`, chỉ trong `generate`, **không** đụng `learn`/
   `train_pair`).
3. `observe()` gọi `ask(force=True)` nên bỏ qua "bỏ qua" của người dùng, làm phiên
   bị hỏi mãi. Nay tách `user_turn`: idle review không nag, lượt mới được hỏi lại.

Kết quả đo: **8/8 câu trả lời dùng được** (trước đó có câu bị routing/lặp),
**0 vòng lặp**, vision đúng chữ số 1 (confidence 0,647). Nhưng **đúng nội dung
mong đợi chỉ 4/8** — `chào`, `chào bạn`, `cảm ơn`, `bạn tên gì` vẫn dở.

> **Đính chính (xem mục "ĐÍNH CHÍNH" phía trên):** con số 4/8 này đến từ thước đo
> còn lỏng. Với thước đã sửa (đủ từ + không lặp + câu hoàn chỉnh), số đúng là
> **0/8**. Phần "8/8 dùng được" và "0 vòng lặp" vẫn đúng vì chúng đo tiêu chí khác.


**Phát hiện quyết định — vì sao dạy không cải thiện được:** đo trên bản sao,
`train_pair` **hội tụ 12/15** trường hợp (nói đúng câu muốn dạy), nhưng guard
**chặn 15/15**. Nguyên nhân cụ thể: mọi lần dạy đều làm hỏng cùng cặp
`'QA kiểm tra tab: 4 + 4 bằng mấy?' → '8'` (**24/24** lần trong tìm kiếm bước
1–8). Cặp đó **rất vững**, không phải trùng hợp: xác suất token `'8'` = **0,817**,
margin trên token thứ hai **0,771**. Vậy guard đang bảo vệ đúng một kỹ năng thật,
và chính vì thế nó chặn gần như mọi bài dạy mới. Đây đúng là hạn chế ADR0009 đã
ghi: *"guard có thể chặn cả bài học mới"*.

Thêm một khác biệt đo được: guard chấm trên **clone reset state**, còn sản phẩm
trả lời bằng **state liên tục**. Cùng câu đó, clone cho `'8'` (đúng) nhưng state
sống cho `'8 là scar-x.'` (sai). Hai phép đo khác nhau, chưa từng được đối chiếu.

**Chưa làm và cần duyệt:** không nới guard, không đổi learning rule, không đổi
kiến trúc. Ba hướng còn lại đều vượt quyền tự quyết: (a) đổi định nghĩa guard
sang so trên state sống, (b) tránh mở rộng vocabulary bằng cách giới hạn từ vựng
đáp án, (c) mở protocol representation. Tất cả cần người dùng chốt.

Main ALIF/trọng số giữ nguyên (`updates` không đổi), CPU 2 GB giữ nguyên.

## Audit 39: tín hiệu học cục bộ đúng hướng, nhưng gần như không tổng quát hóa — 2026-11-10

[Protocol39](research/39_LOCAL_CREDIT_ALIGNMENT.md) trả lời đúng câu hỏi audit 38 để mở:
hướng cập nhật cục bộ có thực sự làm giảm loss, hay chỉ là nhiễu có norm lớn? Chỉ đọc
15 checkpoint36, áp **chính** hướng local (trung bình endpoint, Adam/clip của checkpoint)
lên **bản sao dùng một lần**, đo lại NLL teacher-forced thật. 0 neural update, không
Torch, checkpoint nguồn và trọng số chính không đổi; 214 kiểm thử qua.

Kết quả dev (192 case, macro 8 family): **63,54% endpoint đi đúng hướng giảm loss**,
ΔNLL trung bình **−0,1291**; đảo dấu cho **+0,0872**. Chênh so đảo dấu **+0,2164**
CI95% 0,1971..0,2371. Vậy quy tắc cục bộ **không phải nhiễu vô hướng** và **không**
phản tác dụng: `LOCAL_DIRECTION_ALIGNED` đạt, `DIRECTION_CONFLICT_CONFIRMED` không đạt.

**Phát hiện quan trọng về chính phép đo:** Adam chuẩn hóa theo từng phần tử, nên control
khớp norm *trước* Adam bị áp lực **lớn hơn thật 4,91×**, control hoán vị **2,13×**.
Chỉ control **đảo dấu** khớp chính xác độ lớn (0,01498 vs 0,01505). Mọi "gain so random"
đều bị nhiễu bởi độ lớn và **không** được dùng làm bằng chứng. Bản chạy đầu
`20261110T080003Z-...-d8de33` có lỗi đo (norm áp dụng luôn đọc 0,0) đã đánh dấu
`do_not_cite`/`superseded`, giữ lịch sử chứ không xóa.

**Giới hạn quyết định:** held-out (hướng gộp từ TRAIN, áp một lần, chấm dev) ΔNLL chỉ
**−8,43e−05**, CI95% −1,93e−04..1,91e−07. Tức tín hiệu đúng hướng **trên từng bài**,
nhưng tổng quát hóa sang bài khác gần như bằng 0. Đây là chẩn đoán cơ chế, **không**
phải normalchat/vision/retention/P1/P2 mới. Các STOP35–37 và PIVOT_NO_SCALE giữ nguyên.
Không mở learner mới; mọi thay đổi learning rule vẫn cần protocol riêng và phê duyệt riêng.

## Audit 38: bài học còn hẹp, có khoảng cách tiền tố — 2026-11-10

Sau câu hỏi người dùng về khácbiệt SNN/Transformer, [protocol38](research/38_WORD_TRAINING_OBJECTIVE_AUDIT.md)
chỉđọc15checkpoint36: taskprefixstructure/FREEvsGOLDPREFIX/đổicâuhỏi và
potentiallocalcredit, khôngoptimizer/learningrule mới hoặcneuralupdates.
201 kiểm thử đầy đủ và 4 kiểm thử chẩn đoán bổ sung qua. Audit15sources×128
cases/7.680forwardsegments hoàn tất,0neuralupdates/khôngTorch. Corpus8mẫu có
8từđầukhácnhau,27/35targetpositions là các từsau; chưa là ngônngữ tổngquát.
Correctdev cùngvịtrí: GOLDPREFIX77,21% vs neuralprefix47,87%, gap29,34đ
CI26,69..31,90,89/648laterpositions bịdừngsớm/censored. Cờprefixreliance
khôngđạt ởcorrect; khôngkếtluận modelchỉdựatiềntố hoặc tựlọcbỏlatercredit.
FREEexacttoken+EOS32,81% khớp source36strict;35,94% cũ lànormalizedstyle,
không thaymodel/metricgate.18,84s/RSS52,74MB/commit37,49MB. Các STOP35–37,
mainALIF/2GB/P1/P2 giữ nguyên; chưa selective learner hoặc training mới.

## Directvisual37: chưa có representation/groundinghữuích — 2026-11-10

[Protocol37](research/37_DIRECT_VISUAL_BRIDGE.md) giữsame32ALIF/input/cache,
eligibility35 mappedonto actualpooledfeature, khôngTransformer/mainchange.
P0`20261110T064337Z-direct-visual-preflight-v1-927d70`:21same32calibration/
forward/headparities/16TRAINresponses mỗi nhánh khớpbytes, pooledsensitivity64
finite-diffmax1,78e−7, midimage/endword/Adam/RNGexact. Forecast1327,90s/21trials,
array485832bytes/model (cảpendinginput/cachetraces), peakcommit987,24MB.
5directvisualtests và2archive/metrics tests qua; fullsuite197passed/48,00s với
BLAS/OMP/MKL1thread trướcformal, guard2GB khôngđổi.21trial đãcompleted/coldexact:
correct25,10%dev/51,53%train, frozen15,05%, shuffled21,32%, BPTT26,30%.
Correct−frozen10,05đ CI7,81..13,32 nhưngcorrect−shuffledCI−6,16..11,00 và
resetheadprobegain−9,26đ CI−11,46..−7,29. Swappedimagedrop2,18đ CI−4,69..11,89;
shape1,74%, chỉfrozengainqua. STOP_DIRECT_VISUAL_BRIDGE_V1, khôngnormalvision/
P1/P2/deploy/scale hoặcretuning.330,14s, treeRSS643,47MB/commit1042,43MB.
UIactual48neurons/RGB/wordoutputs vàcompletedfilearchive864cases/96ảnh;
mainALIF/privateweights khôngđổi, Qwenactualtext36 giữriêngkhônggiảvision.
3freshseed37313/37321/37339 ×7conditions×1536experiences đãkhóa trướcscore.
ReusedactualAGY32questions/owned192PNG, khôngnewteacher/Qwenvision hoặcfakeoutputs.

## Wordbridge36: hidden cóđónggóp nhưng tổngquáthóa chưađạt — 2026-11-10

[Protocol36](research/36_DIRECT_WORD_BRIDGE.md) là pilotword đãnêu35 sauP0,
khôngconfirmation35/P1waiver/scale. AGYCLI đãsinh128prompt thật/62,66s; review
toànbộ tìm1mixedlanguage, giữgốc/cáchly và1AGYrepair đãđọc.36A sửa ownedtarget
apostrophe trướcneuralcode/measurements, giữtokenizer.36B sửa languageinvalid
trướctraining; cảhaiamendments giữhistory/criteria. Dataset64train64dev,3prompt
trùnghistorical,6caveats, không gọi freshfinaltest hoặcteachertruth.
Wordprototype khớpG1legacycontext/3tickinputs,4tests codec/rollback/control qua.
Preflightseed36303/64TRAINresponses:zero=frozen/graphparity/continuationexact,
forecast472s/15trials, peakcommit983,07MB. Formal15trials completed:
correctdev35,94%/train99,48%, frozen16,67%dev, shuffled17,19%, BPTT36,46%dev.
Correct−frozen+19,27đ CI18,75..20,31; resetheadprobegain14,06đ CI0..25 nên
chưagate, macro/familyaccuracy cũngfail. STOP_DIRECT_WORD_BRIDGE_V1, không
tuning/scale/deploy/P1waiver.15coldprocessresumesexact,68,28s/RSS627,22MB/
commit1024,74MB.190tests qua1BLAS/OMP/MKLthread; default2fullruns1legacyguard
HTTP507 retained, khôngnớiguard. Qwen64samepublicprompt đãđo:64/64,14lengthcut,
320,99s, RSS3,363GB/commit4,047GBreferencebudgetriêng, ownedserver đãtắt.
Rawoutputkhôngthànhnhãn/gradeexactstyle; GUI tựhiện64pair/archive và bảng5methods,
có cảcâu sai. Mainprivateweights/core giữnguyên, chưa generalchat/vision/lifelong.

## Nhánh30 v1 đã thử: XOR có tín hiệu, chưa qua P1 — 2026-11-10

[Protocol35](research/35_DIRECT_READOUT_ELIGIBILITY.md) khóa equation own-neuron
forwardeligibility × one-stepreadouterror, chưadeployHermes hoặc scale.
6targetedtests qua:64synapsefinite-difference anchoredsurrogate, forwardbytes,
partialstate/optimizer/RNG, zero=frozen, locality/clipcutoff và transactionrollback.
P0pass maxfinite-differror2,59e−8. Screen45trials/3freshseed đã hoàn tất:
order98,44%direct/+2,34đprobe; XOR91,80%/+10,94đ; binding63,41%/−1,95đ.
Chỉ1/3qua screen, STOP_DIRECT_READOUT_ELIGIBILITY_V1, không8seedconfirmation.
XORcorrect−frozenaccuracy+10,55đ (CI3,52..18,75), đúngtínhiệu tốt hơn shuffled;
3seed chưa đủP1, khôngnormalchat/vision/retention. BPTTbinding63,54% cũngfail.
184tests qua. LỗiWinreadheartbeat đãsửa bằngboundedretry/filelocktestthật;
38completedmodels/coldproof giữbyte,7thiếutrain, partial832+countkhôngbiếtđủ.
Round624,82s tính cảrepair; recoveryRSS621,75MB/commit1055,83MB, originaltreepeak
chưa đủsốđo. MainALIF/241363updatesgiữ nguyên; master1.25researchngoạilệđãduyệt.

## Audit đường thông tin34 đã hoàn tất — 2026-11-10

[Protocol34](research/34_VISUAL_INFORMATION_PATH.md) chỉđọc6checkpoints32/33,
đo actualinput/state/cache/BOS/readout và fixedprobe ngoàicore. Đúng1536train
exposures/seed, không thêm nhãn/coreupdate, Torch hoặc mainchange.3tests
diagnostic đãqua; fullsuite176passed/64,97s trướcaudit. Run
`20261110T000339Z-visual-information-audit-v1-882afb`:10368BOSobservations,
373248ticks, neuralupdates0. Rawsemanticprobe50,22%dev, headfeatureprobe8,30%
frozen/11,60%control; controlgainCI95%−4,19..8,14đ, chưa diagnosticgate.
Firstwordprobe kémactualhead16–19đ; khôngBOSreadoutgap. Các fullpathstages
khôngcontradictoryexactcollisions, clip0,157%/1,079%, khôngnewclipcollapse.
Flagcomponentcollapse chỉquestion-only nhánhthiếuRGB, khôngdiscoverycoreloss.
Không normalization/stateview/decoder/encodergrid mới. PeakRSS133,39MB/
commit150,06MB,47,60s; main/proposal30 giữnguyên. Representationgeneralization
chưađạt; learner mới cần chốtallowedsignal vàprotocol, không tự chọnprobe làmAI.

## Đối chứng visuallearnability33 đã kết thúc, chưa đạt — 2026-11-10

[Protocol/báo cáo33](research/33_VISUAL_WORD_LEARNABILITY_CONTROL.md) giữ48ALIF,
input/head/data/seeds32; hiddenBPTT chỉđốichứng côlập, khôngmain/proposal30.
Preflight`20261109T233806Z-visual-word-control-preflight-v1-805213`:3×64train
responses khớpfullstatebytes/sourcecalibration; graphstateerror3,82e−6,
featureerror5,96e−8. Timing64responses0,85s, forecast317s/round; peakRSS565MB/
commit986MB, cap2GB. Codec/optimizer/nextupdate/generation khớp. Chưa formal
qualityscore trongpreflight. Sau đó173tests qua trướcformaltraining.

Formalrun`20261109T234349Z-visual-word-learnability-control-v1-e51fe9`:3paired
seeds/1536experiences, meanmacrotrain48,43%/dev25,67%, baselinefrozen18,88%dev.
Gain+6,79đ (CI95%0,30..10,76), novel27,44%, shape5,21%; mọiqualitygate fail,
CONTROL_LEARNABILITY_NOT_DEMONSTRATED_IN_BUDGET. Khôngfittraingate65%.
3coldresumesexact;97,82s, treeRSS639,85MB/commit1025,50MB, cap2GB/process.
Hiddencontrol đổi thật nhưng khônglà main/locallearning. Stopcontrol, không
tune/input/epoch/scale hoặc importHermes. Main241363updates/sourceweights
giữnguyên. Localhost tựhiện864dev/96ảnh và actuallearning cónhãnCONTROL.
Tiếp theo ưu tiên readonlyinformation/rootcauseaudit theo master, chưa learner mới.

## JointRGB/question→neuralwords v1 chưa đạt — 2026-11-10

[Báo cáo32](research/32_JOINT_VISUAL_WORD_READOUT.md):9cấu hình/3freshseeds,
hiddenfrozen/localwordreadout,96train/96devscenes và864QA/split,720noveldevcases.
Fullmacrodev18,88% so noimage17,97%/noquestion21,50%; gainvsnoimageCI95%
−4,69..5,99đ, novelmacro16,94%. Causalimageswapdrop3,26đ, chưa20đgate. STOPv1,
không normalize/tuneLR/cachemixture/epochs/scale/mainimport saufailure.
Đã nốijointinputs và sinh từthật trongprototype, chưa learnedvisualgrounding;
khônghiddenlearning/normalvision/retentionclaim.9coldworkingstateresumesexact.

LỗiNaNemptydiagnosticgroup vàwrapperarg đãgiữlog/sửa reportingonly;2completed
models được táichấm khônghọc lại,7cấuhìnhthiếu trainmới. Mainprivateweights/
architecture/proposal30 chưađổi.170testsqua. RecoveryRSS197,39MB/commit150,34MB;
originalfailedattempt peakmemory khôngđođủ nên khôngkhẳngđịnhfullprogrampeak.
Đềxuất30 vềnguồntínhiệu vẫn chờquyếtđịnh rõ của người dùng, khôngsuyraapproval.

## Visionquestionpath đã kiểm thật — 2026-11-10

[Báo cáo31](research/31_VISION_QUESTION_PATH.md):AntigravityCLI sinh32câu hỏi,
đãđọcreview (3wordingcaveats).240actualGeneration.vision calls trênpublicclone,
12ownedrenderedscenes; cùngảnh/state/48neurons thì đổiquestion vẫn cùnginput/
probabilities, contextchannels luôn0. Câu hỏi bịbỏ trước neuralcore; hiện chỉ
10digitoutputs, chưa jointimagequestion→language. Colorloss6cặp là replication
của13, không discovery hoặc visionqualitygain. RGBprototype giữ khácinput,
chưa learnedgrounding. Không training/mainweightschange hoặc proposal30approval.
PeakRSS59,06MB/commit42,59MB,6,94s. Qwen4Btext-only không là visionreference;
không tạoảnh-as-text hoặc tải/bậtVL/9B. Trace cóảnhRGB/8×8gray/rawneuralJSON.
165testsqua, source/mathematics/protocol/240outputs/12PNGhashes đãkiểm;
UI đãquan sáttrực tiếp. Mainprivateweights/topology/vocab/encoding giữ nguyên,
counter241363, RAMlive≈65,0MB tại mẫu đo; mộtmodelID, khôngllama-server kèm.

## Observedassociation v1 chưa đạt — 2026-11-10

[Báo cáo29](research/29_OBSERVED_RESPONSE_ASSOCIATION.md):12trials/3freshseeds,
labelconditionedHebb chỉ contextinputsynapses, khôngbackward. Paired/frozen đều
2,15%dev; pairedprobe29,03% so frozen31,18%, gainCI95%−6,45..3,23đ. Thay đổi
7168transactions/model không tạoquality/representationbenefit. STOPv1; không
autoLR/encoder/epoch/normalization hoặcround2 tune saufailure, khôngmainimport.
Baselinecũngyếu, không suy ra mọiHebbianSNN bấtkhảthi.12coldresumesexact,
frozen=erase-trace/tick/sampler khớp. PeakRSS155,74MB/commit179,01MB,14,23s.

[Đề xuất30](research/30_PROPOSED_LEARNING_SIGNAL_BOUNDARY.md) đang chờ người dùng
chốt có mở nhánh readout-error→neuron một bước hay giữ ranhgiới tínhiệu cũ.
Chưa code/huấnluyện nhánh đó; main vẫnLIF/ALIF+locallearning, cấmTransformer/
pretrainedproxy/BPTTmain. Các công việc audit/UI/docs không phụ thuộc câu trả
lời vẫn thực hiện; chưa coi quyềnnghiêncứu là quyềnđổi allowederrortransport.

## Audit nhiễu tín hiệu học — 2026-11-10

[Báo cáo28](research/28_CREDIT_SIGNAL_NOISE.md):144conditionalinputs×128pairs,
9immutablecheckpoints/3pairedseeds, không training. Tạilocalcheckpoint, scoreEMA
cosinetái lập≈0,202 so surrogate≈0,999; crossfit/EMAnoiseenergyratio≈0,227,
CI95%0,104..0,382. Baselinecrossfit chỉvariancecontrol, haihalfmeans bịcoupled;
không coi cosinecrossfit là lặp độc lập hoặc kết luận qualitygain. Surrogate
biased, khônggroundtruthgradient. STOPword-scorev1/v2 giữ nguyên, khôngscorev3.
9sourcehash/144weight-learningstate-RNGchecks khớp; source/mathprotocolbấtbiến.
PeakresearchRSS594,03MB/commit994,04MB,159,05s; không mainweights/architecture
change hoặc teacherAPI mới. Representation/credit/retention/normalvision còn mở.
159testsqua gồm nativeparity; bảng9tínhiệu/CI đã kiểm trực tiếp trênlocalhost.
Privateweights/topology/vocabulary/encoding/counter241363 giữ nguyên; appmột
coreALIF, RAMlive khoảng57,8MB tại mẫu đo, khôngllama-server chạy kèm.

**Khóa định hướng theo yêu cầu người dùng:** giữ lõi Spiking Recurrent Neural
Network LIF/ALIF và local synaptic learning; cấm tự chuyển thành Transformer,
pretrainedLLM hoặc proxy. Qwen chỉ là đối chứng độc lập. Bản cập nhật trace
không sửa thuật toán neural hoặc thay trọng số đang phục vụ.

## Trace tự động và cập nhật kết quả — 2026-11-10

[Đối chứng27](research/27_ISOLATED_WORD_LEARNABILITY.md) đã xong6lượt/3pairedseeds:
ALIF48+dense+BPTTcontrol dev44,09% so frozen8,60% và localscore7,53%; train16
91,67% chưa là normalchat. Tín hiệu học có ảnh hưởng rõ trong bài8styles,
nhưng globalcontrol không đáp ứng localmainconstraint và có compute thêm.
Mainword-score vẫnSTOP, tổng thểPIVOT; không importweights/backward/Transformer.
6coldresumesexact,156testsqua; mainprivateweights/topology/vocab/encoding/counter
241363 giữ nguyên. PeakcontrolRSS623,37MB/commit1,015GB. Sự cốWindowsatomicwrite
đã tái hiện và sửa boundedretry; thời gianlỗi/sửa tính trong deadlinegốc.

47Qwenreference mới đã đo sau khiTorchdừng; đủ95prompt reference gồm41bịcắt64tokens,
không dùng làm nhãn. PeakQwen3,236GB báoriêng với4,5GBreferencecap, serverđãtắt.
Mục `#answer-archive` tự hiện đủ95case đãlưu và Qwenkhớpprompt, có nhãncontrol/
archivaltime. Không chạy model để tô dữ liệu hoặc mở lại family đãSTOP.

[Vòng word-score cuối26](research/26_FIXED_INPUT_NEURAL_TRANSFER.md) đã xong:
27trials/3freshseeds, sameH48 và rule. Densefull dev7,53% so densefrozen8,60%,
gain−1,08đ CI95%−3,23..3,23; priorfull9,72% so frozen13,89%. STOP, đóng hướng
word-score application sauv1/v2, không tune/scale/import.27/27coldresumes khớp,
152tests qua, peakresearchRSS163,45MB. Maincore và privateweights giữ nguyên.
Readonlyinputinterventionv1 thấy có sensitivity nhưng chưa devbenefit rõ;
không giải thích mọi thất bại là model bỏ qua câu hỏi. Goalnormalchat/vision
và P2retention vẫn chưa hoàn thành; ưu tiên learnability/creditassignmentcontrol.

Mở [nhật ký công việc](http://127.0.0.1:8765/#agent-work): tự hiện việc Codex
công bố, câu trả lời thật, loss/ΔW/guard và tiến độ. Không chọn model hoặc bấm
chạy. Nhật ký kỹ thuật tách khỏi neural trace. Acceptance thật:148tests qua,
inference21ticks, baGET giữ updatecounter241363,100logloss cũ còn đọc được,
RAM live56,6MB tại mẫu đo. Không huấn luyện mới hoặc thay liveweights.

Các lượt trước đã hoàn tất và được bổ sung vào lịch sử localhost:
[representation/search24](research/24_REPRESENTATION_DECODER_AND_SEARCH.md) và
[word-score/input25](research/25_WORD_SCORE_TRANSFER_PROPOSAL.md). Beamsearch
và chuyển paired-score sang từ chưa đạt; stateview và inputcontrol cũng STOP.
Không gọi cải thiện token teacherforcing hoặc inputclassifier là normalchat,
không mở lại các family đã dừng. Goal nói/nhìn bình thường vẫn chưa hoàn thành.

## Controller stop-loop đã kiểm thật — 2026-11-10

Hai ví dụ chào hỏi từ fixture dự án có 4 cập nhật được giữ và 60 rollback,
nhưng câu trước/sau không cải thiện. Controller đã sửa để dừng sau hai đề xuất
giống hệt về model state, prompt và target cùng bị guard từ chối. Kiểm tra live
đã quan sát 2 rejection rồi dừng; trọng số/counter giữ 241363. Không đổi guard
hoặc quy tắc synapse. Hậu kiểm 48 ảnh chữ số giữ 81,25%, không mất câu đúng cũ
hay hai bài toán đã học trong phép đo hẹp này.

140 kiểm thử qua, gồm native baseline parity. 48 câu công khai mới đã chạy
trên lõi live, ghép đúng prompt với Qwen đã lưu; counter không đổi. RAM live
khoảng 60 MB, Qwen reference đã tắt. Không có candidate curriculum nào được
nhập vào live, không bật optional surface normalization. Normalchat/vision và
retention rộng vẫn chưa đạt.

## Hội thoại thật và curriculum nhỏ — các candidate chưa đạt

[Báo cáo23](research/23_REAL_DIALOGUE_STATE_AND_CURRICULUM.md):Antigravity đã sinh
48diagnostic +160curriculum prompts; rà soát giữ117train/31dev. Qwen4B đã trả
lời48câu mới, serverđãtắt; referenceRAM3,2GB được ghi riêng, không gọi dưới2GB.

Stateaudit cho thấy resetclone không sửa được bảnpersonal; rehearsal phục hồi
câu quen nhưng cách nói mới còn kém.12curatedtrainingclones dùng dataAGY thật,
hiddenfrozen/readoutlocal, lựa chọn dev trước acceptance; tất cả chưađạt để
thay live. Optional terminalnormalizer chưa bật. Source-aware projectteaching
được tách riêng, không giảhumanfeedback hoặc importfailedweights.

Timescaleview60tổ hợp đãSTOP, khôngfreshtraining từscreen. P1/P2gate lịch sử
không đổi. Normalchat/vision, broadskill và lifelongretention vẫn chưa đạt.

## P2 baseline chưa đạt; cần xử lý representation/state trước protection

[Báo cáo21](research/21_P2_RETENTION_AND_CUE_INFORMATION.md):18lượt/3seed,20context
trộn dần, same5760updates. Full+reservoir final34,74%, acquisition57,64%,
forgettingTB29,09đ; mọi condition fail joint gate. Frozen hidden vẫn quên27–32đ,
head overwrite có bằng chứng; offlinejoint/refit cũng dưới65%, representation
hiện chưa đủ. Không mở importance tuning hoặc confirmation từ kết quả này.

Audit immutable checkpoints cho MC64/tick6 probe khoảng67–68%, endpoint chỉ
56–60%; inputproductcontrol100% không là neural skill. Chưa giải noise/state/
association. Maincore/P1historicalgates giữ nguyên, khôngscale/deploy.

## P1 paired-score qua gate hẹp — 2026-11-09

[Báo cáo20](research/20_SCORE_ELIGIBILITY_AND_VARIANCE.md):108development+120
confirmation trials, 8 seed xác nhận mới/task. Selected raw-independent Adam
0,002 dùng local pre/post score × scalar return, không backward. Gate P1 qua
trên order/XOR: gain18,33/20,00 điểm, lowerCI14,65/15,06; direct91,99/74,56%.
Binding direct49,56%, gain−1,29 điểm: thất bại. GO_P2_RESEARCH_ONLY, không scale
hoặc deploy. Retention/quy tắc mới, normalchat/vision và broad skill còn thiếu.

24/24 cold continuations khớp,129 tests qua. Peak nghiên cứuRSS165MB/commit188MB.
Reward đúng hơn shuffled trên2tasks, nhưng chưa chứng minh vượt mọi clipped
baseline. Paired sampling đối ngẫu không được chọn. Các family đãSTOP giữ nguyên.

## P1 audit optimizer/coverage — 2026-11-09

[Báo cáo19](research/19_CONTROL_OPTIMIZER_AND_COVERAGE.md):45isolatedcontrols,
3freshseeds, sameforward/head/data, factorialAdam/SGD×all48/rotating12.
Bindingmean: frozen65,10%; Adam-all82,42%; Adam-rot12:63,28%; SGD-all61,07%;
SGD-rot12:62,76%. Optimizer/coverage tương tác; không quy hết gap cho credit.
Chỉ thăm dò, Holm12contrasts chưađạt; không mở lại familylocal-classifier.

DIAGNOSTIC_ONLY, khôngnewlocalcandidate/deploy/scale. Loss/ΔW/tensorpredictions
tự lênlocalhost. Khoảng302s, peakRSS751MB/commit1,25GB. Liveupdatecounter
giữ241347; maincorekhôngBPTT.122testsqua, gồmnativebaselineparity.

## P1 local classifier — đóng family sau vòng cuối

[Báo cáo18](research/18_LOCAL_CLASSIFIER_FINAL_AND_LEARNABILITY.md):108development
trials,3freshseeds, conditioning/timing có ablation; selected hidden probe gain
order+2,60đ, XOR+3,78đ, binding+0,26đ. Không qua gate5đ trên2tasks, không mở
confirmation hoặc deploy. **NO_GO_LOCAL_CLASSIFIER_FAMILY**, không vòng3.

18isolatedBPTTcontrols cùngforward/head cho binding91,67% so với62,37%readout;
đây là learnability diagnostic, không đưa backward vào lõi chính.9/9readout
control predictions khớp,9/9cold-process continuation khớp. Peak nghiên cứu
RSS750MB/commit1,25GB. Livecore không đổi sang candidate. Normalize có ích
thăm dò choorder/XOR nhưng khôngbinding; timing không cải thiện nhất quán.

Lượt và learning/answer measurements đã tự lênlocalhost khi chạy thật.
Protocol/result lịch sử giữ nguyên. Mục tiêu hội thoại/vision, retention,
Experience-to-Skill vẫn chưa đạt; PIVOT theoP1/P2, khôngscale.

## Trace thụ động, inference và learning thật — 2026-11-09

Theo yêu cầu mới, bảng localhost **tự theo dõi**; đã bỏ chọn nguồn/bấm chạy khỏi
observatory. `#test-answers` tự hiện input thực, generated, target và archived
measured Qwen; `#learning-observer` hiện loss/ΔW/changed weights/guard thật.
Log cũ thiếu metric ghi thiếu. Không dùng mock model/teacher proxy để điền bảng.

Đã chạy [24 inference + một lời dạy số học](../artifacts/live-observatory-latest.json)
trên chính phiên đang dùng.17 accepted micro-updates,34 token steps,
loss7,155→1,144; last training attempt sinh15 cho7+8.0 rejection mới; không
giả loss cho93 rejection lịch sử. Peak RSS quan sát khoảng70,5MB trong lượt.
Targets development không được dùng làm nhãn; Qwen full-prefix và serving
latest-user-only được ghi rõ là khác điều kiện. Chưa language/retention success.

119 tests qua, gồm native baseline parity; UI đã thấy kết quả thật tự hiện
không bấm chạy. [Hướng dẫn](OBSERVING.md), [ADR0012](decisions/0012-passive-real-work-trace.md).
Sau đó idle review tự thêm2 accepted updates nữa (không bấm dạy lại), được
trace ghi loss0,9957 và0,7706. Đây là số đo các giao dịch tiếp theo, không sửa
kết quả lượt17updates ban đầu. Restart vẫn đọc lại25answers/19lossrecords.

## Quan sát neuron và công việc nghiên cứu — 2026-11-09

Website có **Neuron theo task / Đang thử nghiệm / Lõi đang trò chuyện** (nay
bổ sung hai mục trace thụ động ở trên).
Đã quan sát UI và chạy 5 task trên core live: 64 → 64 → 96 → 128 → 64 neuron;
128 đã cấp được giữ lại, nền có thể dùng 48. Có checkpoint trước lượt và nút
dừng; không thêm lesson hoặc câu hỏi giả. Bản sao public đã đo dãy
48 → 64 → 96 → 128 → 48. Đây là chọn theo tải, chưa learned difficulty hay ACT.

[Hướng dẫn](OBSERVING.md) giải thích trace, RAM, job state và lịch sử qua restart.
Lượt nghiên cứu mới tự đăng ký; trạng thái chạy xác minh PID/create-time, không
tự restart. Trace thí nghiệm tách khỏi model live, giữ cả kết luận thất bại.

[Local classifier adaptation](research/17_LOCAL_CLASSIFIER_ADAPTATION.md) đã chạy
54 development trials và hiện tiến độ/mẫu neuron trên trang. Chưa qua gate
representation; không mở confirmation/deploy hoặc nâng quality claim.

## Real RGB grounding diagnostic — 2026-11-09

[Báo cáo16](research/16_REAL_RGB_GROUNDING.md):640train/160heldoutRGBobjectảnh,
3freshseed×2head, same48ALIFhiddenfrozen. Raw18,96%/normalized20,42%accuracy,
blank10%, crossclassmismatchđãaudit10–10,83%. Chưađạtgate30%, khôngnormalVQA
haylearnedcaption. Originalroll1mismatchgiữ150/160sameclass; đượcauditadditive,
khôngretrain/thaygate. Datasetpinnedhash, nooriginaltest/privateinputs/redistribution.
Livevisionvẫndigits, khôngthayweights. ByteworkingfamilyđãSTOPsau2vòng.

## Byte codec và working-state — 2026-11-09

[Báo cáo15](research/15_BYTE_CODEC_AND_WORKING_MEMORY.md): UTF8codec fixed bỏOOV,
nhưng12localtrials vẫn0devutteranceexact; syntaxvalidity dohostmask, khôngIQ.
Isolated6Adam/BPTTcontrols cũng0exact, trainbyteaccuracychỉ26–28%trongbudget.
Không đủ quy mọi lỗi cho localrule hoặc nói SNNbất khả thi.

Audit6model/32freshpairs thấy inputdistinctioncollapse ở127forcedbytes; thử
heldactivation192byte trên9trials không tạo taskgain,0devexact. **STOPbytefamily
sau2vòng**, coldresumequa nhưngqualityfail. Không đổi modelHermes hoặcweights.
Visionảnhthật đang có protocol nhỏ riêng, không gọi RGBfeature là đãhiểuảnh.

## Event dialogue + Qwen đối chứng — 2026-11-09

[Báo cáo14](research/14_EVENT_DIALOGUE_AND_QWEN_REFERENCE.md): đã thật sự train
96Antigravity responses trên3freshseeds×4conditions, chronological input→một
ALIF48→word decoder. Dev exact0/24mọi condition; outputvẫn lặp/vô nghĩa, dừng
theo protocol, không deploy.30dev words không biểu diễn được bằng word dictionary
train-only. Không lấy classification95% để gọi normalchat. Cold resume12checkpoints
khớp; là bằng chứng persistence, không quality.

Qwen4B đã trả lời cùng24dialogueprefix, có câu có nghĩa hơn nhưng11bịcắt và có
lỗi tự nêu ngày/khả năng tra cứu. Server Qwen đã tắt; không proxy vàoSCAR-X.
Corpus curatedđã dùng **trong experiment**, chưa vào phiên cá nhân. Mục tiêu
chat/visionbình thường còn thiếu representation/codec/retention; goal chưa xong.

## P1/P2 input và learning audit — 2026-11-09

[Báo cáo13](research/13_INPUT_AND_LOCAL_LEARNING_AUDIT.md): encoder G1 mất thứ tự
ở128/128cặp last-write, grayscale mất màu. Prototype ordered/RGB events cùng
channel budget sửa các counterexample; v2không trùng256cặp mới, chưa serving.

Hai local-target rounds108development trials không qua representation gate:
**NO-GO nhóm ứng viên**, không mở confirmation/scale. Output-local normalization
được xác nhận8fresh seed trên order/XOR (95,53%/86,99%), hidden frozen. Tuy
nhiên20context retention fail; normalization làm forgetting tăng35,14pp. Đây
là readout acquisition gain, chưa hidden learning hoặc solved lifelong.

Antigravity40dialogue mới đã qua repair/editorialreview:194targettokens,
train20vi/12en,dev4vi/4en. Rawfalse/actionclaim labels giữ và không train. Không
tự nạp corpus hoặc candidate weights vào phiên người dùng; model live giữ nguyên.

## Tab Hỏi để học và ba chế độ — 2026-11-09

Đã cài disk plugin Hermes qua cửa desktop/backend sẵn có, route
`/scarx-learning`, nav **Hỏi để học**. Native GUI đã quan sát hai chiều: gửi
QA4+4 → nhận câu hỏi → trả lời8 → thẻ chuyển Đã trả lời, có acknowledgement và
nhãn kiểm bằng phép tính. Hai câu hỏi không liên quan còn giữ. History512tin,
reply theo source IDs, clientID dedup, feedback theo verifier/guard hiện có.
Main chat mới chỉ routing notice, không blocking inline clarify cho mode này.

98 SCAR-X tests (gồm4native parity),2 UI behavior tests và real Hermes plugin
registration/auth → SCAR-X HTTP acceptance qua. Screenshot QA giữ riêng trong
`local/hermes-learning-tab-qa.jpg`, không đóng gói private transcript.
[ADR0010](decisions/0010-learning-tab-and-cognitive-modes.md).

Người dùng đã làm rõ3chế độ cùng core: Fast Processing / Deep Thinking /
Autonomous Idle Learning. [Tài liệu12](research/12_ADAPTIVE_COMPUTE_AND_IDLE_COGNITION.md)
đối chiếu nguồn và nối vào P0–P5. ACT/halting và tự khám phá có verifier vẫn
Proposed, không được nói như đã chạy. Mục tiêu chat/vision bình thường chưa đạt;
giữ PIVOT, không mở rộng data/model để che acquisition fail.

## Tăng tải, retention và Qwen4B — 2026-11-09

[Báo cáo11](research/11_KNOWLEDGE_LOAD_RETENTION.md): đã thử ladder128bài tối đa,
5condition×3orderseed từ1public initialization. Các nhánh học dừng ở16bài theo
stop rule: paraphrase0%, chưa có useful acquisition. Full/readout/replay quên
oldskill trung bình25pp; guarded8,33pp sau audit, vẫn vượt3pp/5pp. Replay giữ
vài đáp án nguyên văn, chưa đạt kỹ năng. Cold restart khớp không chứng minh nhớ
ngày/tháng. Negative-answer decoder limitation được lưu ở run abort riêng.

Qwen3-4B-Instruct-2507 Q4_K_M đã chạy thật:160/160bài số học, đã tắt server.
SCAR-X yếu hơn rất rõ trong phép đối chiếu hẹp; không chứng nhận4B-equivalence.
QwenRSS peak2,924GB, SCAR-X diagnostic102,7MB; budgets không tương đương.
Không thay liveweights hoặc đổi cấu trúc. Người dùng đã làm rõ fast/deep/idle là ba chế độ cùng core;
master/code hiện vẫn một pool hidden, không phải ba model riêng.
Không dùng kết quả flatpool để kết luận cho kiến trúc phân tầng khác.

## Học xuyên hội thoại và dữ liệu Antigravity — 2026-11-09

Một learner giữ trạng thái xuyên lượt, nhận lời sửa/đáp án, cập nhật cục bộ
và idle review có guard. Lưu controller/lesson/questions/RNG cùng neural core;
rollback giữ object core tại chỗ. Không tự dùng output làm nhãn đúng. Guard hiện
chỉ bảo vệ bộ probe phát triển nhỏ, chưa chứng nhận retention.

[Kiểm tra mới](../runs/20261109T064249Z-continuous-dialogue-acceptance-ce1b2c/report.md):
9 lesson,15 microstep chấp nhận/116 từ chối, exact resume và neural/RNG continuation
khớp. Câu `1+1` vẫn sinh “Bạn có..” sau review. **Acquisition chưa đạt**; cập nhật
trọng số và giữ trạng thái không đồng nghĩa đã học kỹ năng. Không thay release
weights bằng candidate thí nghiệm. Những số đo cũ bên dưới giữ như lịch sử.

Antigravity CLI đã tạo240 ví dụ; validator giữ217 (178train/39development), trong
đó53 nhãn arithmetic kiểm độc lập.23 ví dụ cách ly. Bổ sung40 hội thoại/120response
và16 visual QA +2caption từ2 ảnh công khai. Nhãn style/ảnh vẫn là candidate;
train/dev hội thoại bị confound ngôn ngữ nên chưa dùng kết luận tổng quát hóa.
Không tự nạp corpus vào phiên cá nhân. [Báo cáo và giới hạn](research/10_DIALOGUE_DATA_AND_ONLINE_LEARNING.md).

Hermes có câu hỏi controller riêng và terminal event request-correlated;
không chờ LLM tool continuation.6 Python integration tests và5 UI tests qua;
GUI đã quan sát: trả lời6 cho câu QA3+3, thẻ settle; gửi tiếp hi nhận Hello.,
không còn thẻ bị khóa hoặc spinner. [Bằng chứng](../artifacts/hermes-online-resolved.jpg).
Footer là lượng nạp thực + neuron,
không là thước đo intelligence. General chat/normal photo QA vẫn chưa đạt.

## Capacity linh động và tải từng lượt — 2026-11-09

Theo phê duyệt mới,48không là trần sản phẩm. Prototype mộtcore48–128 có actual
selected-row/readout-column compute, lazy dormant decay, growthsnapshot và
topology/activecount/RNG/clock/ID persistence. Budget dựa inputvolume/modality
hoặc override; chưa có detector độ khó ngữ nghĩa. Newreadout0cần học; tăng neuron
không là tăng năng lực đã chứng minh. Foreground không tắt các cột readout đã
nối, chưa có causalimportance/pruning. [ADR0008](decisions/0008-adaptive-neurons-and-request-load.md).

Hermes lấy requestload theoID: inputchars/UTF8bytes/images, vectorchannels/bytes,
neuron đang tính/đã cấp/trần. Thanh chính hiển thị **Nạp X ký tự · N neuron**.
Memorycapacity và retainedskills vẫn chưa đo, không dùng load làm intelligence.

Đã mở explicitlesson với từ đầu ra mới≤256token và migrationđúng. Engineering
clone học được `1+1→2` sau32lesson, có exactresume; đây là trainingpoint, không
general toán và không âm thầm thêm vào trọng số phiên cá nhân.

[Vòng cải thiện3seed/9condition-runs](../runs/20261109T015554Z-generation1-improvement-fdd0ba/report.md)
đã kết thúc: lexicalencoderfull so originalfull **−3,125điểm** textconfirmation;
full so lexicalreadout **+6,25điểm**, nhưng accuracy tuyệt đối thấp. Chưa có
representation/qualitygain hữu ích; không phát hành candidate kém hơn, giữ
trainedweights G1cũ. Dataset mới32promptkhác18promptcũ, cùng8families, không
phải bộ đo generalchat hoặc lifelong. Giữ PIVOT và forgettinggate lịch sử.

Cần data mới cócurriculum/verifier; đã chuẩn bị
[brief cho Antigravity](../data/contracts/ANTIGRAVITY_DATA_BRIEF.md); việc tạo/kiểm tra
dataset đã hoàn thành theo cập nhật trên, chưa nạp corpus chưa xác minh vào phiên cá nhân. Có thêmdata chưa đủ
để cam kết sửa localcredit/representation.

## Hồ sơ triển khai trước prototypecapacity

## Hermes có trạng thái — 2026-11-09

Sửa Hermes theo yêu cầu người dùng; không sửa neural core/weights/topology G1.
Mode `stateful_chat` named-provider giữ buffer4096, bỏ floor64K dành cho LLM
agent, không tools/prompt injection/history replay/fallback. Lịch sử UI vẫn
append-only, backend giữ neural state riêng. Real AIAgent + HTTP + SessionDB
đã kiểm ở [nonstream](../artifacts/hermes-stateful-check.json) và
[stream](../artifacts/hermes-stateful-stream-check.json). Đã khởi động lại Hermes
và thử hai lượt trong cùng phiên desktop: không còn lỗi64K, lịch sử giữ nguyên,
`hi → Hello.`. Lượt `chào bạn` vẫn sinh câu sai/lặp; không nâng quality claim.
Footer đổi thành **Backend state**, popover phân biệt input2048ký tự và memory
capacity chưa đo.48 Python tests có liên quan và10 UI tests qua, không phải toàn
bộ suite Hermes. [Bằng chứng GUI](../artifacts/hermes-stateful-ui.jpg) và
[hướng dẫn](../integrations/hermes/README.md).

[Memory/learning measures](research/09_MEMORY_LEARNING_MEASURES.md) thay mục
tiêu độ dài context bằng đường cong retention, working-memory horizon,
acquisition,transfer và skill capacity có điều kiện. Đây là thiết kế phép đo,
chưa chạy benchmark mới, chưa có giá trị capacity/horizon được xác nhận.

## Hiện hành — Thế hệ đầu, 2026-11-09

Một core 48 ALIF chung cho text/ảnh/nền, một checkpoint và một model Hermes
`scarx-generation1`. Điểm vào `run_app.py` dùng `generation_server.py` và
`Generation/UnifiedCore`; các mục V0.1/V0.2 bên dưới là hồ sơ lịch sử.

Đã có neural idle loop, foreground priority, topology/spike/voltage/probe trace,
pause/speed, explicit local teaching và full-session restore/RNG/controller.
Không reset neuron trong runtime, không tự học weights lúc rảnh hoặc grow.
Trace không giả độc thoại nội tâm. Cả48neuron vẫn được tính mỗi tick.

Huấn luyện chung hai seed: greeting50%/50% (fail80% target), text33,33%/22,22%,
digits88,33%/89,44% diagnostic. Seed17 chọn bằng development. Các câu `hi`,
`chào`, `chào bạn` là training points trả lời đúng; không là bằng chứng general
chat. [Run](../runs/20261109T000625Z-generation1-joint-bd6c92/report.md) và
[audit protocol deviation](../runs/20261109T000625Z-generation1-joint-bd6c92/PROTOCOL_DEVIATION.md).

72 tests đã qua, gồm native parity và 13 test G1. Formal Phase2.2, useful hidden
representation, retention gate, general vision/actions và useful consolidation
vẫn chưa đạt. Nghiên cứu PIVOT_NO_SCALE; xem [master1.2](RESEARCH_PLAN.md).

[Engineering acceptance](../runs/20261109T005218Z-generation1-engineering-9111b3/report.md)
đo khoảng57MB, idle ticks tăng trong khi weight fingerprint bất biến, shared
image/text và exact continuation. Decoder dừng cuối câu thử trên development
giảm37,5%→25%, nên không bật. Gói `SCAR-X-Generation-1.zip` có kiểm manifest và
khởi động từ thư mục giải nén mới; không chứa localdata/Hermesconfig/secrets.

Hermes provider config và native model-picker API chỉ có `scarx-generation1`;
model overrides tắt nhãn reasoning/tool chưa có. Existing chat có thể cần phiên
mới hoặc `/model custom:scarx-local:scarx-generation1`. Qwen9B autostart vẫn tắt.
Không có plugin `ponytail` khả dụng trong phiên cập nhật; không giả lập công cụ.

## Hồ sơ các sản phẩm đã ngừng phục vụ

## Huấn luyện và V0.2 LAB — 2026-11-09

- PilotV2 đã kết thúc, không còn process training: languageexact3/16=18,75% (fail), digitclassification321/360=89,17% (diagnostic pass, không phải screenshotunderstanding).
- Decoderbeamfixed4 không cải thiện đủ; không tiếp tục tune trên test hoặc thay threshold để gọi languagepass.
- Tạo modelLAB opt-in riêng trong app/Hermes: `scarx-language-lab-v0.2` sinh text từ trainedwordhead, `scarx-digits-lab-v0.2` phân loại ảnh chữ số. V0.1không bị thay.
- Một lesson trực tiếp cho ví dụ `chào bạn` đã loại lặp trên trainingprompt bằng localupdates; metadata vẫn ghi languagequalitygatefalse. Không gọi đây là generalization.
- LAB có explicitteaching, vocabulary expansion≤256token/48hidden, SQLitecheckpoints và exactrestore/restart. Chưa có generalchat, ảnhgiao diện, OCR, agentactions hoặc learnedreasoning.
- [Pilot V2](../runs/20261108T190315Z-language-vision-pilot-v2-6a889e/report.md) và [release descriptor lịch sử](../artifacts/pilot-release.json) giữ checkpoint/giới hạn. Hướng dẫn hiện hành ở [README](../README.md); llama.cpp autostart vẫn tắt để tránh Qwen9B chạy kèm.

## Product-first — MVP V0.1 chạy được

Người dùng đã thay phê duyệt docs-only bằng scope triển khai MVP. Chạy `python run_app.py` và mở `http://127.0.0.1:8765`; [README](../README.md) là hướng dẫn dùng/cài/teach/restore duy nhất.

Đã có: UI localhost, PlasticCore48ALIF/864synapse thật, dạy nhãn theo verifier A/B, kiểm before/after và A→B, spikes/weightdrift, SQLitememory/history, atomicfullsessioncheckpoint/restore/restart và resource display. Windows processcommit cap2GB áp khi Job attach được. Không cloud/LLMpretrained, không đổi rule/topology hoặc tăng scale.

[Acceptance demo](../runs/20261108T175211Z-mvp-v01-acceptance-3d887d/report.md): A50%→99,2% sau hai đợt; sau hai đợtB, A89,8%/B74,2%; chưa có mẫu exact trong train cho tập chấm. Restart/restore exact được kiểm. Đây là product regression sau build iteration, không phải researchgeneralization claim. Scope còn yếu:2tác vụ có cấu trúc, chưa natural-language, chưa solvedforgetting, chưa customverifier/importUI hoặc consolidationnền.

Xem [ADR0005](decisions/0005-product-first-mvp.md). Các mục nghiên cứu bên dưới giữ kết quả và gate; **không còn là điều kiện chặn việc dùng/phát triển MVP**.

**Hermes backend:** `/v1/models` và `/v1/chat/completions` nonstream/SSE cho structuredJSON đã có, provider `scarx-local` được thêm vào configHermes và giữ nguyên Qwen9B đang chọn. Không MCP, không GGUF. Automaticcompute chọn4/16passes, chưa là learnedreasoning; vision/actions chưa được hỗ trợ. PackageZIP có source và checkpointdemo; xem README mục8 và capabilities.json.

Cập nhật **2026-11-09**. Định hướng: **Lifelong Adaptive Intelligence**. Mục tiêu 1M context và độ dài token đã được loại bỏ.

**Kế hoạch chính duy nhất:** [RESEARCH_PLAN.md](RESEARCH_PLAN.md). SCAR-X giữ mục tiêu neural architecture, không chuyển thành memory/RAG cho LLM. Bản1.0 chỉ cập nhật docs; ADR0005 nay phê duyệt MVP/code/UI. Protocol và kết quả nghiên cứu cũ vẫn giữ nguyên.

**Quyết định sau Phase 2.1: PIVOT_NO_SCALE.** Giữ mạng nhỏ 48 neuron. Chưa có bằng chứng đủ cho lợi thế representation và continual learning của các quy tắc đã thử.

## Kết quả hiện tại — Phase 2.1

- [Báo cáo kết quả và câu trả lời nghiên cứu](research/08_PHASE21_RESULTS.md)
- [Kế hoạch đăng ký trước](../experiments/phase21-v1.md) và [bổ sung stream không task boundary](../experiments/phase21-lifelong-amendment.md)
- [Phản biện khả năng học dài hạn](research/07_LIFELONG_LIMITS.md)
- [Audit hiệu lực phép đo](decisions/0003-phase21-evaluation-audit.md)
- [Danh mục run](../runs/INDEX.md), [kết quả máy đọc](../artifacts/phase21-summary.json)

Đã audit 15 trial/checkpoint Phase 2 và chạy 303 trial mới: eligibility score, STDP, metaplasticity, importance decay, replay/consolidation factorial, readout-only, SI tại readout, modulation nhiễu và BPTT control riêng. Có 3 seed phát triển, 8 seed xác nhận và 8 seed mới cho strict binding audit; không tính các condition ghép cặp như seed độc lập.

Stream có 1.280 trải nghiệm, không task ID/boundary cho learner, 20 cửa sổ đánh giá và 20% nhãn nhiễu. Fast weights, stable anchor và consolidation định kỳ đã triển khai. Neural/RNG/importance/replay continuation được kiểm tra.

Kết quả: hidden chưa đạt gate representation; metaplasticity giảm forgetting nhưng accuracy gần 50%; không condition nào đạt đồng thời mean forgetting ≤3 điểm phần trăm, worst ≤5 điểm và chất lượng tối thiểu. RAM quan sát trong trần 2 GB, chưa phải OS hard cap.

Hướng còn hợp lý: importance gắn với mức hữu ích, ổn định readout, credit assignment và bài representation hợp lệ. Cần protocol/seed mới trước vòng tiếp theo. Không tăng tham số, huấn luyện ngôn ngữ hoặc tối ưu C++/iGPU trong vòng này.

## Roadmap nghiên cứu Phase 2.2 — không chặn MVP

1. **P0 kỹ thuật:** compatibility `plasticity-v1`, whole-state checkpoint/rollback cho learner/replay/RNG, evidence verification và budget RAM toàn phiên. Save/load riêng đã có; transaction cho lõi mới chưa hoàn chỉnh.
2. **P1/P2 ưu tiên khoa học:** representation/credit assignment và readout stability/usefulness-based importance. Không dùng replay tối ưu để che các vấn đề gốc.
3. **P3:** Experience-to-Skill với tình huống giữ kín, reset/restart và retrieval-off; semantic/procedural memory vẫn là mục tiêu chưa chứng minh.
4. **P4:** offline consolidation thủ công có matched compute, verifier/guard và rollback; chưa mở automatic idle scheduler.
5. **P5:** độ bền20→50→100giai đoạn có gate, giữ capacity và plasticity reserve.

Chi tiết giả thuyết, code, dữ liệu, baseline, metrics, pass/fail, rủi ro và phụ thuộc ở [roadmap master](RESEARCH_PLAN.md#7-experiment-roadmap). Tất cả stage trên cần duyệt triển khai/protocol trước. Ngưỡng mean forgetting≤3 điểm phần trăm và worst≤5 trên từng seed vẫn giữ; kết quả cũ không chấm lại.

Chưa có Git trong project; [master mục8.3](RESEARCH_PLAN.md#83-quản-lý-phiên-bản-và-dữ-liệu) chỉ đề xuất phương án version control local, chưa init/commit/push hoặc đổi cấu trúc. [ADR0004](decisions/0004-master-plan-and-phase22-docs.md) ghi phạm vi quyết định mới.

## Hồ sơ Phase 2 lịch sử

- Cấu trúc src theo domain, project map, ADR, protocol và run index.
- ALIF recurrent, block-local reward-modulated STDP, readout local, stable weights.
- Checkpoint có checksum, atomic save, exact resume; guard và rollback.
- SQLite exact episodes, FTS5, provenance; byte generation plumbing gắn nhãn chưa huấn luyện.
- CLI init/generate/observe/search/learn/rollback/benchmark.
- C++ CPU SIMD kernel, build bằng MSVC, kiểm tra parity.
- 15 trial local/analog qua3 seed,20 task/chặng; run xác nhận cùng seed.
- 3 trial surrogate BPTT control riêng; runtime/memory/retrieval smoke.
- Sáu báo cáo nghiên cứu trong research/.

## Kết quả Phase 2 cần nhớ

Học được liên kết nhỏ, nhưng readout-only cũng đạt100%. Hidden plasticity chưa tạo lợi ích đo được ở bài này. Forgetting trung bình full20,8–23,9 điểm %, cao hơn ngưỡng3 điểm. Vì vậy dừng mở rộng.

## Chưa thực hiện — giữ rõ trong backlog

- Huấn luyện ngôn ngữ10M token và model10M/30M/100M tham số.
- Bộ đánh giá đa năng600+ bài và matched commercial comparison.
- Low-bit/pretrained LLM+RAG baseline trên cùng tác vụ ngôn ngữ.
- Runtime C++ độc lập, hard memory cap, quantization, iGPU scheduling.
- Kiểm thử feedback noise thực tế ngoài giả lập Phase 2.1, 8h endurance, điện năng đo thật.

Lý do: gate nghiên cứu bắt buộc đã không đạt; không được diễn giải những mục này là hoàn thành.

Tra cứu: [Phase 2.1](research/08_PHASE21_RESULTS.md), [Phase 2 lịch sử](research/06_FEASIBILITY_VERDICT.md), [run index](../runs/INDEX.md), [bản đồ](PROJECT_MAP.md).
