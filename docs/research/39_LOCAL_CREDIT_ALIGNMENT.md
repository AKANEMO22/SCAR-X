# Tín hiệu học cục bộ có thực sự đúng hướng?

2026-11-10. **Đã hoàn tất audit chỉ đọc. Không huấn luyện, không cập nhật trọng số
nguồn.** [Protocol39](../../experiments/local-credit-alignment-audit-v1.md) đăng ký
trước khi viết code và đo. Đây là bước tiếp theo mà [audit38](38_WORD_TRAINING_OBJECTIVE_AUDIT.md)
để mở: audit38 chỉ đo **norm** và **cosine** của hướng cập nhật, rồi tự ghi rằng
"norm lớn không có nghĩa là tín hiệu hữu ích". Câu hỏi còn treo: hướng đó có thật
sự làm giảm loss không?

## Vì sao đây không phải chạy lại audit 38

Norm và cosine chỉ là thuộc tính tự thân của vector. Chúng không phân biệt được
"tín hiệu hữu ích" với "nhiễu lớn nhưng nhất quán". Audit39 áp **chính** hướng mà
quy tắc thật sẽ áp (trung bình trên các endpoint, dùng Adam/clip của checkpoint)
lên **một bản sao dùng một lần**, rồi đo lại NLL teacher-forced thật. Đây là phép
thử đối chứng ngược, không phải benchmark năng lực.

## Cách đo

Với mỗi checkpoint nguồn, mỗi prompt và chuỗi token chuẩn, tại các endpoint đã có
từ audit38:

1. `g` = hướng local tiềm năng hiện có (`potential_direction`, **không đổi quy tắc**).
2. Áp `g` lên bản sao dùng một lần bằng đúng Adam/clip của checkpoint, rồi đo lại NLL.
   `ΔNLL(g) = NLL_sau − NLL_trước`; **âm nghĩa là loss giảm**.
3. Control: đảo dấu `−g` (**khớp chính xác độ lớn**), hoán vị `permuted(g)` (giữ
   nguyên tập giá trị, phá ánh xạ synapse), và Gaussian khớp norm trước Adam.
4. Held-out: gộp hướng chỉ trên TRAIN, áp một lần, chấm trên dev. Nhãn dev chỉ dùng
   để chấm, không dùng để tạo hướng.

Nguồn được kiểm SHA256 trước và sau; mọi field được bảo vệ (weights/stable/topology/
head stats/Adam/counters/RNG) được assert không đổi. Mọi lần áp dụng xảy ra trên
`copy.deepcopy`; đối tượng nguồn không bao giờ bị sửa.

## Kết quả thực tế

Dev, 192 case, macro trên 8 family:

| Phép đo | Giá trị |
|---|---:|
| Endpoint đi đúng hướng (ΔNLL < 0) | **63,54%** |
| ΔNLL của hướng local | **−0,12913** |
| ΔNLL của hướng đảo dấu | +0,08724 |
| ΔNLL control hoán vị | +0,00397 |
| ΔNLL control Gaussian | +0,00707 |
| Chênh so đảo dấu | **+0,21636**, CI95% 0,19706..0,23708 |

Theo từng seed dev, tỉ lệ đúng hướng 0,6719 / 0,6875 / 0,5469; ΔNLL hướng local
−0,13268 / −0,12387 / −0,13083. Cả ba seed cùng dấu.

### Vì sao train chỉ có 9,38–10,94% đúng hướng — hiệu ứng sàn, không phải quy tắc sai

Bảng trên localhost cũng hiện train, và tỉ lệ đúng hướng ở train rất thấp
(9,38% / 10,94% / 10,94%). Con số này **không** nói quy tắc hỏng. Nó phản ánh
việc train đã **bão hòa**:

| Seed | NLL train trước khi áp | NLL dev trước khi áp |
|---|---:|---:|
| 36313 | 0,01566 | 0,90683 |
| 36319 | 0,01259 | 0,93625 |
| 36341 | 0,01046 | 0,76429 |

Train NLL đã gần 0 (khớp train 99,48% của vòng 36), nên **không còn dư địa để
giảm loss**; một bước cập nhật ở đó gần như chỉ thêm nhiễu quanh điểm tối ưu đã
đạt, và đảo dấu cũng cho kết quả tương đương về mặt thống kê. Dev còn dư địa lớn
(NLL 0,76–0,94) và đó là nơi tín hiệu thể hiện rõ. Vì vậy **chỉ đọc phần dev**
khi đánh giá hướng tín hiệu; phần train được giữ lại đầy đủ để thấy hiệu ứng sàn
chứ không bị cắt bỏ.

## Phát hiện về chính phép đo: control đã sai độ lớn

Đây là phần ít ai để ý và **làm thay đổi cách đọc mọi con số trên**. Adam chuẩn
hóa theo từng phần tử với moment/variance còn lại sau huấn luyện. Vì vậy "khớp
norm trước Adam" **không** bảo đảm khớp lực thực sự áp lên trọng số:

| Control | Độ lớn áp dụng thực / hướng thật |
|---|---:|
| Đảo dấu | **0,995×** (khớp chính xác) |
| Hoán vị | 3,52× |
| Gaussian khớp norm trước Adam | **8,13×** |

Nghĩa là control Gaussian và hoán vị đều **đẩy trọng số mạnh hơn nhiều** so với
hướng thật, nên mọi "gain so với control" bị nhiễu bởi độ lớn chứ không chỉ bởi
hướng. Vì thế **chỉ chênh so đảo dấu** là bằng chứng sạch, và nó chính là con số
được dùng để kết luận. Bản chạy đầu tiên mắc lỗi chụp norm *sau* khi cập nhật nên
mọi `applied_norm` đọc 0,0 và việc khớp control không kiểm chứng được; bản đó đã
đánh dấu `do_not_cite` và giữ lại làm lịch sử.

## Điều này có nghĩa gì — và không có nghĩa gì

**Có nghĩa:** quy tắc học cục bộ hiện tại **không phải nhiễu vô hướng**. Nó chỉ
đúng hướng giảm loss ở 63,54% endpoint và đảo dấu nó thì loss tệ đi rõ rệt. Đây là
bằng chứng cơ chế rằng dấu tín hiệu là đúng — điều audit38 chưa chứng minh được.

**Không có nghĩa:** đây **không** phải chứng nhận năng lực ngôn ngữ. Held-out mới
là con số nói về tổng quát hóa, và nó **gần như bằng 0**: ΔNLL trung bình
**−8,43e−05**, CI95% **−1,93e−04..1,91e−07**, tỉ lệ case dev cải thiện 1,56%–3,13%.
Tín hiệu đúng hướng trên từng bài, nhưng hướng gộp từ train **không** chuyển sang
bài khác. Đây khớp với bức tranh chung: train 99,48% / dev 35,94% của vòng 36.

Vì vậy kết luận trung thực là: **cơ chế tín hiệu đúng, nhưng credit không tổng quát
hóa.** Vấn đề không nằm ở dấu của tín hiệu, mà ở chỗ thông tin dùng để gán tín dụng
không mang nội dung chuyển được sang tình huống mới. Đây là lý do một learner mới
hoặc bài đánh giá mới vẫn cần protocol riêng — không tự động được mở từ audit này.

## Giới hạn

- Chỉ 15 checkpoint / 3 seed ghép cặp. 15 điều kiện **không** phải 15 seed độc lập.
- Chỉ một bước cập nhật tuyến tính cục bộ; áp lặp nhiều lần **chưa** được thử.
- Không tính và không xấp xỉ gradient (sẽ cần backward — ngoài phạm vi).
- 3 seed: CI chỉ mang tính thăm dò.
- Đây là chẩn đoán cơ chế, không phải cổng chất lượng; STOP35–37, P1/P2 và
  PIVOT_NO_SCALE giữ nguyên.

## Tái lập và quan sát

Chạy `python scripts/audit_local_credit_alignment.py`; kết quả ở
[run](../../runs/20261110T080426Z-local-credit-alignment-audit-v1-db01fc/report.md),
receipt `artifacts/local-credit-alignment.json`. 214 kiểm thử qua; peak RSS 53,40 MB,
commit 38,24 MB, cap 2 GB không đổi. Mục `#research-lab` tự hiện bảng AUDIT 39 với
ΔNLL, tỉ lệ đúng hướng và cảnh báo độ lớn control; số liệu này **tách khỏi** câu
trả lời tự sinh và không được dùng làm nhãn.
