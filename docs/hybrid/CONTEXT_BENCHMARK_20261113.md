# Benchmark context Hybrid — 16K / 24K / 32K, 13/11/2026

**Quyết định tại trần 4 GB đã khóa: DO NOT INCREASE.** Giữ 16.384 token mặc
định. Sau khi đo xong, người dùng cho phép vận hành lên **4,5 GB**; đây là
ngân sách cho lượt mới, không sửa kết quả hay tiêu chí của lượt đã hoàn tất.
[Amendment mới](../../experiments/hybrid-v1-resource-amendment-4_5gb.md).

## Phạm vi và cách đo

Run `20261113T144438Z-hybrid-context-v1-a7312f`, base
`b1871172c63c60eec8eb6b46f3d862310d2c4a4b`.
[Protocol đã khóa](../../experiments/hybrid-context-engineering-v1.md),
[manifest](../../runs/20261113T144438Z-hybrid-context-v1-a7312f/manifest.json),
[toàn bộ đầu vào và nhãn](../../runs/20261113T144438Z-hybrid-context-v1-a7312f/cases.json),
[receipt gốc](../../runs/20261113T144438Z-hybrid-context-v1-a7312f/receipt.json),
[chấm lại từ SSE thô](../../runs/20261113T144438Z-hybrid-context-v1-a7312f/independent-audit.json).

Qwen3.5-0.8B frozen, native BF16/F32 và vision projector nguyên precision;
llama.cpp b12360, CPU 4 thread, batch128/ubatch64, một slot, KV BF16. Chỉ đổi
context 16.384/24.576/32.768. Temperature0, seed20261113, output<=96, thinking
tắt như profile đang phục vụ. Không nén trọng số, huấn luyện, cập nhật synapse,
replay, web lookup, retrieval, compaction hoặc context shift trong bài đo.

Đây là pilot **engineering DEV, văn bản trực tiếp qua native VLM**, không phải
benchmark toàn bộ agent Hermes, video, continual learning hay model online.
Hermes và process con vẫn nằm trong RAM được tính. Controller đo và một ALIF
H128 độc lập nằm trong process đo; phần thiếu so với RSS gateway trước đó được
dự phòng riêng. Không đánh đồng phần dự phòng với RSS đo thực. Shared GPU được
cộng bảo thủ; không bảo đảm đã khử trùng mọi trang dùng chung. R13 không bị dừng
hoặc thay đổi; độ trễ chịu ảnh hưởng tác vụ khác trên cùng máy.

Cases/labels, templated token counts, protocol và script được hash trước sinh
đáp án. Manifest SHA cases:
`59ad57e9b44951a9b5b388b03adf5b87f398b6f3026512292482e957b955e000`.
Sáu bài chung giống nhau ở ba mức; hai bài dài bổ sung chỉ kiểm tra khả năng
nạp thêm thông tin. Không tính câu không vừa cửa sổ là trả lời sai.

## Kết quả

| Context thực | Đúng / 6 bài chung | RAM đỉnh tính cả dự phòng | Bài 22.020 token | Bài 30.011 token |
|---|---:|---:|---|---|
| 16.384 | 2/6 | 3.785.834.496 byte | Không nạp: vượt cửa sổ | Không nạp: vượt cửa sổ |
| 24.576 | 2/6 | 3.874.160.640 byte | Timeout 180,031 giây, chưa có đáp án | Không nạp: vượt cửa sổ |
| 32.768 | 2/6 | 3.969.126.400 byte | Dừng sau 100,953 giây vì hết ngân sách tổng | Chưa chạy; hết ngân sách tổng |

Đầu ra **trùng nhau trên cả sáu bài chung**. Tổng 18 câu hoàn tất, hai yêu cầu
dài chưa ra đáp án. 32K không được dành đủ 180 giây cho bài22K: không dùng
100,953 giây đó để tuyên bố nhanh hơn 24K. Bài30K chưa được kiểm chứng.

| Bài chung | Đáp án đúng | Đáp án ở cả ba mức |
|---|---|---|
| Sửa cấu hình tiếng Việt | port9237, days6 | Đúng |
| Toán: còn12 đỏ, tăng9 xanh thành18 | red12, total30 | red12, total26 — sai |
| Chạy đoạn Python với pop/append/sum | answer9 | answer10 — sai |
| Không có dữ kiện năm sinh | birth_year:null | Đúng |
| Nối thùng→đội→địa điểm sau khi chuyển | room:Amber, keeper:Ruma | room:Cedar, keeper:Ruma — giữ địa điểm cũ |
| Vị trí token trước/sau chuông | before:Amber, after:Birch | before:drawer Cedar, after:drawer Amber — sai quan hệ thời gian |

Hai bài dài dùng 12.018 và 12.007 token, đều nằm trọn trong 16K. Token usage
thực khớp input đã khóa, cache_n=0; log native ghi không truncation cho các câu
hoàn tất. Vì vậy lỗi này không có bằng chứng là do thiếu chỗ trong cửa sổ.

| Context | Ra token nội dung đầu: bài quan hệ12K | Bài thời gian12K |
|---|---:|---:|
| 16K | 136,656s | 165,110s |
| 24K | 143,578s | 154,453s |
| 32K | 144,250s | 149,078s |

Đây là prefill lạnh, cache bị tắt để so cùng điều kiện. Không suy rộng thành
độ trễ mọi lượt chat có prefix cache. Median trên cả6 câu lần lượt0,4145 /
0,453 / 0,3825s bị chi phối bởi4 câu ngắn; không dùng nó để che hai lượt chờ
hơn2phút. Sáu bài tổng hợp có chủ đích quá ít và không phải mẫu ngẫu nhiên để
kết luận thống kê về năng lực tổng quát hoặc ngang model online.

## Kiểm toán và lỗi harness được giữ lại

`scripts/audit_hybrid_context_benchmark.py` chấm lại nội dung từ mọi SSE chunk,
JSON keys/values/types, token count, cache reuse và tiêu chí đã khóa. Không có
sai lệch answer/label/token/hash được phát hiện. Đây là chương trình chấm lại
riêng, không phải tuyên bố đã có người/agent độc lập review.

Runner có race sau `terminate()`: nhãn tổng arm24K/32K ghi `COMPLETED` trong
khi bản ghi từng câu đã ghi `TIMEOUT`. Receipt gốc được giữ nguyên. Auditorv2
ưu tiên trạng thái từng câu, kết luận cả hai arm có timeout. Regression kiểm
trường hợp này; không biến nhãn tổng thành bằng chứng tất cả câu đã hoàn tất.
Số sơ bộ3/4 trong commentary đã được đính chính2/4; số cuối lấy từ raw.

Trước benchmark, phiên live16K trước đó đã bị guard dừng ở4.031.107.072byte;
[receipt riêng](../../runs/20261113T144438Z-hybrid-context-v1-a7312f/preexisting-live-guard.json).
Không gộp sự cố cũ đó vào peak của arm16K. Sau benchmark, gateway được khôi
phục ở16K/4GB rồi mới áp amendment4,5GB theo phê duyệt tiếp theo.

## Quyết định và bước tiếp

Theo protocol4GB, yêu cầu peak<=3,85GB, đủ6 bài chung, không giảm điểm, ít nhất
một bài vượt16K trả lời đúng và giới hạn độ trễ. 24K/32K không đạt headroom và
không có đáp án đúng bổ sung. **Không tăng context mặc định.** Không thay đổi
gates sau đo, không gọi lỗi của VLM là bằng chứng SNN hidden không học được:
ALIF không học và không tạo đáp án ngôn ngữ trong phép đo này.

Tại4,5GB, các peak cũ dự kiến để lại thêm khoảng626MB ở24K và531MB ở32K.
Đó là dự toán từ bài văn bản đã đo, không phải PASS mới cho video/audio,
checkpoint/rollback hay workload dài. Cần đánh giá riêng lợi ích trên task thật,
chi phí prefill và phối hợp ảnh/âm thanh trước khi mở context mặc định.
Ưu tiên cải thiện dùng đúng bằng chứng, cập nhật sự kiện, kiểm kết quả toán/code
và ổn định RAM. Cửa sổ lớn hơn không tự biến thành học synapse hay trí nhớ bền.

Theo [config gốc của Qwen](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json),
max_position_embeddings là262144. Đây là metadata model; không phải chứng nhận
runtime laptop trong ngân sách này. Tham sốcontext/cache/stream trong
[tài liệu llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
được đối chiếu thêm với source b12360 trên máy. KV full-attention dự toán tăng
201.326.592byte từ16K lên32K; số đo toàn hệ ở trên mới là căn cứ vận hành.

Tái lập harness từ baseSHA + bundle `runs/.../source/`, assets có hash trong
manifest; chạy chuẩn bị tạo runID mới, không chạy đè run cũ. V1 giữ cap4GB ngay
cả khi default runtime đã lên4,5GB. Audit:

```powershell
python scripts/audit_hybrid_context_benchmark.py <bản-sao-run-chưa-có-independent-audit.json>
python -m pytest tests/test_hybrid_context_benchmark.py -q
```
