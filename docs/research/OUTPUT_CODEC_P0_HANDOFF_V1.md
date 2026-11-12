# SCAR-X — bàn giao codec đầu ra P0 cô lập

**Quyết định: `P0_TECHNICAL_PASS_NOT_TRAINING_OR_RELEASE_APPROVAL`.**

Người dùng chỉ duyệt P0 theo `BASIC_KNOWLEDGE_CODEC_PROPOSAL_V1.md`. Kết quả dưới
đây kiểm chứng đường mã hóa/giải mã và trạng thái của bản sao nghiên cứu. Chưa có
bằng chứng QA, học representation, khái quát hóa, unknown detection hay retention
mới. Trạng thái sản phẩm vẫn **`NOT_PRODUCTION_READY`**.

## Phạm vi và mã nguồn

- Nhánh: `codex/scarx-output-codec-p0-20261111`.
- Commit triển khai được dùng cho lượt kiểm cuối:
  `1f4879212da86c5272c2c0f07969f0191bc28314`.
- Protocol trước đo: `experiments/output-codec-p0-v1.md`;
  ghi nhận sai số ban đầu: `experiments/output-codec-p0-numerical-note.md`.
- Prototype: `src/scarx/evaluation/output_codec_p0.py`; fixture continuation:
  `src/scarx/evaluation/output_codec_p0_validation.py`.
- Runner: `scripts/run_output_codec_p0.py`, `scripts/output_codec_p0_cold.py`;
  regression: `scripts/regress_output_codec_p0.py`.
- Test: `tests/test_output_codec_p0.py`.

Một quần thể recurrent 48 neuron ALIF được sao chép từ checkpoint public G1.
Topology, phương trình neuron, STDP và cập nhật readout giữ nguyên. Codec V2 chỉ
đổi token đầu ra số nguyên/thập phân có dấu; không đổi tokenizer của câu hỏi,
encoder context, không có answer lookup, Transformer, pretrained model hay
backward/BPTT mới. Model vẫn tự chọn token từ readout khi gọi `generate`.

Vocabulary cũ 24 token được mở rộng thành 55 token, trong trần 256. Inventory lấy
từ target train, vocabulary cũ và các câu UNKNOWN đã đăng ký; không chọn theo kết
quả final. Token mới có input weight bằng 0, output weight bằng 0, bias −6 và bị
mask cho đến khi bật rõ ràng. Context/image input columns và digit output rows
được ánh xạ lại; ID của từ cũ được giữ.

## Bằng chứng kỹ thuật

Lượt cuối: `runs/20261111T171706Z-output-codec-p0-v1-ecc876/`.
Receipt: `artifacts/output-codec-p0-v1.json`.

| Kiểm tra | Kết quả |
|---|---|
| Coverage target, cả hai inventory | 320/320; cùng 55 token |
| Encode → render, legacy-expanded | 192/320 chuỗi giữ đúng |
| Encode → render, numeric-atom V2 | 320/320 chuỗi giữ đúng |
| Context cũ, word probabilities và neuron state | Sai số lớn nhất 0 trên fixture đã đăng ký |
| Digit readout và neuron state sau remap | Sai số lớn nhất 0 trên hai fixture 8×8 |
| Tiếp tục giữa fixture ở process mới, cả V1/V2 | JSON continuation và hash state trùng hoàn toàn |
| RNG, replay selection/cursor, eligibility, stable weights | Lưu/khôi phục và continuation exact trong fixture |
| Guard reject / exception / checkpoint lỗi | Khôi phục toàn snapshot; corruption bị từ chối |
| Bộ test P0 | 31 passed |
| Peak RSS process tree, lấy mẫu mỗi 20 ms | 172.41 MB |
| Peak committed/private process tree | 142.23 MB |

Các chuỗi ở bảng là **đáp án tham chiếu đi qua codec**, không phải câu trả lời do
model sinh đúng. Chỉ sinh tối đa bốn token của một prompt P0 tổng hợp để so sánh
continuation, không chấm kiến thức. Không có train dataset, final-quality
inference, AGY/Qwen call mới hay mô hình được chọn theo accuracy.

Runner thực hiện 11 bước cập nhật fixture theo số đếm logic, hoặc **17 lần gọi
cập nhật thực tế** khi tính cả việc chạy lại continuation ở process mới. Đây là
các fixture trạng thái nhỏ, không phải một curriculum. Số đó chưa tính các bước
fixture nội bộ của pytest. Thời gian P0 khoảng 2.13 giây trong vùng monitor.
Windows áp trần commit 2 GB cho từng process; tổng process tree được đo bằng lấy
mẫu, chưa có trần cứng cho tổng shared memory. Không dùng iGPU. Số đo này không
chứng minh runtime production hoặc stress dài hạn đạt 2 GB.

## Sai lệch và lỗi được giữ lại

Mở rộng ma trận rồi dùng GEMV toàn chiều ngay lập tức làm voltage lệch
`4.76837158203125e-7`. Phương án đó **không qua exact parity**. Prototype cuối giữ
thứ tự tính của khối input và output cũ, tính các token mới bằng khối riêng.
Ngưỡng parity vẫn là 0; không đổi ngưỡng để nhận PASS.

Lượt ban đầu `20261111T170938Z-output-codec-p0-v1-e3f834` cùng snapshot và 24 test
được giữ nguyên. Review tiếp theo tái hiện năm khoảng trống kiểm tra state trước
khi sửa. Chi tiết nằm trong `OUTPUT_CODEC_P0_REVIEW_NOTES_V1.md`; lượt cuối có
31 test và thay thế receipt sơ bộ khi đánh giá code hiện tại.

Loader mới kiểm header/shape/dtype trước khi materialize NPZ; từ chối NPY khai
báo mảng khổng lồ, float64, object, duplicate ZIP members, metadata lạ và checksum
sai. Checkpoint không có codec metadata tiếp tục là legacy V1. Serving loader
không nhận wrapper P0. Các mutator legacy không tương thích bị chặn thay vì tự
đổi vocabulary hoặc học bằng tokenizer cũ.

## Hồi quy và bảo vệ G1

Lệnh tái lập, tại worktree này:

```powershell
python -X utf8 scripts/run_output_codec_p0.py
python -X utf8 scripts/regress_output_codec_p0.py
```

Mỗi lệnh tạo run ID mới; không ghi đè run cũ. Runner lưu source snapshot,
manifest SHA-256, câu lệnh, stdout, exit code, JUnit và RAM. Bộ hồi quy được chia
nhóm để test Word control có process Torch riêng không chạy dưới parent đã nạp
Torch; assertions và ngân sách 2 GB giữ nguyên. Build lại kernel CPU hiện có chỉ
để chạy regression/native parity, không tối ưu C++ hay mở nhánh học mới.

Kết quả hồi quy cuối được lưu riêng tại
`artifacts/output-codec-p0-regression-v1.json` và run được receipt chỉ tới.
Run `20261111T171709Z-output-codec-p0-regression-v1-baa310` qua **354/354 test**
trong 76 file, không failure/error/skip. Bốn chế độ CPU/native
`frozen/readout/stdp/full` đều chạy và qua. Peak process-tree RSS 923.10 MB,
committed/private 1,449.70 MB, thời gian khoảng 125.50 giây; không ghi nhận vượt
2 GB. Snapshot hash của các file neural/app/learning và public release không đổi
trước/sau hồi quy. Các test control cũ giữ trong nhánh đánh giá, không thêm
backward vào prototype hoặc learner G1.

Không sửa `src/scarx/neural`, `src/scarx/app`, `src/scarx/learning`, checkpoint
public G1 hoặc frozen dataset. Không đọc/ghi `local/generation1`, restart Hermes,
train model live, push hay merge main. Mirror báo skip vì original checkout đang
ở main; chứng cứ và nhật ký activity thật nằm trong worktree nghiên cứu. Chưa
đưa kết quả P0 lên phiên localhost live.
Original checkout ở `e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541`, sạch khi kiểm
trước/sau lượt này. Đây là commit đã có từ bên ngoài, không phải commit do P0
merge vào main. Diff nhánh P0 so với base không đổi các đường dẫn production nói
trên hoặc frozen pool.

## Những điều chưa được chứng nhận

1. P0 chỉ hỗ trợ public H48, active 48. Không đổi capacity policy 48–128 của G1;
   chưa chứng nhận codec migration khi đã cấp thêm neuron.
2. `PlasticCore`, importance/optimizer hoặc controller state ngoài schema bị từ
   chối. Không có tuyên bố chúng đã được migrate hay có thể rollback trong app.
3. Replay ở đây là tối đa bốn record fixture và RNG/cursor; không phải cơ chế
   replay/consolidation tự động của sản phẩm. SQLite/Hermes migration ngoài P0.
4. Codec V2 khắc phục biểu diễn thập phân; không sửa credit assignment, readout
   overwrite hay chứng minh hidden synapses học representation hữu ích.
5. Task unknown chưa có ví dụ train được duyệt. Không lấy unknown dev/final để
   bổ sung train hoặc gọi policy từ chối là kiến thức neural đã học.

## Bước tiếp theo cần duyệt riêng

Chưa thực hiện P1/P2. Nếu mở nghiên cứu chất lượng, cần protocol khóa trước:
common refit, frozen/readout-only và correct/zero/shuffled hidden controls,
free generation không đưa nhãn vào inference, seed độc lập và các contrast.
V2 rút số token target nên phải báo cả ngân sách theo trải nghiệm và ngân sách
khớp số update/compute; không quy toàn bộ lợi ích codec cho hidden learning.

Giữ nguyên các cửa factual 80%, paraphrase 75% và unknown 95%; acquisition/final
65%, mean forgetting tối đa 3 điểm phần trăm và worst tối đa 5 điểm. Codec P0
không cấp quyền mở retention pilot, tăng neuron/model, train corpus lớn hoặc
đưa learner vào production. **GO cho P0 kỹ thuật; `NEEDS_RESEARCH_APPROVAL` cho
phần nghiên cứu tiếp; sản phẩm vẫn `NOT_PRODUCTION_READY`.**
