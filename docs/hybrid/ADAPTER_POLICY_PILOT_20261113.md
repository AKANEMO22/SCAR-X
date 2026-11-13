# Dạy chọn cách học / LoRA — pilot13/11/2026

**Kết quả: NOT QUALIFIED.** Đã dùng Antigravity tạo dữ liệu và thực sự chạy
ALIFlocal-learning pilot, nhưng chưa đủ điều kiện để tự tạo/kích hoạtLoRA.
Không đưa9checkpointnghiên cứu vào modelHermes đang phục vụ; VLM vẫn frozen.
Đây là tiếp nhận lại yêu cầu đã được người dùng gửi lại sau khi từng rút nhầm;
phần search/GPU được xử lý trước theo chỉ đạo tiếp theo.

## Dữ liệu và giới hạn nhãn

InstalledAGY được gọi2lần, đúng `gemini-3.8-flash-high --effort high`; init của
cả2lượt xác nhận model. CLI báo1.5.8 khi tạo (help probe trước đó1.5.5). Tổng
96wordings Việt/Anh,16mỗi nhóm: base, memory, repair, verify, reuse_adapter,
propose_adapter. Không nhét tênlabelLoRA/adapter vào phần prompt đầu vào.
Nguồn/streamhash/usage và kết quả48+48 được lưu ở
`runs/20261113T155741Z-adapter-data-v1-614f34`; rawCLIlogs nằmlocal cùngrunID.
Không đính kèm hội thoại/video/trọng số riêng vào prompt sinhdata.

AGY viết lời yêu cầu, **không quyết định nhãn đúng**. Nhãn từ contract riêng:
pipelinehỏng→repair; thiếu xác minh→verify; dữ kiện nhất thời có kho nhớ→memory;
baseđã đủ→base; adapterđã kiểm/chungbase/cógain/đủheadroom→reuse; thiếu kỹ năng
lặp lại và đủdata/resources/thờigianrảnh→propose. Cùngcâu hỏi với fault hoặc
unverifiedstate phải đổi hành động. Các con sốstate là **synthetic fixture**,
không phải kết quả đo realadapter; không được dùng làm bằng chứngLoRA có lợi.

Wordings có nghĩa gần nhau giữa nhiềuID. Đã nhóm lexicalduplicates trước khi
train:95clusters;716train/100DEV/256TEST, tổng1.072rows. Clusters không giao
giữa cácsplit; điều này **không bảo đảm semanticOOD**. CIlexicalfamily chỉ mang
tính thăm dò. Label/sourceID không đi vào encoder; structuredstate và prompt có.

Contractquyết định và tiêu chí giữ nguyên trong
[protocol trước training](../../experiments/adapter-lifecycle-v1.md).
Hai AGYcalls mất158,734s và242,328s wallclock. CLI đặt4mtimeout, script có255s
harddeadline gồm startup; so với câu chữ4min/call trongprotocol, lượt2 vượt
2,328s wallclock. Giữ số thật, không gọi hoàn toàn tuân thủ240swalllimit và
không mở thêm call bù. Budget đã đóng, có receipt ghi rõ đóng sau khi chạy.

## Thực sự đã học gì?

Pilot isolated dùng đúng LocalALIFH128/equations/eligibility hiện có, không
torch/autograd/BPTT, không đổi servingcore. Wrapperpurpose riêng ngăn coi nó
là controllerproduction.3seed270113/270127/270131; mỗiarm512bước, sameorder
trongseed. Learnedhidden/readout-only/frozen; không tune theoTEST sau nhìn điểm.

| Seed | Hidden học: balancedaccuracyTEST | Readout-only | Frozen | Đề xuất tạo/dùng sai /256 ở hidden |
|---|---:|---:|---:|---:|
|270113 |89,71% |75,17% |14,46% |23 |
|270127 |83,50% |71,62% |8,85% |21 |
|270131 |81,22% |73,28% |7,78% |13 |

Trọng sốhidden thực sự đổi, không chỉreadout; checkpoint coldnextupdate khớp
byte ở cả9arm. Chênh lệch có lợi cholearned trong bảng này là tín hiệu trên
syntheticcontract, **chưa chứng minh representation hữu ích cho ngôn ngữ hoặc
quyết định realLoRA**. Feature/normstatistics cũng tham gia pipeline; không suy
diễn từ weightdelta thành hiddenknowledge. Không có retention20stages ở đây.

Ngưỡng đã khóa: balancedaccuracy≥85% **mọiseed**,0unsafeproposal, vượt frozen
≥10pp và persistenceexact. Hai seed dưới85%; cả3cònunsafe. Vì vậyNOTQUALIFIED,
không hạ ngưỡng sau xem kết quả, không thêm mask sau đo rồi tuyên bố modelđạt.
Unsafe ở đây là định nghĩa contract: chọnreuse/proposekháclabel, **không phải**
một adapter đã thật sự được train/áp dụng sai. Không cóLoRAartifact được tạo.

Runtimepilot6,671s; peakprocessRSS47.616.000byte. Đây là bản sao nghiên cứu,
không phảiRAMtoànhệGPU. Source/archive/cases/2.304predictions/learningloss /
checkpoints ở [runbc7652](../../runs/20261113T165230Z-adapter-policy-pilot-v1-bc7652/receipt.json),
[kiểm lại label/split/metric](../../runs/20261113T165230Z-adapter-policy-pilot-v1-bc7652/metric-audit.json).
CItrongreceipt là familymean, **không phải CI của primarybalancedaccuracy**;
không dùng chúng để chứng nhận statisticalgate cho primarymetric. Không có
người/agentđộc lập review; đây là chương trình chấm lại riêng.

## Lỗi generator được phát hiện sau audit

V1 chọn counterfactual bằng substring `-0-`: ngoài variant0, cácvariantkhác
củafamilyIDindex0 cũng trúng. Nó tạo thêm48rows, total1.072 đã được chốt trong
data-onlyamendment **trước training**. Nhãn đúng theo state, không tạo split
leakage; nhưng không mô tả đây là256counterfactuals sạch đúngýđịnh. Giữ nguyên
run và sourceas-executed. Muốn sửa phải versiondataset/protocol mới; không sửa
và chạy lạiTEST đã nhìn. Budgetlock chặn thêm lần chạyv1.

## Phần chưa triển khai và đường đi tiếp

Người dùng muốn hệ biết khi nào cần/không cầnLoRA, tái sử dụng khi phù hợp và
phối hợp với memory/SNN. Pilot đã thử học phần phân biệt workflow, **chưa cung
cấp tính năng end-to-end**. Còn thiếu:

1. State lấy từ kết quả thật: lỗi có verifier, nhiều trải nghiệm độc lập, lượng
   dữ liệu kỹ năng có nguồn; không dùng confidence/tự phê bình làmgroundtruth.
2. MộtLoRAứng viênVLM với dữ liệu train/DEV/TESTkhácnguồn, đo realbase-vs-adapter
   skillgain/cost/retention. Hiện AGYwordings làdatacho controller, chưa làSFT
   corpus đã kiểm chứng để dạy VLM ăn nói/vision.
3. Registrybindbasehash/artifacthash/precision/purpose/evaluation, kiểm RAM/VRAM,
   singleactiveadapter/request, khôngswitchmidanswer, save/rollback/persistence.
   Không córegistryloaderqualified trongserving ở lượt này.
4. Dạy policy từ realoutcomes/utility trong tập hành động được runtime cho phép;
   đánh giázero-positiveunsafe và hidden-vs-readout/frozen trên dữliệumới.

LoRA thuộc nhánhVLM; ALIF vẫn học cục bộ. Sẽ ưu tiên tái dùng/cập nhật một
adapterđã kiểm, không sinh một adaptercho từng câu. Cơ chế tạo/áp dụng cần tách
khỏi bộ sinh câu trả lời và có kiểm chứng. Giới hạnGPU đã được người dùng bỏ,
nhưng phần cứng thật và resourcecost vẫn phải đo; CPUtarget4,5GB giữ nguyên.

Mã nghiên cứu: `src/scarx/evaluation/adapter_policy.py`,
`scripts/train_adapter_policy_pilot.py`, `scripts/adapter_data_factory.py`.
Không import cácmodule này vào serving. Statusđúng là đã có data và pilot học
thật, **chưa có khả năng LoRAautonomous đã được chứng minh hoặc triển khai**.
