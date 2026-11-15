# LoRA FP32 chạy được trên native runtime, chưa đủ kết luận retention

**Native portability qua bộ kiểm hẹp; chưa cài vào Hermes.** Adapter đã học32
bước ở pilot5c09fa được nạp trong **worker llama.cppb12360 riêng**, cùng native
Qwen0.8B BF16/F32, per-request scale0/1. Không đổi native server của Hermes,
không merge hoặc giảm precision. Worker thử đã được đóng theo PID/ctime/exe.

[Protocol khóa trước64calls](../../experiments/hybrid-lora-native-portability-v1.md),
[raw native run3dcddc](../../runs/20261115T080312Z-lora-native-portability-v1-3dcddc/result.json),
[independent effect/data audit04dce0](../../runs/20261115T080700Z-lora-native-data-audit-v1-04dce0/result.json).
Source trước run `410e90683355a8f4c44a2712154d08cd41095632` theo manifest; không chọn scale tốt nhất hoặc tune
prompt sau output. Bản Transformers pilot cũ giữ nguyên, không trộn backend.

## Kết quả paired trong cùng worker native

|Bộ câu hỏi|Adapter scale0|Adapter scale1|
|---|---:|---:|
|Sửa API,4DEV+16TEST|6/20|20/20|
|Trong đó16TEST|5/16|16/16|
|Toán4|2/4|2/4|
|JSON formatting4|4/4|4/4|
|Đọc nguồn4|4/4|4/4|

40câu trả lời sửa API được replay bằng ToolTalk thật: alarm/time và unchanged
other state khớp exact scores, không chỉ JSON parse.0 model/learning calls
trong auditor. Old-control answers được chấm lại từ raw text/JSON theo gold đã
khóa.12control có0 observed correct→wrong nhưng **không đủ mẫu chứng minh**
population regression≤3pp/no family>5pp. Không lấy số0/12 để đánh dấu joint
retention hoặc production PASS; chưa thêm20giai đoạn học hoặc long time return.

Hai câu toán vẫn sai ở cả hai scale:91−17 trả4/44 thay vì74;81÷3 trả9 thay vì27.
Không sửa output để làm đẹp điểm. Bảo toàn correct count không đồng nghĩa model
toán tốt; native answer thay đổi dù binary correct label giữ nguyên.

TEST repair paired gain68,75pp; exploratory bootstrap theo4 error-family cluster
CI95%[25;100]pp. Bốn lỗi/API và ít template, cùng16cases từ pilot đã xem; đây
là portability confirmation trên bộ đã biết, **không fresh unseen benchmark**.
Không suy thành hiểu tiếng Việt/Anh đa năng hoặc học SNN vượt RAG.

## GPU, tốc độ và giới hạn

64top-level generation requests×max64newtokens; không đổi decodebudget mỗi
scale.33,186s worker-run wall, physical dedicated GPU free ở điểm đo min
924.844.032byte=882MiB, trên512MiB. Đây là sampled telemetry, không continuous
whole-system stability proof. Runtime/native và worker thử đều dùngCUDA0,
hostSQLite/tokenization vẫnCPU, không modeCPU fallback.

Median request latency0,283s scale0 và0,323s scale1; p950,346s/0,346s. Cố định
thứ tự0rồi1, shared prompt cache có thể ảnh hưởng timing, nên chỉ mô tả số đo,
**không tuyên bố adapter nhanh hơn hoặc kết luận overhead chuẩn**.

Không tự tin `--lora-init-without-apply` đã để global scale0: GETlora-adapters
trong worker trả scale1. Mỗi request đều truyền scale0/1 rõ ràng theo source
API, nên phép paired dựa vào per-request override; không dùng global defaults
hoặc sửa chúng trên Hermes. Audit hiệu ứng portability thuộc worker riêng này.

Adapter F32GGUF và PEFT artifacts giữ nguyên hashes. Source/model base weights
không sửa; độc lập với ALIF updates. Live source58payloadRC14,27saved arrays/
learned metadata/RNG/replay/updates5 vẫn kiểm nguyên vẹn sau thử.

## Kết luận có thể sử dụng

-Có candidate adapter nhỏ chạy được trong native runtime và có hiệu quả trên
  bộ API repair này; không cần train thêm chỉ để xác nhận lại cùng portability.
-Chưa bật trong Hermes vì chưa đủ old/general control và retention evidence;
  không tự triển khai một adapter chỉ vì pilot100%.
-P1B SNN vẫn STOP trên action bank cũ do oracle bằngRAG ởTEST. Adapter gain
  không sửa thất bại đó và không được quy thành hidden-synapse improvement.
-Nếu mở nghiên cứu tiếp, cần task/action usefulness hoặc information path mới,
  tách preregistered data/split và giữ acquisition/CI/forgetting gates. Không
  thêm epoch/neuron trên bank đóng hoặc bỏ strong baseline để có điểm thắng.
