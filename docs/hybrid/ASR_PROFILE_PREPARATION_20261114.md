# Bộ chọn ASR đã chuẩn bị; runtime live chưa đổi

**Đã có module profile và các bản sao tích hợp có kiểm thử; chưa áp vào source
đang khóa của lượt8giờ.** ASRlive vẫnWhisper-base. Phần này chỉ sửa việc chọn
đúng model/identity/device khi tích hợp Whisper-small đã qua componentsoak;
không thay ALIF, tín hiệu học, weights, VLM hoặc LoRA.

[Protocol chuẩn bị](../../experiments/hybrid-ASR-profile-resolver-p0-v1.md),
[data-only storage791db8](../../runs/20261114T084052Z-ASR-profile-resolver-p0-v1-791db8/result.json),
[shadow contracts và nguồn](../../runs/20261114T084052Z-ASR-profile-resolver-p0-v1-791db8/shadow-preparation-v1/result.json),
[kế hoạch tích hợp thật](../../experiments/hybrid-ASR-profile-integration-p0-plan-v1.md).

## Vấn đề được ngăn

Source hiện có revisionASR nhưng hardcode filebase. Chỉ gán revisionSmall hoặc
đổi tênfileSmall thànhBase sẽ làm hệ khai báo một model mà chạy model khác.
Module [asr_profiles.py](../../src/scarx/runtime/asr_profiles.py) giữ allowlist
tên/revision/nativepath/checksum/bytes của hai exportFP32 đã được audit.

- Defaultbase vẫn CPU/GPU được; small chỉ GPU vì CPUsmall chưa qua giới hạn
  RAM vận hành. Unknown/mismatchedrevision bị từ chối, không tự download/fallback.
- Nativepath và receipt phải nằm trong data directory được chỉ định; symlink
  escape, file thiếu/đổi size/checksum, metadatadtype/source xung đột đều bị chặn.
- Metadata đúng chỉ là provenance kiểm thêm; checksumfile thực tế vẫn bắt buộc.
  Cờ quality_qualified không bao giờ tự tạo qualityPASS.
- Checksum của cả hai file model đang lưu đã được tính thực, không nạp weights
  vào một neuralmodel hoặc thực thi ASR. Lượt này không có nativecall mới.

## Bản sao tích hợp làm gì?

[Recipe đúng nguồn](../../integrations/hermes/asr_profile_integration_patch.py)
sinh bốn bản sao config/backends/qualification/gateway theo SHA đã khóa; chưa
sửa các path đó trong src/scarx/hybrid. Module mới cũng chưa được source active
import. Candidate có các hành vi sau cần được giữ khi áp dụng thật:

1. Config đọc tênASR và revisionpin, dùng cùng resolver cho start/transcribe/
   integrity/telemetry. Hai filebase/small có path riêng, không giả identity.
2. Cache storagecheck theo path/size/mtime/ctime; khi asset đổi phải hash lại,
   lỗi thì xóa trạng thái đãverified. Đây không phải bảo đảm chống mọi race từ
   một tiến trình có quyền sửa file; không nhận securityclaim ngoài phạm vi.
3. SmallGPU cần1800MiBheadroom trước launch, thiếu thì báo lỗi; không unload
   VLM hoặc chuyển CPU để giả modeGPU. Trong lúc ASRđang chạy còn kiểm512MiB
   reserve, timeout và foregroundpreemption; finally chỉ dừng ownedASRworker.
4. Sau khi workerhoàn tất, kiểm **phần log mới của riêng call** bằng byteoffset,
   tối đa256KiB. Log cũ từng cóCUDA không đủ. Phải cóftype0, useGPUđúngmode và
   actualusingCUDAđúngdevice; CPUfallback dù exit0 cũng bị từ chối trongGPUmode.
5. Status và asr_result ghi tên/revision/modelpath/storagecheck/device-log thật,
   không chỉ in lại một cờGPU trongconfig.

## Đã kiểm và chưa kiểm

20checks qua: config/default/mismatch, originalFP32metadata, sourcepath escape,
tamper recheck, launchpressure không unloadVLM, actualcommand field/identity,
staleCUDAlog không che CPUfallback ở call sau, foreground/reserve polling.
Các nativeworkertransport của shadowtests **được thay bằng fixture**; kết quả
không phải20lượt ASRthật hoặc bằng chứng CUDAinference mới. Unitlog có format
khai báo rõ, không được xuất làm câu trả lời model trên trace.

Guardlog mới cũng đọc lại165log thật từ componentsoak292b32 và xác nhận tương
thích vớiCUDA0/FP32 đã đo, không gọi lại ASR. Shadowtests hiện import các bản
sao archived trongrun để tự chứa dữ liệu, không phụ thuộc một thư mục local
tình cờ tồn tại. Teststatus đầu thiếu engine.model.device trong fakegateway;
failedJUnit đã giữ, fixture được sửa theo contract có sẵn, không sửa runtime
để làm test qua. Source/profile/library/shadowtest cuối được lưu ở supplement
riêng, không ghi đè storageP0 hoặc manifest trước đó.

**Chưa kiểm actualruntime mới chọnSmall**, chưa gọi ASR hoặc VLM bằng source
backend mới, chưa áp config lênHermes. Budget tích hợp≤4ASR/≤4VLMcalls vẫn0.
Sau original8hterminal và independentaudit mới áp đúngbảnsource, chạy nghiệmthu
thật với checkpoint/configbackup, rồi quyết định profilelive. Componentsoak
hoặc20unitchecks không thay các điều kiện này.

GoalSNNhọc từ lỗi/transfer/retention vẫn chưa đạt; rulehistory mới chờ người
dùng duyệt trước nghiên cứu. Việc chuẩn bị ASR không được báo như cải thiện
representation hoặc khả năng tự nhận nguyên nhân toolloop.
