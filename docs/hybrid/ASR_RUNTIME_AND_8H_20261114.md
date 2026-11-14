# ASR-small đã chạy trong cấu hình Hermes; lượt 8 giờ cũ đã khép audit

**Kết quả hiện tại:** một gateway, một Qwen0.8B nguyên precision, ALIF H128 học
cục bộ và Whisper-small FP32 được chọn trong GPU config. Không LoRA, không
quantization/pruning hoặc đưa trọng số neural sang CPU. SNN-only, main và peer
R13 giữ nguyên. Đây là nghiệm thu runtime; chất lượng video/học kỹ năng chưa đạt.

## Bằng chứng vận hành

| Phần | Bằng chứng | Kết luận đúng phạm vi |
|---|---|---|
| 8 giờ trên profile base cũ | [554c19 result](../../runs/20261114T011841Z-shared-endurance-v1-554c19/result.json), [audit602b72](../../runs/20261114T091909Z-endurance-data-audit-v1-602b72/result.json) | 28.800,391s;96video/QA,16cold-GPU/rollback checks,100privateupdates/20replay,0critical errors; engineering gate và đối chiếu dữ liệu qua |
| ASR-small thành phần | [292b32](WHISPER_SMALL_SOAK_RESULT_20261114.md) | 30phút,165CUDAFP32calls,p95RTF0,377,minfree1086MiB; đã khép component audit |
| Chọn ASR bằng backend mới | [57a681 private result](../../runs/20261114T090807Z-ASR-native-integration-p0-v1-57a681/private-result.json), [audit](../../runs/20261114T090807Z-ASR-native-integration-p0-v1-57a681/private-audit-v1.json) | 4actualcalls baseVi/smallVi/smallEn/baseEn; đúng path/SHA/FP32/CUDA0;13,906s,minfree1429MiB,privatefullstate/coldGPU/rollback exact |
| Nạp runtime/config live | [activation](../../runs/20261114T090807Z-ASR-native-integration-p0-v1-57a681/live-activation-result.json), [data audit](../../runs/20261114T090807Z-ASR-native-integration-p0-v1-57a681/live-data-audit-v1.json) | Gateway32012/native4728 sẵn sàng;3freshparitycalls qua,probe7+8→15;ASRsmall/storage đúng;21,25s cho toàn giao dịch stop/apply/restore/start/check |
| Trạng thái học qua restart | Cùng activation/audit và normal-controller-restore.log | Normal HybridEngine constructor khôi phục toàn payload exact trước background start; sau start mọi learnedarray/RNG/replay/non-controller metadata nguyên vẹn,updates5/hidden5/replay1 |
| Hồi quy | [regression logs](../../runs/20261114T090807Z-ASR-native-integration-p0-v1-57a681/regression) | 299Python/native/CUDA checks qua;22Node composer/trace checks qua sau cập nhật wording lịch sử |

96×21s = **33phút36giây cùng một clip lặp**, không phải8giờvideo liên tục
hoặc1–2giờnội dung đa dạng. 20mục tiêu arithmetic vẫn cùng một họ, không phải
20kỹ năng độc lập. Điểm QA chưa được chấm trong endurance. RootHermes mới bị
monitor cũ bỏ sót sau reload: peak cũ là nhóm đã đo, không nâng thành full-system
peak/CPU4,5GBPASS. Source và kết quả cũ được giữ nguyên; rootunion đã sửa ở live.

Không có ASRcall thứ5 hoặc VLMcall thứ5 trong57a681. Bốn ASRcall dùng backend
cô lập cùng exact source đã nạp; chưa có lượt phiên âm video live bằng small
trong nghiệm thu này. Một mixed-profile longrun mới vẫn cần protocol/preflight
riêng; không dùng8hbase hoặc30phútcomponent thay cho8hsmall.

## Thay đổi kỹ thuật đã cài

- Config đọc tên và revision ASR; resolver chung cho start/transcribe/integrity/
  telemetry. Base/small có file riêng, không đổi tên file để giả identity.
- Giữ nguyên FP32 và kiểm checksum/source receipt. Kiểm ftype/useGPU/CUDA trên
  đoạn log mới của từng call, không lấy log cũ để che CPUfallback.
- Small cần1800MiB trống trước nạp, giữ512MiB dự phòng khi chạy; thiếu thì yield/
  báo lỗi, không unload VLM hoặc tự chuyển ASR sang CPU.
- ResourceMonitor giữ explicit roots, attachment API và hợp thêm Hermes roots
  tìm được động. Tại audit, Hermes gateway thực tế được tính trong nhóm đo.
- Lease bảo trì PID/create-time/nonce chặn Hermes auto-start giữa giao dịch.
  Không thêm daemon/port. Installer chỉ ghi đúng6target đã review và backup
  private đầy đủ; không commit controller hoặc nội dung cá nhân.
- Localhost hiển thị receipt profile mới,4phiên âm thực và giới hạn phạm vi;
  không cộng labupdates vào live hoặc mô phỏng suy nghĩ.

Backup private ở `local/hybrid-v1/profile-backups/20261114T093248692755Z`.
File `private-full-live-checkpoint.bin` chỉ ở local, không đưa vào Git. Source
trước/sau và hash ở [57a681](../../runs/20261114T090807Z-ASR-native-integration-p0-v1-57a681).
Lease lỗi/partial fail-closed; nếu cần phục hồi, kiểm PID/create-time trước
khi xử lý đúng `.startup-maintenance.lock`, không xóa lock khi chủ còn sống.

## Dùng phần VRAM còn lại

Thêm năng lực phiên âm Việt là lợi ích đã đo: ViWER45,95→21,23% trên32speechDEV
trước đó; không có mẫu TEST mới ở vòng tích hợp. Small cũng chậm hơn base trên
cặp tiếngViệt ở P0:3,875s so với2,125s. Không tuyên bố tăng tốc ASR hoặc QA.
Hai phiên âm tiếngAnh ở P0 giống nhau. Video QA đã đo trước đó vẫn1/3.

Snapshot ngay sau start cóRSS nhóm4,063GB/privatecommit5,070GB và dedicatedVRAM
2,221GB; đó không phải peak khi ASR đang chạy. GPUmode bỏ capRAM theo phê duyệt,
CPU vẫn4,5GB. Mô hình ASR nạp theo chunk nên VRAM tăng khi cần rồi được nhả;
không giữ bộ nhớ vô ích để đạt100% occupancy. Native2B thiếu512MiBreserve;
LFM1.6B chưa qua qualitygate; không promote chúng hoặc phình SNN khi học cònfail.

CPUmode phải khai báo rõ `asr_model_name=openai/whisper-base` và revision
`e37978b90ca9030d5170a5c07aadb050351a65bb` cùng `execution_mode=cpu`.
GPUsmall dùng revision `973afd24965f72e36ca33b3055d56a652f456b4d` và modegpu.
SmallCPU chưa qualified và bị từ chối; không có silentfallback. Restart đọc lại
config; không cần đổi modelID trong Hermes.

## SNN có mạnh hơn RAG hoặc tự học cách sửa lỗi không?

**Chưa có bằng chứng.** [MatchedDEV](RAG_SNN_AND_VRAM_20261114.md) dùng cùng VLM:
readout46,53%,hidden37,50%,strong provenance/timeRAG100% trên contract hữu hạn.
Hidden−readout−9,03pp,CI[-31,25;9,72]pp; không suy thành universal ranking.
[Common probe](HIDDEN_BINDING_AND_COMMON_PROBE_20261114.md) xác nhận hidden đổi
representation nhưng gainDEV chưa chắc chắn. [Retention](PREDICATE_RETENTION_DEV_20261114.md)
còn fail; giữ nguyên ngưỡng3/5pp và5pp neural advantage.

Input hiện mất tín hiệu error/probe cần phân biệt nguyên nhân; verifier/recovery
còn thiếu bao phủ. Runtime phát hiện tool lặp hoặc sửa parser không đồng nghĩa
SNN đã tự hiểu nguyên nhân. Importance/stable trong Hybrid learner hiện còn
inert; prototype protection chưa đạt jointgate. Rulehistory ablation vẫn chờ
duyệt riêng, chưa code/train/deploy. ASRgain không giải quyết các nút thắt này.

**Quyết định:** giữ profile0.8B+small như bước kỹ thuật có giới hạn; PIVOT phần
representation/credit/retention. Chưa fullproduction/lifelong/selfrepairPASS.

## Mixed-profile tiếp theo: đã đăng ký, chưa là PASS

[Protocolsmall](../../experiments/hybrid-small-mixed-endurance-v1.md) giữ clip,
cadence,ALIF và cửa kiểm kỹ thuật cũ. Thêm kiểm actualsmallASR/currentHermesroot
trong mẫu đo, budgetphase rõ ràng; không nới gate hoặc thay P1/P2. Runner
hybrid_small_mixed_endurance.py dùng sameNativeVLM và privatelearner, monitor
kiểm đúng path+PID+create-time. Preflight phải actualterminal+dataaudit qua mới
cho phép8h cùngsource. Không lấy tiến trình còn sống làm kết luận thành công.
