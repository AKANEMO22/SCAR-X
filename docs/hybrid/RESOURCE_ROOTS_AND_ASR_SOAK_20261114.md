# Bản sửa bộ đo đã có P0; Whisper-small đang kiểm thành phần

**Chưa cài resourcepatch hoặc thay ASR đang dùng trong Hermes.** Hai lượt thật
đang chạy độc lập: mixedendurance8h554c19 và pacedWhispersmall30phút292b32.
NativeVLM39372 vẫn là một worker dùng chung, ALIF live không nhận các dữ liệu
thử này làm nhãn. Rulehistory mới vẫn chờ duyệt, chưa code/train biến thể.

## Thu hẹp đúng nguyên nhân của thiếu resourcecoverage

[P0protocol](../../experiments/hybrid-resource-client-roots-p0-v1.md) và raw:
[d0a75f](../../runs/20261114T074426Z-client-roots-p0-v1-d0a75f/result.json),
[0c0835](../../runs/20261114T074607Z-client-roots-p0-v1-0c0835/result.json),
[da6f4c](../../runs/20261114T081540Z-client-roots-p0-v1-da6f4c/result.json).

Phép kiểm thực cho thấy discovery gốc **vẫn tìm đúng gateway47768 khi bật**.
Vì vậy giả thuyết phải thay toàn bộ nhận diện client không được xác nhận; đã
thu hẹp patch. Khoảng trống của lượt8h đến từ việc truyền danh sách client ban
đầu cố định. Lỗi bổ sung đã chứng minh: khi include_hermestrue, discoveryrefresh
ghi đè explicitroot35036, nên một shared-native monitor có thể bỏ sót VLMowner.

Bản patch cuối chỉ:

- Giữ explicitroots và clientroots discovery riêng, hợp nhất trước đo.
- Giữ cả root được thêm qua API `extra_pids.add` đang được gateway dùng.
- Báo rõ hai tập trong telemetry; không đổi accounting/cap/precision hoặc
  discovery hiện có đang hoạt động đúng.

P0cuối dùng actualPID/create-time của gateway47768, VLMowner35036/worker39372 và
private lab41904. Constructorroot và root gắn thêm còn sau forcedrefresh của
chính monitor cô lập. Không gọi screen/capturetool, không inference/học hoặc
sửa `src/scarx/hybrid/resources.py` đang khóa. Candidate trước0c0835 được ASRsoak
dùng với constructorroots; candidate cuối thêm compatibility cho publicsetAPI.
Hai source/hash được giữ riêng, không gọi chúng là cùng một profile đã deploy.

[Patch recipe](../../integrations/hermes/resource_client_roots_patch.py) đúng SHA
của source gốc; [helper identity](../../src/scarx/evaluation/hermes_resource_roots.py)
chỉ dùng trong audit/P0, không thêm một lớp discovery mới vào production.
Sau hai job terminal và finalaudit, mới áp dụng patch rồi kiểm backendreload/
measurement. Không lấy P0 này làm peakRAM toàn hệ hoặc bằng chứng giảm bộ nhớ.

## Kiểm thành phần sử dụng thêm VRAM có ích

[ProtocolASR](../../experiments/hybrid-whisper-small-paced-soak-v1.md),
[runner](../../scripts/soak_hybrid_whisper_small.py).
Whispersmall giữ nativeFP32 và tensor/source parity đã kiểm trước. Dùng lại
32file speechFLEURSDEV đã khóa, tổng345,76s audio duy nhất; phát nhịp bắt đầu
theo duration nguồn và lặp. Không download/modeltrain, không thêm ASRservice
hoặc modelHermes mới. Một ownedCLIprocess mỗi audiochunk, CUDA0/logftype0.

Preflight đầu [67183e](../../runs/20261114T075330Z-whisper-small-soak-v1-67183e/result.json)
dừng trước ASRspawn vì xóa lock khi handleWindows còn mở. Giữ lỗi và charged
attempt1; không hoàn/reset ngân sách. Đã tách helper budget, test release khi
success/error/exhaustion, chỉ xóa lock cũ sau khi xác minh job chết và không có
soak khác. Ledger giới hạn256attempts chung, không reset theo runID.

[Preflight3f9957](../../runs/20261114T075849Z-whisper-small-soak-v1-3f9957/preflight-acceptance-v1.json)
đã hoàn tất30,234s:3CUDAFP32calls,31,98audioseconds,9activefreeVRAMsamples,
minimum1.434MiB, RTFp95~0,204. Explicitroots35036/41904 và discoveredclients
35648/47768 đều nằm trong group đã kiểm. Đây là mechanicalpreflight, không
thêm3mẫu chất lượng độc lập hoặc chứng minh hiểu video.

Lượt [292b32](../../runs/20261114T080220Z-whisper-small-soak-v1-292b32/manifest.json)
chạy30phút bằng detachedhiddenworker **PID18444**, create-time trong
[processidentity](../../runs/20261114T080220Z-whisper-small-soak-v1-292b32/process-identity-v1.json).
Launcher4580 không được dùng thay cho liveness của actualworker. Source và
monitoroverlay0c0835 khóa riêng; không đụng source của lượt8h. Khi production
serving, job nhường trước chunk mới và có thể dừng ownedASR đang chạy khi phát
hiện foreground. Đây là cơ chế best-effort, không phải cam kết QoS mọi tình huống.

Gate thành phần giữ như protocol: đủ1800s,≥80completecalls,≥1000audioseconds,
RTFp95≤1, không lỗi/OOM/CPUfallback, sampledreserve≥512MiB và đúng resource
scope. Chưa finalreceipt thì chưaPASS. Lặp speech không trở thành video liên
tục đa dạng, đồng thời8hlab là mixedGPUconfound phải công bố. Không chấm lại
primaryquality trên các lần lặp như những mẫu độc lập.

~~~powershell
.\local\hybrid-v1\runtime-env\Scripts\python.exe scripts/monitor_hybrid_asr_soak.py runs/20261114T080220Z-whisper-small-soak-v1-292b32
~~~

Monitor chỉ đọc PID/create-time, args, source/overlayhash, heartbeat/finalreceipt;
không tự restart nếu quan sát chậm. Các số tiến độ trong tài liệu là lịch sử,
trạng thái hiện tại lấy từ monitor. ASRHermes vẫn là base; componentsoak qua
cũng chưa đủ để promote trước khi sửa/kiểm configuredmodelpath/identity/integrity
và đo profile tích hợp thật sau khi source8h được mở khóa.

## Quan sát và kiểm thử

Localhost tự hiện ASRtranscript/latency/RTF thật, khoảng trống resource scope và
phân biệt component với model live. Các phép đo đã xong được gom vào details;
hai job đang chạy hiện trước. HeaderRAM ghi “nhóm đang đo”, tránh hứa toàn hệ
khi coverage chưa được xác nhận. Heartbeat cũ chỉ cảnh báo phải kiểm PID,
không suy thành jobđãchết. Không thay rendererHermes hoặc làm mất draft đang gõ.

270Python/native/CUDA checks và20Node checks của checkout qua. P0resourcepatch
được kiểm bằng actualprocessmembership; chưa kiểm productionactivation vì
source vẫn khóa. Các greenchecks không thay quality/retention gates hoặc
[kết luận SNN/RAG hiện tại](RAG_SNN_AND_VRAM_20261114.md).
