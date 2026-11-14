# Kế hoạch nối ASRprofile thật vào runtime — chưa áp dụng nguồn đang khóa

Phạm vi engineering đã được người dùng yêu cầu: tận dụng VRAM bằng thành phần
có lợi ích đo được, một gateway/một nativeVLM, nguyên precision, vận hành gọn.
WhispersmallFP32 chỉ được cân nhắc sau pacedsoak1800s/independentaudit qua;
profileHermes hiện vẫnbase. Không training/newlocalrule/LoRA hoặc mở rộng model.

## Khoảng trống source hiện tại

HybridConfig có asr_revision nhưng không kiểm/đọc trường này từconfig.json.
NativeBackends.start/transcribe hardcode models/native/whisper-base-f32.bin.
verify_native_storage yêu cầu đúng3export gồmbase; vì vậy chỉ đổi revision
hoặc đổi tênfilebase thànhsmall sẽ báo sai identity/integrity. Không làm vậy.
Source8h/ASRsoak đang khóa phải kết thúc và audit trước mọi edit vào các file này.

## Thay đổi cần review/test trên bản sao trước

- Một cấu hình asr_model_name mặc địnhopenai/whisper-base và revisionpin khớp.
  Allowlist chỉbase/small với nativeFP32source/exportSHA đã có receipt. Unknown/
  mismatchedrevision bị từ chối, không âm thầm tải/chọn model khác.
- Một resolver trả đúngfile/modelidentity cho start,transcribe và audit. Filebase
  giữ nguyên; small dùng candidatepath hiện có, khôngcopy/rename giảbase.
- Storagecheck giữ manifestVLM/baseline nguyên nghĩa, bổ sung receiptcandidate
  độc lập khi chọnsmall; kiểm size/hash/allFP32/noquantization/pruning từ source
  đã audit. Không dùng qualityflag chưađượcchứng minh thay checksum.
- Runtimeconfigload/telemetry/toolUI ghi đúng model thực thi. Một chunkASR dùng
  một worker như cũ, không thêmproxy/server. Logprecision/device kiểm thực tế,
  không chỉ khai báo GPU trongconfig. CPUmode mặc địnhbase; smallCPU chưa được
  resourcequalified, phải báo chưa hỗ trợ thay vì lénfallback hoặc đổiprecision.
- Audioheadroom projection dựa profile/modelresourceprobe; GPU vẫn stop/yield
  khi không đủ physicalreserve. Không unload nativeVLM vì muốn testsmall trên
  profile đãđượcđo giữcùngVLM, không silentlymove layer sangCPU.

## P0 kỹ thuật tối thiểu

1. Unitresolver/config: base/default vàsmallpin đúng, mismatch/pathescape/missing/
   tamperedexport từ chối; không đọcnhầm basekhi configsmall.
2. ExistingstorageVLM/base, CPU/GPU/persistence và frontendusage tests tiếp tục
   qua. Không sửa nghiêncứu/thresholds cũ để đọc candidateformat khác.
3. Nativeactualcall trên ownedFLEURSsample đã dùng: chọnsmall tạo command/path
   đúng/hash/ftype0/CUDA0; outputtranscript/latency lêntrace. Đây là integration
   check, không thêm independentqualitysample.
4. Save/load/rollbackALIF+controller nguyênstate; chỉ config/ASRsourceprofile thay.
   Wake/sleep/video→ASR→chat không mở thêmVLM hoặc lẫn foreground/background.

P0budget≤5phút/≤4actualASRcalls/≤4nativeVLMcalls, mỗirequest cótimeout. Khóa exact
ownedcases/source trước output, unique immutable run vàdata-onlyauditor. Stop
on source/integrity/device/reserve/persistencefailure, giữbase/livebackup.

## Điều kiện áp vào Hermes

Chỉ sau hai worker554c19 và292b32 thực terminal, finalaudits khép đúngphạmvi.
Archive/checkpointlive trước gracefulownedservicerestart; không dừngR13/ứngdụng
khác, không sửa main hoặc import labweights. Áp configsmall chỉ khi P0 qua và
probe tích hợp sourceidentity/resourceguards đúng. Có một configđểrollbackbase,
không thêm modelID thứhai trongHermes; debug ghi profileASRđểbảo trì.

Sau deploy cần đo mixedsession tiếp theo có correcteddynamicclientroots, đủ
chat/videoASR/learning/rollback trong8h hoặc báo rõ chưa qualifiedproduction.
Không lấy30mincomponentsoak thay8hfullprofile, không hứa QA/causelearning tốt
hơn nếu chỉ WERgain. GoalSNNadvantage/retention vẫn độc lập với ASRprofile.
