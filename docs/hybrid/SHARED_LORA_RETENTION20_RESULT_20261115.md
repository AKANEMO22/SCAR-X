# Một adapter học20chặng: giữ bộ API đã đo, chưa đủ kết luận lifelong/SNN

**Có dữ liệu retention20chặng cho LoRA; toàn protocol gốc vẫn STOP/NOT-PASS.**
Lõi ALIF không được cập nhật trong lượt này. Không dùng kết quả adapter để nói
SNN đã học representation, thắng RAG hoặc hết catastrophic forgetting.

[Protocol](../../experiments/hybrid-shared-lora-retention20-v1.md),
[raw run89ee2c](../../runs/20261115T082629Z-shared-lora-retention20-v1-89ee2c/result.json),
[diagnostic data/effect auditda3cec](../../runs/20261115T085358Z-shared-lora-retention-audit-v1-da3cec/result.json),
[learning/retention figure](../../artifacts/20261115T082629Z-shared-lora-retention20-v1-89ee2c/retention20-and-loss.png).

## Đã học và đã đo gì?

Một rank4/alpha8 adapter dùng chung,159.744FP32parameters, cùng immutable
Qwen0.8B base ở dtype gốc.20chặng xoay AddAlarm→AddReminder→CompleteReminder→
DeleteAlarm, lặp5lần; mỗi chặng12current exposures+12replay, reservoir64.
480optimizer steps gồm240current/240replay, từ68unique source-verified TRAIN
rows; không phải480trải nghiệm/20kỹ năng độc lập. TEST40không đi vào gradient
hoặc replay. Không tăng model, tạo20adapter product hoặc train vision.

|API|Trước học thêm|Sau20chặng|Final forgetting đã đăng ký|
|---|---:|---:|---:|
|AddAlarm16TEST|16/16|16/16|0pp|
|AddReminder8TEST|3/8|8/8|0pp|
|CompleteReminder8TEST|6/8|8/8|0pp|
|DeleteAlarm8TEST|4/8|8/8|0pp|
|Tổng|29/40 (72,5%)|40/40|Mean0pp, worst API0pp|

Auditor tính lại mọi stage prediction từ raw JSON và targets đã source-verify,
không tin trườngscore do trainer tự gắn.40final outputs được replay qua original
ToolTalk APIs, effect/state khớp. Acquisition1,0 và final3/5pp gate qua **trên
bộ đã đo**; one seed260117, fourknownAPIs/knownTEST/templates, không fresh
unseen-family test, chưa khoảng tin cậy để chứng nhận general population.

**Có quên tạm thời:** sau chặng3, AddReminder từ8/8 xuống7/8 (−12,5pp), mean
drop khi đó3,125pp, rồi phục hồi8/8 ở chặng4. Finalforgetting0không có nghĩa
không từng ghi đè kỹ năng trong stream. Transient statistic báo riêng, không
thay metric final đã đăng ký hoặc che dữ liệu ở giữa.

Bộ12general controls: toán2/4→3/4, JSON4/4→4/4, đọcnguồn4/4→4/4; quan sát
10/12→11/12. Mẫu này quá ít để chứng minh population regression≤3pp/no family
>5pp, và chưa kiểm vision/video/code/hội thoại dài. Không suy100%API thành model
thông minh đa năng hoặc “ngang online model”. Chưa có no-replay/matched extra-
updates control nên không quy riêng thành quả cho replay hoặc một cơ chế mới.

## Lỗi comparator và ranh giới kết luận

Runner hoàn thành20chặng/480updates rồi STOP ở frozen-base-RAG view:
duplicate full metadata/schema trong mỗi example làm24/40prompts vượt512tokens,
max682. Đây là lỗi preflight ngân sách của Codex. [Data-only observation](../../runs/20261115T082629Z-shared-lora-retention20-v1-89ee2c/RAG-token-budget-observation.json)
và [scope note](../../experiments/shared-lora-retention20-audit-scope-note.md)
ghi trước khi comparator chạy. Không nới bound/truncate/đổi labels hoặc verdict.

Original Python script thiếu returnnonzero ở error path, nên wrapper thấy
exitcode0 dù structured resultSTOP/NOT-PASS. Exitcode không được dùng làmPASS;
raw result giữ nguyên. Source tương lai đã sửa returncode, source snapshot cũ
vẫn archive. Original post-training basehash/reload step sau comparator chưa
chạy; không tuyên bố toàn bộ persistence/resource protocol qua từ một metric.

Auditor có `--diagnostic-comparator-stop`, chỉ được đọc complete20stage trajectory
của đúng failure512token đó, không công nhận wholeprotocol. Tất cả TRAIN-only
update/replay IDs, RNG states và20checkpoints được kiểm lại;0newmodel calls.

[Comparator-view correction](../../experiments/shared-lora-retention-comparator-view-repair-v1.md)
đăng ký riêng sau failure, giữ cùng2TRAIN examples và512bound; chỉ extract public
task/error/repair thay vì lặp đầy đủschema. Không training lại,42extra top-level
calls trong156quota còn lại; tổng1386/1500, original1344calls/480updates không
refund. [Corrected run9790fe](../../runs/20261115T085216Z-retention-comparator-view-v1-9790fe/result.json)
max478tokens, adapter artifact fixed-logit reload exact.

**Không claim thắng RAG:** corrected frozen-baseRAG chỉ0/40; independent replay
[ca2aed](../../runs/20261115T085544Z-retention-comparator-audit-v1-ca2aed/result.json)
thấy31outputs chép đúng một TRAIN correction thay vì giá trị/ID của yêu cầu
hiện tại. Examples đứng sau currentrequest là prompt-binding confound ở model
nhỏ; đây không phải strongestRAG đã được xác nhận. Giữ rawoutputs, không tune
prompt trên TEST để đổi baseline. Không dùng0điểm này để gọi LoRA/SNN hơnRAG.
Original wholeprotocol vẫn STOP/NOT-PASS dù measured component có giá trị.

## Tài nguyên, file và restore

Original runner wall1384,54s (~23,1phút),480optimizer attempts,1344top-level
model invocations. Correction81,33s/42calls/0optimizer; tổng1386top-level calls,
decoder invocations bên trong generate và checkpoint recompute không được gộp
thành số SNNsteps. Mọi generate cap96newtokens; không quantization/precision drop.

Runner kiểm host8GB/dedicated reserve512MiB trước updates nhưng không persist
mọi số đo. [Late observer](../../runs/20261115T082629Z-shared-lora-retention20-v1-89ee2c/late-resource-observer.json)
chỉ có132samples trong khoảng cuối: observed RSSsum peak2.228.195.328byte,
privatecommitsum5.213.573.120byte, dedicatedfree min2.862.612.480byte=2730MiB.
Không backfill các chặng trước hoặc gọi đây là peak toànrun/toànsystem; RSSsum
không dedup sharedphysical RAM. Toàn study resource certification chưa đầy đủ.

[Checkpoint data audit](../../runs/20261115T082629Z-shared-lora-retention20-v1-89ee2c/checkpoint-data-audit.json)
load optimizer state bằng `weights_only=True,map_location=cpu`, không initCUDA:
20checkpoints có24LoRA matricesFP32/159.744params, Adam steps24…480,
momentsfinite, CPU/CUDA RNGrecords và replay IDs hợp lệ. Same-next-training-step
resume chưa thực thi; file hợp lệ không tự là numerical-continuation proof.

-Final PEFT candidate: [stage19 weights](../../artifacts/20261115T082629Z-shared-lora-retention20-v1-89ee2c/stage-19/adapter_model.safetensors),
  vẫn một sharedadapter và20versioned research snapshots.
-[Final F32GGUF export](../../artifacts/20261115T082629Z-shared-lora-retention20-v1-89ee2c/shared-adapter-final-f32.gguf)
 24F32tensors, không nénbase. **Weights mới này chưa có native generation
 acceptance**; native qualification trước thuộc initial32step adapter khác.
-[Post-live audit](../../runs/20261115T082629Z-shared-lora-retention20-v1-89ee2c/post-live-state.json):
 Hermes auto/warmGPU/learning restored, exact27savedALIFarrays/RNG/replay,
 liveupdates5. Không cài adapter, đổi source/weights main hoặc R13.

## Chốt nghiên cứu

**Có tiến bộ đo được:** một adapter nhỏ học thêm3API và giữ kỹ năng ban đầu
trên40knowncases qua20learning stages, có transientdrop rồi phục hồi. Cần gọi
đúng là LoRA+replay/VLM learning, không hiddenSNN advantage hay memoryvideo.

**Còn thiếu:** reliable strongest same-budget RAG comparator,3seeds/fresh wider
tasks, confidence cho3/5pp old/general regression, no-replay/control attribution,
full telemetry/persistence qualification và latest native/product integration.
Không tạo thêm datasets/epochs chỉ để giữ100% trên bộ này. SNN family6actions
đã STOP sau2DEVrounds, chưa mới readout-only/hidden learning/retention result.

Giữ toàn thresholds và failures. Overall **PIVOT** cho nghiên cứu SNN utility;
**candidate evidence only** cho sharedLoRA, **chưa production/lifelongPASS**.
