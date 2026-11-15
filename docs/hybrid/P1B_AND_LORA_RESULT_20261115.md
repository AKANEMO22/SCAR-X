# P1B và LoRA — học được một sửa lỗi hẹp ở adapter, chưa có lợi thế SNN

**Native follow-up:** [qualification cùng llama.cpp worker](LORA_NATIVE_PORTABILITY_20261115.md)
đã có, không còn chỉ là GGUF conversion. Các dòng “chưa native acceptance” bên
dưới mô tả thời điểm pilot; current scope vẫn chưa population retention hoặc
liveHermes promotion. Không sửa receipt/history pilot ban đầu.

User duyệt P1B/actual errors/local ALIF GPU/retention, sau đó cho phép train và
tạo LoRA. Hai phần được tách attribution: SNN selector dùng frozen VLM trong
llama.cpp; adapter pilot dùng Transformers/PEFT. Không dùng điểm hai backend
để suy rằng SNN hoặc LoRA thắng RAG trong cùng điều kiện.

**Quyết định:** STOP training selector trên action bank hiện tại vì không thể
qua gate đã đăng ký. LoRA tạo được candidate cải thiện task hẹp trên pilot,
chưa đủ điều kiện production, retention hoặc cài tự động vào Hermes.

## Nguồn, dữ liệu và điều kiện trước học

[Protocol P1B](../../experiments/hybrid-p1b-error-learning-v1.md), [LoRA pilot](../../experiments/hybrid-lora-error-repair-pilot-v1.md)
khóa trước training. Đã đọc paper/code tác giả Reflexion/ExpeL, ToolSandbox
evaluator pin và audit ToolTalk cũ, LoRA paper2106.09685 + PEFT0.19.1/source
và Transformers Qwen3.5 source. Không tái lập toàn benchmark hoặc chạy lại
old132 cells/history FAIL/old oracle-margin4,17pp để tìm điểm tốt hơn.

[Data019e7c](../../runs/20261115T073358Z-verified-repair-data-v1-019e7c/result.json):
56 owned public requests,32TRAIN/8DEV/16TEST,112 ToolTalk calls thật gồm invalid
request và corrected request. Lỗi missing/unknown argument, invalid format/value;
mọi teacher correction được executor chấp nhận và exact alarm/state checker xác
nhận. Teacher không phải output model hoặc dữ kiện của người dùng thật.

Audit trước inference phát hiện template index đi cùng error kind ở view đầu.
[Qualified view2d102f](../../runs/20261115T073642Z-repair-data-qualified-v2-2d102f/result.json)
cân bằng từng template×error kind, không gọi lại tool/model hoặc sửa raw results.
Wording và requested times tách split; bốn loại lỗi/API vẫn chia sẻ giữa split,
không phải unseen-tool/family benchmark. Target TEST không vào optimizer.

## P1B: DEV qua usefulness, nhưng bank đầy đủ không thể đạt gain

392 actual native Qwen0.8B inputs đo6 recovery actions và RAG experience arm
qua same source/executor. Mỗi native reply được thực thi trong sandbox độc lập.
Rule khóa trước output: error→show_schema, otherwise finish. RAG dùng2 verified
TRAIN examples truy xuất lexical; không dùng DEV/TEST answer để tìm nguồn.
Đây chưa là maximum-strength production RAG hoặc full semantic retriever.

|Split|Rule show_schema|RAG verified experience|Best measured6-action oracle|Margin oracle−strongest|
|---|---:|---:|---:|---:|
|TRAIN32|27/32|30/32|29/32|−3,125pp|
|DEV8|5/8|6/8|7/8|+12,5pp|
|TEST16|9/16|10/16|10/16|0pp|

[DEVce3cae](../../runs/20261115T073811Z-p1b-action-usefulness-v1-ce3cae/result.json)
đủ mở outcome expansion; [TRAIN/TEST7fa85d](../../runs/20261115T074020Z-p1b-train-test-outcomes-v1-7fa85d/result.json)
cho thấy fixed action space không thể vượt control thêm5pp. Inherited verdict
field trong expansion là diagnostic, không thay quyết định bằng TEST để chọn
một winner. [Stop receipta5fac4](../../runs/20261115T074453Z-p1b-local-error-learning-v1-a5fac4/result.json)
giữ gate,0 ALIF updates/forwards. Không bỏ RAG baseline hoặc đổi macro/prompt
sau xem TEST để hợp thức hóa training. Không chạy readout-only/hidden/shuffle
hay20stage retention trên bank không thể đạt mục tiêu.

**Vì vậy chưa có kết quả retention P1B mới.** Không lấy không học/không quên làm
retention PASS. Historical mean forgetting8,17pp/worst20pp và protected5/12pp
giữ nguyên; gate mean≤3/no family>5pp, acquisition65%, gain5pp/lower95%CI>0 không nới.

[Independent action-bank replaya80718](../../runs/20261115T074831Z-p1b-action-bank-data-audit-a80718/result.json)
thực616 owned tool calls để kiểm lại392 saved outputs bằng implementation effect
checker khác;0 new model/learning calls,0 mismatch. Đây là xác minh dữ liệu đã đo,
không392task chất lượng độc lập hoặc một inference confirmation mới.

RAG là arm riêng ngoài6 slots; kể cả cộng nó vào action oracle, TEST vẫn10/16
vì union những action measured không giải thêm ca. Kết quả là **NO-GO cho bank/
action contract này**, không phải định lý SNN không bao giờ hơn RAG. Bước sau
phải có task/action utility phân biệt được, cùng quyền truy xuất và hypothesis
mới đã đăng ký; không thêm neuron/epoch trên bank này.

## LoRA: candidate sửa tham số API có cải thiện thật trong pilot

Lượt loader552928 dừng trước forward/update vì Transformers không khai báo
mtp.fc.weight. Raw failure giữ nguyên. File ở path unqualified-partial-adapter
của lượt đó là **full untrained base snapshot**, không một LoRA đã học; có
artifact-classification receipt, giữ local và không đưa vào Git/release.

Loader tương thích tiếp theo giữ **đủ488 source tensors** ở dtype gốc (452BF16,
36FP32);15MTP tensors không dùng trong standard forward được đăng ký frozen,
không bị drop/prune. LoRA có159.744 trainable FP32 parameters, rank4/alpha8,
q_proj/v_proj của text attention, không train vision hoặc base/ALIF.

[Actual pilot5c09fa](../../runs/20261115T074627Z-native-lora-repair-v1-5c09fa/result.json):
32optimizer steps/32 unique TRAIN rows, lr1e-4, batch1, one pass;79,175s.
Completion-only causal loss0,08858→0,003638.32steps/loss là training evidence,
không tự chứng minh generalization. Greedy20probes trước/sau gồm4DEV+16TEST.

|Transformers native-precision backend|TEST exact repair|Independent effect replay|
|---|---:|---:|
|Base/zero-trained adapter|6/16 (37,5%)|6/16 đúng alarm/time/unchanged other state|
|After32steps|16/16 (100%)|16/16 đúng effect, không chỉ JSON string|

[Audit2c1c46](../../runs/20261115T075140Z-lora-repair-audit-v1-2c1c46/result.json)
replays40 before/after outputs và xác nhận optimizer chỉ thấy TRAIN. Exploratory
paired gain62,5pp, error-family bootstrap95%[25;100]pp trên4 clusters, **một seed,
16câu từ2 wording templates**. Khoảng này rất yếu về external generalization;
chưa3seed/20stage/old-general controls. Audit đầu1135a8 và audit bổ sung cùng40
outputs không tính như80 mẫu chất lượng độc lập.

24 adapter matrices đềuFP32; tất cả12 B matrices nonzero sau update. Source
base tensors được hash trước/sau trong worker, exact; saved/reloaded adapter
trên same base cho fixed logits byte-exact. Worker kept dedicated-GPU reserve
ở các sampled checks; không giảm precision hoặc dùng quantization.

Raw report `forward_calls=74` là **74 top-level runner invocations** gồm40
generate probes,32training forwards và2reload probes. Mỗi generate có tối đa64
new tokens nên nhiều decoder forwards nội bộ; không gọi74 là số physical forward
kernel thực tế hoặc lấy nó làm tương đương ngân sách SNN. Tối đa40×64+34=2594
model decoder invocations theo configured generation bound; không log token-level
hook trong pilot. Không dùng thiếu sót telemetry này để claim production PASS.

## Artifact và vận hành

-[PEFT weights](../../artifacts/20261115T074627Z-native-lora-repair-v1-5c09fa/adapter/adapter_model.safetensors):
  642.472byte, SHA f1286d4fa6d6eb967a1599fe6540a5f424f3a320a561b3b1aaa7d4b9c501898c.
-[GGUF FP32 candidate](../../artifacts/20261115T074627Z-native-lora-repair-v1-5c09fa/adapter-f32.gguf):
  641.280byte,24 F32 tensors; element counts/Frobenius norms khớp source sau
  converter layout permutations. Không quantization; metadata quant-version
  không có nghĩa tensor đã lượng tử hóa. **Chưa native inference acceptance**.
-[Candidate manifest](../../artifacts/20261115T074627Z-native-lora-repair-v1-5c09fa/CANDIDATE_MANIFEST.json)
  pin base revision/adapter/config/file hashes, không chứa model base.
-Hermes được sleep trong owned idle window để nhường GPU, learning nền tạm tắt;
  sau worker exit đã restore auto/learning và warm-resident native GPU. Không
  kill process ngoại, không checkpoint rewind. [Live-state audit](../release/evidence/P1B_LORA_LIVE_STATE_20261115.json)
  giữ exact27 saved arrays/RNG/replay, updates5. Adapter **chưa cài vào Hermes**.
-Localhost có56 source/error/proof records,392 native action answers,32 actual
  optimizer losses và40 before/after LoRA answers, tách origin rõ. Không mock.

**Chốt:** có adapter candidate học được quy tắc output/repair hẹp trên bộ đo này.
Chưa biết nó giữ được kỹ năng sau học liên tục hay ảnh hưởng kiến thức/công cụ
khác. SNN contribution và retention vẫn chưa được chứng minh. Không merge vào
main/live, không tự kích hoạt LoRA, không tuyên bố project đạt production.
