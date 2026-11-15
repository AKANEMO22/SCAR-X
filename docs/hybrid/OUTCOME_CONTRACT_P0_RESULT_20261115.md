# Outcome contract P0 — kiểm đúng tác vụ trước khi nghiên cứu credit

**GO cho checker trong lab; chưa triển khai vào Hermes hoặc huấn luyện SNN.**
Người dùng duyệt bằng “làm đi” ngày15/11/2026. Đây là12 fixture kỹ thuật với14
trace thực thi, không phải accuracy100% của model. Lõi ALIF, encoder, trọng số,
precision và protocol/FAIL trước giữ nguyên. Mục tiêu khoa học vẫn **PIVOT**.

## Đã chạy và kết quả

[Protocol khóa trước chạy](../../experiments/hybrid-stateful-outcome-contract-p0-v1.md),
[manifest/source](../../runs/20261115T044756Z-outcome-contract-p0-v1-5842c6/manifest.json),
[raw result](../../runs/20261115T044756Z-outcome-contract-p0-v1-5842c6/result.json),
[data/state rescore](../../runs/20261115T044756Z-outcome-contract-p0-v1-5842c6/independent-data-audit.json).
Source commit `6b872fdf` được khóa trước42 thao tác thực trên file/SQLite riêng.
Không gọi Hermes tools/API bên ngoài hoặc sinh câu trả lời bằng model. Các final
text/status thử nghiệm do fixture đặt, có origin rõ, không giả thành model output.

| Fixture | Bằng chứng thực thi | Checker kết luận |
|---|---|---|
|01: đúng nguồn|Đọc requested.txt và so nội dung/hash|Hoàn thành task đọc|
|02: nguồn khác, cùng nội dung|Đọc other.txt có cùng bytes|Không chứng minh task gốc|
|03: API success không effect|Acknowledgement không thay trạng thái yêu cầu|Unverified|
|04: effect thiếu|Đổi value nhưng status còn pending|Unverified, không đủ yêu cầu|
|05: đúng record/effect|SQLite target→new/committed, record khác giữ nguyên|Hoàn thành|
|06: nhầm record|SQLite other đổi, final fixture vẫn báo completed|Không hoàn thành|
|07: đúng/đảo thứ tự|Hai trace có final state và bag actions giống nhau|Chỉ prepare→commit thỏa dependency|
|08: bị chặn rồi mới làm đúng|Registry từ chối forbidden attempt, sau đó update đúng|Effect đúng nhưng whole-task không qua|
|09:24 read lỗi rồi read đúng|24 FileNotFoundError thật, tiếp theo đọc đúng nguồn|Ghi loop24/call dư23; effect cuối đúng, terminal không stuck nhưng whole-task không qua|
|10: poll có tiến độ|SQLite progress1→2→3/completed|Không phạt như loop; task hoàn thành|
|11: chỉ tự báo thành công|Final fixture text có, không event/effect|Unverified|
|12: hỏi sai/đúng field|Queue SQLite hỏi login so với alarm_time đang NULL|Chỉ đúng field là clarification_supported; chưa là task completion|

14/14 trace khớp assertions đã khóa của12 fixture. Không lấy12/14 làm số mẫu
AI độc lập, không CI/seed sweep vì không huấn luyện hoặc đo generalization.
Không ghi các outcome thành nhãn synapse: `learning_authorized=false`,
`neural_credit_eligible=false`, selector toàn bộ là `fixed_fixture`.

## Tài nguyên, persistence và trace

-Runner elapsed **1,078s**, RSS đỉnh tiến trình **32.407.552 byte** (~32,4MB),
  dưới budget300s/128MiB. Đây là process P0, không phải peak toàn Hermes/hệ AI.
-Raw artifacts khoảng0,59MB ở lúc audit, dưới10MB. Giữ real SQLite DB, file,
  contract, snapshots, decision cutoffs, tool JSON, observed prefixes và outcome.
-0 SNN forward/learning,0 native VLM/ASR calls; không import NumPy/CuPy/Torch/
  scarx trong worker. Files/SQLite chạy CPU; mode serving không chuyển CPU.
-[Live saved state](../../runs/20261115T044756Z-outcome-contract-p0-v1-5842c6/live-state-audit.json)
  giữ exact hashes của27 mảng và learned metadata gồm RNG/replay, updates5.
  Controller time/journal được tách khỏi neural state; không restore DB để làm
  audit qua. Technical trace chỉ append sự kiện orchestrator, không là experience
  hoặc nhãn của model.
-[Post-runtime audit](../../runs/20261115T044756Z-outcome-contract-p0-v1-5842c6/post-runtime-audit.json)
  xác nhận58 payload RC14 còn khớp, gateway/native GPU đúng ownership/device,
  ready và0 failure. Không main/R13/source serving change, merge hoặc push.
-Trước chạy có12 pure preflight regressions qua: sai scope/task, clock NaN,
  future/extra observation, result/snapshot tamper, neural attribution giả,
  nguồn sai với verified:true, empty trajectory và side effects. Không model calls.
-Localhost nhận start,14 kết quả trace và completion thật tại
  `http://127.0.0.1:8775/?tab=trace`, actor orchestrator/work_trace.activity.
  Người dùng không cần chọn nguồn/bấm chạy để thấy hoạt động.

## Kiểm toán và giới hạn

Auditor riêng **không import checker**, tính lại source/state/effect/order/cost
từ raw snapshots, so final file/SQLite hiện có và kiểm cutoff/observed prefix.
14 rescore khớp; hai thứ tự case07 có cùng final state/bag actions được xác nhận.
Đây là implementation kiểm toán khác của cùng orchestrator, chưa là external
peer approval. Hash nhất quán và fixture pass không tự chứng minh producer ngoài
lab đáng tin hoặc có verifier sự thật ngữ nghĩa tổng quát.

Đã kế thừa nguyên tắc state/milestone từ [ToolSandbox](https://github.com/apple-aiml-research/ToolSandbox/blob/main/tool_sandbox/common/evaluation.py),
không copy scorer, import framework hoặc dùng soft text similarity làm nhãn thật.
Audit ToolTalk cũ được tái dùng, không chạy lại benchmark đã biết. Manifest/source
pin và [intake](../research/literature/sources/toolsandbox-source-check-20261115.json)
giữ ranh giới phần thực đọc và phần chưa kiểm.

Antigravity `gemini-3.8-flash-high`, effort high đã nhận inline static review
trước chạy. [Receipt](../../runs/20261115T044756Z-outcome-contract-p0-v1-5842c6/antigravity-review-receipt.json)
ghi CLI print_timeout120s/0 verdict. CLI đã terminal, upstream turn báo còn đang
tiến hành; không suy là review hoàn tất, không resubmit hoặc gọi đó là approval.

## Bước tiếp theo, chưa được mở tự động

P0 này chỉ làm rõ outcome đúng phạm vi. Còn thiếu đường nối: observation lỗi/
probe → neural input phân biệt được → neural action thực thi → outcome gắn đúng
decision → local update → kỹ năng unseen và retention. Audit hiện tại vẫn cho
3 probe outcomes thành1 neural input; shadow và live có action semantics khác.

Không train trên12 fixture này hoặc bank recovery cũ có oracle margin4,17pp
dưới gate5pp. Chưa thay encoder, credit transport, learner, P1/DEV hoặc model live.
P1 sau phải khóa causal binding, action/verifier usefulness và matched strong
RAG/experience/frozen/readout/hidden controls trước đo; giữ acquisition,
gain≥5pp/lower95%CI>0 và20stage forgetting mean≤3/worst≤5pp.

Tái lập **audit không tạo tool/model calls**:

```powershell
& 'local/hybrid-v1/runtime-env/Scripts/python.exe' -I -B `
  scripts/audit_hybrid_outcome_contract_p0.py `
  runs/20261115T044756Z-outcome-contract-p0-v1-5842c6
```

Runner từ chối chạy lại khi execution marker đã có. Không reset marker hoặc
tạo run mới từ chính phê duyệt một lượt này. Raw receipt cũ luôn giữ nguyên.
