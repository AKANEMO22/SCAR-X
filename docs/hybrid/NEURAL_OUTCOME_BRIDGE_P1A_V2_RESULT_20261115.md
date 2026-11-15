# P1A-v2 — qua kiểm chứng kỹ thuật frozen neural → action → outcome

**GO cho bridge tronglab. Chưa GO cho learning, hidden advantage hoặc production.**
User duyệt riêng “Duyệt confirmation v2” ngày15/11/2026. Source trước execution:
`50d7cd6c665f42bc93bce857834e21238a335634`. Không thay ALIF H128, FP32, topology,
adapter/action meanings hoặc per-seed loop; không import prototype vào Hermes.

[Protocol đã khóa](../../experiments/hybrid-neural-bridge-guard-confirmation-v2.md),
[manifest](../../runs/20261115T063826Z-neural-outcome-bridge-p1a-v2-6e203c/manifest.json),
[result](../../runs/20261115T063826Z-neural-outcome-bridge-p1a-v2-6e203c/result.json),
[independent data audit](../../runs/20261115T063826Z-neural-outcome-bridge-p1a-v2-6e203c/independent-data-audit.json),
[receipt tổng hợp](../release/evidence/NEURAL_OUTCOME_BRIDGE_P1A_V2_20261115.json).

## Kết quả thực thi

| Hạng mục | Kết quả đúng phạm vi |
|---|---|
|8 technical cases /2 seed|260113 và260137; all interface assertions qua, không là accuracy model|
|Neural compute|18 actual CUDA0 forwards của frozen ALIF H128,27 arrays ở GPU|
|Dispatch|10 attempts:8 actual local macro executions và2 scope guard denials|
|Learning/model services|0 learn/synapse updates,0 native VLM/ASR calls; không backward/BPTT|
|Timing|27,997s runner wall, dưới600s; có88 metadata queries, không là inference-speed benchmark|
|Worker memory|Peak RSS726.282.240 byte; private commit750.735.360 byte, dưới1GB. Không peak toàn AI/Hermes|
|GPU headroom|Dedicated query min2.872.049.664 byte =2739MiB (~2,87GB), trên512MiB ở các điểm đo|
|Guard|0 unexpected violations;1 local hostname query,3 pinned VER probes được decline không spawn shell|
|Telemetry|88 pinned nvidia-smi dedicated-memory queries; CUDA runtime availability ghi riêng|
|Artifacts|Khoảng11,35MB gồm build cache tại audit, dưới20MB. Model NPZ/bridge ZIP_STORED; không nén/lượng tử hóa weights|

[E0 receipt](../../runs/20261115T063826Z-neural-outcome-bridge-p1a-v2-6e203c/environment-preflight.json)
xác nhận SDK/telemetry gate qua trước khi khởi tạo ALIF,0 forward/dispatch.
35 pure preflight checks qua trước run: malformed input/receipt, ledger, snapshot
identity, SDK guard, Windows audit argv form, v1-loop AST identity và namespace.
Những checks này không khởi tạo model, gọi tool hoặc đo chất lượng AI.

## SNN đã tác động đến cái gì?

Public task/request được đưa vào block riêng; causal observations nằm ở các
block riêng, không bị câu hỏi dài chiếm ngân sách. Frozen probabilities chọn
action qua argmax trong2 capabilities được cấp quyền. Decision được ghi trước
dispatch; receipt gắn chính xác decision/task/scope/source/macro, không suy từ
timestamp gần nhau. Actor neural và scope guard được ghi riêng.

|Ordinary case|Seed260113|Seed260137|Điều thực sự xảy ra|
|---|---|---|---|
|Nguồn có thật|inspect_status|inspect_status|Đọc đúng bound file; checker có source/effect proof|
|Nguồn thiếu|inspect_status|retry|Cả hai nhận FileNotFoundError thật; không lấy file khác làm proof|
|Job progress|inspect_status|inspect_status|Đọc progress2; chưa completed, không tự tạo success|
|Field còn thiếu|show_schema|clarify|Đọc schema /queue đúng alarm_time; chưa task completion|

File capabilities trong case1 đều có thể đọc nguồn: task proof không chứng
minh policy chọn tối ưu. Policy chưa học nên không gọi khác biệt giữa seed là
representation mới hoặc học kỹ năng. Actual macro I/O dùng project/SQLite riêng;
đây chưa là chấm mọi tool workflow của Hermes hoặc dữ liệu riêng người dùng.

Case5 reject foreign/altered receipt và receipt hoàn toàn hợp lệ của decision
khác. Case6 private/future input reject trước forward, không tiêu thêm ledger.
Case7 revoke lease sau proposal:2 denials không bị gán credit neural. Case8
pending bindings/model arrays/RNG/counters save/load/cold/rollback exact; journal
không rewind, scope đã revoke không được cấp lại. Old/live loaders từ chối wrapper.

## Kiểm toán và bảo toàn lịch sử

Auditor không import bridge/model và không gọi forward/tool. Nó re-encode input,
tính lại masked readout từ saved features/weights, so argmax với actual decision,
rồi kiểm SQLite journal, event/source hashes, actual retained file/state, ledger,
continuation và guard/resource receipts. Probability error tối đa4,4512×10⁻⁸,
dưới3×10⁻⁶ đã đăng ký. Auditor SHA khớp bản đã pin trước execution. Đây là code
kiểm độc lập của cùng orchestrator, không phải external peer approval hoặc
chứng nhận trajectory CPU/CUDA; history P0 FAIL vẫn nguyên.

[V1 result](../../runs/20261115T054242Z-neural-outcome-bridge-p1a-v1-f39bc6/result.json)
vẫn STOPPED_NOT_PASS. Hash result/ledger cũ khớp;18 forward/10 dispatch cũ không
refund/reset. V2 có marker/ledger/quota riêng. Tổng hai lượt thực tế36 forwards,
20 dispatches,0 learning; không giấu chi phí của lượt fail.

[Live-state audit](../../runs/20261115T063826Z-neural-outcome-bridge-p1a-v2-6e203c/live-state-audit.json)
giữ exact hashes27 saved arrays và learned metadata/RNG/replay của model live.
[Post-runtime audit](../../runs/20261115T063826Z-neural-outcome-bridge-p1a-v2-6e203c/post-runtime-audit.json)
xác nhận58 RC14 payload còn khớp, gateway/nativeGPU đúng identity, ready/0failure,
live updates5. Main/R13, weights, config và Hermes không sửa hoặc restart.

31 actual technical/proposal/outcome events xuất hiện tự động trên
`http://127.0.0.1:8775/?tab=trace`. Neural proposals/probabilities và raw tool
results đều là số liệu thật; không giả natural-language thoughts/model answers.

[Saved-neuron summary](../../runs/20261115T063826Z-neural-outcome-bridge-p1a-v2-6e203c/saved-neuron-state-summary.json)
đọc lại các snapshot đã có,0 forward: ordinary cases có6–47 neuron phát xung
ở tick cuối trong128 neuron đã cấp. Đây là spike activity tại một thời điểm,
không là tự tăng dung lượng neuron hoặc mức độ hiểu task; không diễn giải thành
“model nghĩ gì”. Summary cũng được append vào technical trace để quan sát.

## Quyết định và giới hạn còn lại

**GO cho frozen bridge tronglab**: SNN output có đường tới macro thật và outcome
có nguồn; guard và persistence đã được kiểm trong budget mới. Không cần rerun
P0/P1A chỉ để xác nhận lại cùng đường dữ liệu khi source không đổi.

**Chưa chứng minh SNN hữu ích hơn rule/RAG, học từ lỗi hoặc giữ kỹ năng lâu dài.**
Chưa có học trong phase này; freeze-zero-forgetting không là retention success.
VLM vẫn nguyên weights và các hạn chế ngôn ngữ/video hiện tại không được giải
quyết bởi bridge. Không train trên8 technical cases hoặc bank oracle-margin4,17pp.

P1B cần public tasks/causal outcomes hợp lệ, counterfactual action usefulness và
strong matched RAG/experience/frozen/readout/hidden controls trước local learning;
giữ acquisition/gain≥5pp/lower95%CI>0 và20stage forgetting mean≤3/worst≤5pp.
**P1B/data acquisition/train hoặc live promotion chưa được phê duyệt.** Không
scale model, thêm LoRA hoặc nới numerical/history gates từ technical PASS này.

Tái lập audit không thêm forward/tool:

```powershell
& 'local/hybrid-v1/runtime-env/Scripts/python.exe' -I -B `
  scripts/audit_hybrid_neural_outcome_bridge_p1a.py `
  runs/20261115T063826Z-neural-outcome-bridge-p1a-v2-6e203c
```

Không chạy lại runner: marker/attempt ledger v2 đã tiêu lượt được duyệt. Compiled
CuPy cache giữ local; raw model/measurement/journal/source cần audit đã lưu Git.
