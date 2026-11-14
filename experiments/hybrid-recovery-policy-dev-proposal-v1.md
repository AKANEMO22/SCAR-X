# Recovery policy DEV: kế hoạch nối từ trải nghiệm tới hành động

Trạng thái: **đề xuất, chưa chạy training hoặc chọn policy**. P0wrapperac7bb8 chỉ
cho proposal từ weights chưa học. Không coi neuron hoạt động/đổi input/khôi phục
đúng là bằng chứng nó đã học representation hay tự sửa lỗi.

## Giả thuyết và nguyên nhân cần tách

H1. Cung cấp observation lỗi và kết quả probe theo thời gian giúp policy học
chọn recovery ở tình huống mới, vượt cùng frozen VLM + strongest RAG/rule.
H2. Hidden local updates có lợi ích hơn readout-only, mất lợi ích khi hidden
reset/shuffle; không lấy tăng số neuron, thêm nguồn hoặc prompt đặc biệt làm lợi thế.
H3. Improvement tồn tại sau cold restart, retrieval off cho phần synaptic policy,
đồng thời vẫn học mới và giữ forgetting mean≤3pp/worst≤5pp sau20chặng.
H4. Independent outcome verification phát hiện được API thành công nhưng sai
task và finaltext success không có effect; lời self-critique không làm nhãn đúng.

Hiện chưa có bằng chứng bảo vệ synapse trong Hybrid: importance/stable inert.
Vòng DEV đầu giữ nguyên learner equation để tách lợi ích encoder/actionbridge
khỏi thay công thức. Cơ chế importance/anchor mới cần proposal/ablation riêng,
không thêm lén trong cùng arm rồi quy toàn bộ gain cho representation.

## Data và thực thi cần chuẩn bị trước lượt học

- Corpus ngoài: ToolTalk pinned e05f4ce6, schema patches explicit, không sửa
  snapshot/groundtruth. Quarantine37raw response differences để rà theo state,
  order/ID/semantics; không tự gán tất cả là lỗi nhãn hoặc bỏ chỉ vì model sai.
- Phần fault bổ sung phải thực thi trong owned sandbox: missing/ill-typed params,
  stale-but-valid evidence, auth absent, contradictory event updates, legitimate
  repeated query và insufficient task info. Không lấyfamilyID làm target.
- Mỗi recovery action được chạy trên snapshotclone độc lập trước khi chọn nhãn.
  Tool effect và final response được chấm theo source-owned executable contract
  hoặc annotation độc lập có provenance. Outcome tables/gold không vào encoder.
- Show_schema/inspect_status/retry/clarify/defer/finish là sáu researchslot riêng;
  argument repair vẫn do cùng frozenVLM sử dụng bằng chứng, không hardcode đáp án
  cho một arm. Không có generic semantic truth oracle.
- Task-disjoint split theo conversation/template và nguồn; có cases thay thứ tự
  sự kiện nhưng giữ bag-of-values để buộc temporal discrimination. Các task cần
  representation không được chỉ là iferror→ask có ceiling100%.
- Khóa manifests/source/SHA/templatepartition trước model inference; finalTEST
  riêng chưa được xem nhãn khi phát triển. P0bafilecase và4native smoke không
  được tái dùng làm TEST. Không upsample một ca thành nhiều independent samples.

## Điều kiện và ngân sách so sánh

1. FrozenVLM + cùng nguồn/tools + strong fixed recovery/rule guard.
2. FrozenVLM + bounded verified episodic/RAG experience, không neural update.
3. Untrained frozen ALIF policy với encoder mới.
4. Readout-only policy cùng encoder, steps và replay.
5. Hidden-learning policy cùng encoder/local credit.
6. Shuffled-learning diagnostic và hidden-reset trên clone để kiểm causal role.

Cùng VLM native0.8B/16K, precision, tool schemas, allowedactions, sourceaccess,
prompt, nativecall4/task, output128/call, two recovery macros, replay64, elapsed
learning budget và feedback observations. Không cho neural arm thêm retrievable
records hoặc verifier mà rule/RAG không có. Báo rawproposal và executedaction
riêng nếu safetyguard override; không gán override thành learned intelligence.

Dự toán vòng DEV:3seeds260113/260137/260171,20stages×12verifiedexperiences,
≤40held-out probes/stage/final cho khả năng tách family; actualsharednative≤1200
uniqueinputs/≤60phút. Đây là dự toán cần khóa exactcasecount sau dataqualification;
chưa phải lệnh khởi chạy. Không thêm GPU memory pressure khi8htest đang dùng
frozenworker nếu không có resourcepreflight; CPU I/O/checkpoint, neural CUDA.
Không training/LoRA củaVLM, không import researchweights vàoHermes.

Primary: verified gap-resolution + source-supported taskcompletion, không chỉ
API exception=None. Đồng thời báo unsupported-success rate, inventedarguments,
unresolvedrepeat/callcost, safeclarification, refusals và time/peakVRAM/hostRAM.
Không biến keywordlogin/questionmark thành đủ bằng chứng đáp án đúng.

Giữ gate≥5pp over strongestcontrol và lower95%CI>0, seed-/taskcluster-aware;
hidden phải thắngreadout và sensitiveablation. Báo acquisition từng family và
oldskill20stage3/5pp; nolearning gầnchance không được tính retention thắng.
Confidence intervals không được lấy từng token/update làm independent replicate.

## Điểm dừng

- Loader/trace/source/verification/resource failure: dừng arm và giữ receipt.
- Nếu strongrule/RAG đạt ceiling trên DEV: chưa chứng minh neuralbenefit; đăng ký
  taskbank khó hơn riêng, không xóa baseline hay nới gate.
- Không có lợi ích sau tối đa hai registeredDEVrounds: PIVOT representation/
  credit/readoutstability; không tự scaleSNN/VLM hoặc đổi finalTEST.
- Chỉ cân nhắc deployment khi data-validity + acquisition + retention + causal
  hiddenadvantage và runtime gates đều có evidence;8hengineering riêng không đủ.

Kết quả cần bàn giao: full source+datamanifests, input/action/outcome/rawVLMtables,
3seed curves/ablation/statisticalaudit, fullstate coldcontinuation/rollback,
RAM/VRAM/time, actuallocalhosttrace và quyết định GO/PIVOT/NO-GO đúng phạm vi.
