# Gemini R8 — review cập nhật và sửa tầng kiểm toán

**12/11/2026.** Pin Gemini `fe472b3b1257cdb264805bf79fd53e5647e3568d`, nhánh
`research/gemini/r8-causal-learning-retention-20261112`. Đây là cập nhật thực sự:
22file mới gồm MTSG, BF-Cascade, generators, tests và artifact. Không áp nguyên
nhận xét “chỉ có survey” của commit631c890 cho R8.

**Quyết định: giữ candidate để nghiên cứu; `REQUEST_CHANGES`, chưa promote learner.**
Codex đã chạy13test kỹ thuật gốc,39test audit/regression và các counterexample
không gọi learn. Không tái chạy campaign DEV/confirmation, không train dataset
ngôn ngữ, không sửa source tác giả/main/live hoặc core ALIF. Số liệu quality bên
dưới là artifact được đọc/tính lại, chưa phải training được Codex tái lập.

Sau audit, đã sửa persistence bằng adapter độc lập và qua76test liên quan:
xem `R8_SNAPSHOT_REPAIR_HANDOFF_V1.md`. Các tiny local-update fixtures của lượt
engineering này không phải quality training hoặc phát hiện kỹ năng mới.

## 1. Gemini đã sửa/tiến thêm gì

- Có code dùng gamma[.25,.92] thật; khoảng trống **không có implementation** của
  đề xuất trước đã được khép một phần. Vẫn thiếu raw per-seed/prediction để kiểm
  chứng toàn bộ quality claims.
- Gate không dùng `(3<=t<=8)` nữa. Nó quan sát `sum(abs(row[:6]))>.8`, chỉ giảm
  recurrent eligibility xuống.15 khi không đạt; input eligibility vẫn học.
  Đây là thay đổi cơ chế so với card/protocol amplitude gate chung. Hợp lý để
  nghiên cứu, chưa là cảm biến độ liên quan tổng quát: nó biết trước phân kênh
  task0..5/distractor6/end7.
- BF có fast/slow cho cả input, recurrent **và readout weights**, dùng gap cũ cho
  hai phép diffusion. Hai điểm lo ngại ở card trước được giải quyết ở source này.
- Có unconsolidated vs cascade, R-matrix DEV, zero/frozen/readout controls trong
  DEV source và báo cáo FAIL distractor/P2. Không nhận xét mọi kết quả đều vô ích.
- Các module candidate không dùng Transformer, pretrained model hay autograd;
  không có bằng chứng cần thay bản chất ALIF để tận dụng các ý tưởng này.

## 2. P1 có tín hiệu tốt nhưng chưa chứng minh riêng multiscale là nguyên nhân

`confirmation.json` báo cáo:

| Task | Direct | Refit probe | Lower95 gain vs frozen |
|---|---:|---:|---:|
| strict_binding |79,96%|84,18%|+12,08đ|
| distractor_binding |62,73%|66,33%|+10,98đ|
| xor |96,64%|98,63%|+2,91đ|
| order |100,00%|99,92%|+1,14đ|

Giữ các con số như **reported summaries**. `per_seed_records=[]`; DEV `trials=[]`.
Không thể tái tính paired CI hoặc kiểm missing-seed từ các file này. Trong nguồn
confirmation, frozen head không được train nên report50% frozen-direct không có
phép đo tương ứng; frozen **probe** vẫn là một đối chứng feature hợp lệ về thiết
kế. Không được đánh đồng hai chuyện.

Theo master7.2, lower95 representation gain phải>=5đ trên ít nhất2/3bài hợp lệ,
và direct>=65% cùng required contrasts. Trong bảng chỉ strict_binding đồng thời
qua hai điều kiện này. Distractor faildirect; XOR/order lowerCI<5đ. R8 tự dùng
mean gain>=5đ, lowerCI>0 và ngoại lệ ceiling: đó là tiêu chí khác master, không
được coi là đã đạt master.

### Finding P1: shuffled signal đổi luôn trace/gating

`r8_mtsg_eprop.py::step` chỉ dùng heterogeneous decay và recurrent gate khi
`condition=='learned_mtsg'`. Với `shuffled_signal`, nó dùng `.25*e_f+.75*z_e`,
không gate. Trong `learn_episode`, R35 còn dùng hiddenLR/clip khác(.002/.002 so
với .008/.010). Do đó các đối chứng hiện không tách riêng tín hiệu routing,
multiscale, gating và update magnitude.

Fixture cùng initial weights/input, chưa gọi learn: forward voltage hai nhánh
khớp nhưng eligibility lệch max**0,3453747**. Phải có shuffled/zero cùng chính
trace/gating của candidate; thêm multiscale-only và gate-only trước khi kết luận
“heterogeneity đã giải quyết causal credit”.

Full revert hiện reset cả w_in/w_rec rồi **refit probe mới**. Drop vì vậy trùng
learned-probe−initial/frozen-probe; đây là bằng chứng representation hợp lý nhưng
không phải một bằng chứng độc lập về deployed learned readout. Separate w_in/
w_rec interventions có trong DEV summary, chưa có per-seed/predictions để audit.

## 3. Đã tái hiện và chặn các lỗi gate trong evaluator độc lập

Evaluator tác giả vẫn được giữ nguyên. Counterexamples thật:

1. `R8GateEvaluator({}).evaluate_full()` trả **PASS**: all(empty)=True, P2/resource
   không có dữ liệu được gán pass=True/SKIPPED.
2. Direct=.99 làm causal drop=0 và shuffled gain=0 vẫn PASS. Gain=.30 còn làm
   lowerCI=−.5 được chấp nhận qua nhánh OR. Không thể dùng accuracy cao thay
   bằng chứng học hidden hoặc uncertainty.
3. P2 acquisition=.9/final=.7/mean forgetting=.20 so baseline=.30 vẫn PASS:
   evaluator chỉ đòi giảm5đ, không kiểm mean<=3đ/worst<=5đ hoặc từng seed.
4. `P2_MAX_FORGETTING_THRESHOLD=.20` không được dùng; report lại ghi mean<=5đ,
   worst<=10đ. Mọi phiên bản đều lệch ngưỡng master3/5đ.

**Sửa đã triển khai:** `evaluation/forensic/research_gates.py` không nhận thiếu
dữ liệu là PASS; đọc summary để phát hiện FAIL nhưng không chứng nhận từ mean;
tính retention từ raw R-matrix và đòi mọi seed qua đồng thời. Test mutation bảo
vệ NaN/Inf, seed thiếu/trùng, ceiling exception và average che seed thất bại.
Đây là sửa công cụ audit, không chấm lại rồi ghi đè verdict lịch sử của tác giả.

## 4. BF-Cascade: giảm forgetting đi kèm giảm acquisition/final

Tính lại toàn bộ stored DEV matrices, không train lại; ba seed được gán theo
thứ tự loop trong source vì các per_seed rows không chứa seedID:

| Arm | Acquisition | Final | Mean forgetting | Mean worst forgetting | Joint pass |
|---|---:|---:|---:|---:|---:|
| unconsolidated |93,88%|74,05%|19,84đ|36,07đ|0/3|
| BF-Cascade |83,07%|66,54%|16,54đ|33,85đ|0/3|
| slow cascade |87,50%|58,42%|29,08đ|45,05đ|0/3|

BF giảm mean forgetting**3,30đ**, nhưng acquisition giảm**10,81đ** và final giảm
**7,51đ**. Không thể gọi đó là hệ học liên tục tốt hơn chỉ nhìn forgetting.
Không có bằng chứng scalar/EMA+anchor/replay matched baselines đã được giải quyết.

Artifact gọi confirmation báo final60,03%, mean forgetting24,92đ so baseline
30,39đ: vẫn FAIL rõ. Nó không lưu per-seed R-matrix hoặc worst forgetting để
kiểm claim trên mọi seed. Cả DEV và confirmation source đều calibrate trên16mẫu
của **tất cả các task tương lai** trước stage0, trái yêu cầu no future-task
calibration. Không có task-ID channels là tốt nhưng chưa đủ gọi stream nghiêm
ngặt không biết tương lai. Model cũng reset working state mỗi episode.

## 5. Bài 8-key có lỗi query encoding, không chứng minh trần neuron

`generate_multikey_associative` dùng cùng code query cho keys**0/5/7**, và cùng
code khác cho keys**4/6**. Chỉ5querycodes cho8keys. Với bộ nhớ lưu cùng values,
query không cho biết key nào trong nhóm alias được yêu cầu.

Fixture gọi chính generator với background noise=0,256rows: bài2/4key không có
conflicting identical inputs ở mẫu này; bài8key có**7row xung đột**, ví dụ
rows19/39 cùng input nhưng nhãn0/1. Không cần train để chứng minh lỗi.

Liệt kê chính xác mọi8bit stored-value pattern và query uniform cho trần Bayes
**84,375%** do ambiguity riêng của query (giả sử biết toàn bộ writes noiseless).
Đây không phải accuracy neural và không giải thích toàn bộ điểm57–62%; nó bác
bỏ suy luận “điểm thấp chứng minh48neuron không đủ” từ benchmark này. 4key không
có lỗi code đó nhưng vẫn chưa có learnability/encoder/readout matched control để
kết luận capacity là nguyên nhân.

**Sửa đã triển khai:** preflight `require_injective_codebook` từ chối querycode
trùng trước training; test có cả injective/alias/Bayes bound. Chưa thay generator
lịch sử hoặc chạy lại kết quả sau khi đổi đề bài.

## 6. Persistence/runtime chưa đạt độ an toàn learner

Counterexamples không gọi learn:

- MTSG save/load đổi `sensory_threshold=.123` thành default.8; RNG không lưu.
  Một bước forward/trace tiếp theo lệch eligibility max**0,3224364**. Default.8
  không bị lỗi threshold này nhưng shuffled continuation vẫn thiếu RNG.
- State NPZ bị đổi thành float64+NaN vẫn được nhận rồi gán vào float32. Wrapper
  thiếu kiểm dtype/shape/finite/checksum/bounds/duplicate như checkpoint P0 đã có.
- BF chưa có dumps/loads. Không thể đề nghị nó làm chuẩn learner dài hạn trước
  khi có toàn fast/slow/head/bias/normalizer/eligibility/RNG/controller checkpoint.
- BF luôn cấp phát w_ctx nhưng memory_bytes bỏ768byte khi contextgatingFalse:
  báo33.320byte, mảng thực34.088byte. Đây là lỗi accounting nhỏ, không phải OOM.
- `runtime-bench` lấy RSS **cuối lượt** rồi đặt tên peak, chưa đo cả process tree,
  serialization/rollback. Spike sparsity86,89% không có nghĩa CPU bỏ86,89%neuron:
  forward hiện vẫn GEMV trên toàn48neuron. Không suy ra điện năng hay speedup.

Trong audit hiện tại,13unit tests gốc vẫn PASS, cho thấy chúng chưa bao phủ các
khoảng trống trên.39tests audit/forensic qua, trong đó17test cho gate/codebook
mới. Peak RSS process tree các lệnh nhỏ tối đa175,22MB, commit137,60MB, dưới2GB.
Không dùng số đo audit này để chứng nhận runtime R8 dài hạn.

## 7. Protocol, raw và mức tin cậy

- Protocol/source/artifacts được đưa lên trong cùng một commit; Git không tự
  chứng minh protocol đã khóa trước khi chạy. Cần run manifest/protocol hash,
  code SHA, counters, commands/logs và timestamps thay cho nhãn “pre-registered”.
- Seed129001 đã dùng làm R7audit seed; không thể gọi toàn bank5seed là mới chưa
  thấy. Master yêu cầu tối thiểu8freshconfirmationseed. Codex không chạy chúng.
- Report ghi XOR R35probe96,42%, raw cùng file94,921875%; CI/baseline một số dòng
  prose cũng lệch raw. Chấm theo artifact/code, yêu cầu sửa prose và giữ lịch sử.
- Redteam summary viết candidate vượt stress dù cùng JSON có heavy-distractor
  rejected. Reversed mean forgetting17,19đ vẫn được gọi CONFIRMED_ROBUST vì code
  chỉ falsify khi>25đ. Đây không đáp ứng master3/5đ.
- Năm vai trò được chính agent thực hiện tuần tự; đây là tự kiểm nhiều vai,
  không phải năm reviewer độc lập. Không có bằng chứng cần suy diễn giả mạo;
  chỉ không dùng các vai đó như independent replication.

## 8. Tiến độ thực tế và phần nên lấy

**Nên giữ:** candidate multiscale/gated hiện có, diffusion dùng old gap và cả
readout, equal-episode no-replay baseline, source/raw R-matrix, unit fixtures.
**Đã sửa ở Codex:** independent gate, mọi-seed retention, key-code preflight,
model-free ambiguity diagnosis, mutation tests và reproduction các lỗi state.
**Đã sửa model-state persistence ở adapter riêng:** RNG/threshold, all known
arrays/scalars, strict loading, atomic save/rollback và cold continuation;
checkpoint serializer gốc của tác giả giữ nguyên. **Chưa sửa cơ chế học:**
matched-shuffle trace, isolated multiscale/gate ablations và causal matching;
chưa có quyền đưa candidate/adapter vào model live.

Ưu tiên tiếp theo được cụ thể hóa tại `R8_REPAIR_AND_P1_PROPOSAL_V1.md`.
Không tăng H48, không thay ALIF, không thay bằng Transformer/LLM, không mở corpus
lớn. Mục tiêu gần nhất là một đối chứng causal đáng tin cậy, chưa phải chat/vision
hoàn chỉnh. **`NEEDS_RESEARCH_APPROVAL` cho thay learning controls/chạy P1;
`NOT_PRODUCTION_READY`.**

Evidence: `artifacts/gemini-r8-independent-audit-v1.json` trỏ unique run
`20261112T020337Z-gemini-r8-independent-audit-v1-d98cc3`. Peer checkout là bản
detached thuộc Codex, không phải worktree đang viết của Gemini; tracked source
được giữ nguyên. Code/source snapshots, input hashes, commands, logs và JUnit
được lưu trong run. Không chỉnh main/Hermes/private data hoặc ghi đè run cũ.
