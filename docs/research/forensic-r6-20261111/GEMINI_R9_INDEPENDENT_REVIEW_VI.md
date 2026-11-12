# Gemini Round9 — kiểm toán source/raw độc lập

**Kết luận: `REQUEST_CHANGES`; chưa approve learner vào main.** R9 có code và
artifact thực để nghiên cứu, nhưng chưa chứng minh giải được distractor/retention
hoặc sự cần thiết phải đổi neuron. Không dùng số512hay2.400lượt làm metric năng lực.

Source chốt **2c41fdaf6fc491845879d638ac80449caf597471**, base607ffca7c9222c5e3fd9741c3ebdf5939e10bedf.
Đã đọc trọn attachment SHA256
`364b350ffc6e84685e3ff155d35d1963d102cf9d5d8d10880aeda1700ed7c1c0`, protocol,
framework neural, generator, runner và toàn bộ các JSON/JSONL được dùng trong
phép tính audit. Không chạy lại search512hoặc StageF confirmation.

Codex chạy **10test kỹ thuật tác giả**,3test bookkeeping độc lập và các fixture
không gọi learn để kiểm forward/trace/context/RNG. Unit fixtures tác giả có tiny
learning checks; không chấm chúng như bằng chứng hiệu quả nghiên cứu. Source/main/
live/Hermes giữ nguyên. Lượt12seed của Codex tiếp tục trong worktree khác, không
thay code, bank hoặc tiêu chí theo kết quả R9.

## 1. Kết quả thực sự hiện có và giá trị có thể tận dụng

R9 có manifest512config với canonical hash khớp,512screen rows có composite score
tính lại đúng, ranking theo nhóm, ablations, stress và summary10seed của2candidate.
Các sửa mới đáng giữ:

- Factory duy nhất truyền cùng params cho learned/frozen/shuffle/stress.
- `predict_many` context-gated nay dùng đúng context head như `predict`.
- Attenuation được áp nhất quán trong training và inference sau patch.
- Bỏ ngoại lệ causal gate dựa vào accuracy cao; gate retention có cả mean3đ và
  worst5đ. Có list10per-seed worst-forgetting và max khớp summary.
- Phân biệt probe/refit và direct fixed-head revert ở các stage sau là hữu ích,
  nếu ghi rõ metric khác nhau thay vì cộng/trộn như một endpoint.

Summary hiện có (chưa được Codex tái train10seed hoặc chấm row-level):

| Candidate | Binding direct | Distractor direct | XOR direct | Final stream | MeanF | WorstF |
|---|---:|---:|---:|---:|---:|---:|
| H2-012 |74,75%|58,96%|66,74%|56,52%|14,61đ|42,58đ|
| H5-070 |56,29%|51,41%|69,14%|52,92%|18,15đ|48,05đ|

Cả hai giữ **GATE_FAIL**. H2-012 có tín hiệu binding, nhưng XOR probe gain âm;
H5-070 không có representation gain đủ chắc chắn. Không candidate nào trong
bank xác nhận này đạt bộ yêu cầu chung. Kết luận FAIL không cần nới ngưỡng.

## 2. Finding P1 — screening cache cũ sau khi sửa code ảnh hưởng kết quả

So commit39fd104 với2c41fda:

- Sửa inference context head và attenuation; sửa factory frozen/learned/stress.
- **File512screen rows byte-identical trước/sau patch**. Loader resume chỉ dựa
  candidateID, không kiểm hash source/generator/evaluator/budget trước nhận cache.
- Do đó ranking đang dùng screening từ code cũ đã có lỗi để chọn stage sau bằng
  code mới. Hash params không đủ đại diện cho version của mô hình đã chạy.

Phải giữ screening cũ với nhãn legacy và sourceSHA tương ứng. Nếu cần screening
đúng phiên bản, đăng ký một lượt bounded mới; không ghi đè cache rồi gọi lịch sử
đã tái lập trên code mới. Codex không tự mở lại cả512sweep.

## 3. Finding P1 — đổi candidate và dùng lại bank đã mở

Artifact39fd104 xác nhận **H4-049/H4-050**; artifact2c41fda chuyển sang
**H5-070/H2-012**. AST của cả hai runner có cùng bank
191001,191007,191013,191019,191025,191031,191037,191043,191049,191055.
Selection rule cũng đổi sau khi có run trước.

Đây cần được ghi là revised selection/corrected replay trên bank đã bị xem,
không phải một phép xác nhận fresh chưa hề mở. Không suy diễn ý định gian lận;
vấn đề là provenance và tính độc lập. Thiếu per-seed predictions/checkpoint/
source manifests khiến Codex chưa thể tái tính CI/gates sâu hơn từ raw.

Bank R9 **không trùng** bank12seed20261112xxđang chạy của Codex. Candidate/code
của Codex đã khóa trước review này và giữ nguyên, không đưa ranking R9 vào đó.

## 4. Finding P1 — BH/FDR chỉ được định nghĩa/import, không được áp dụng

AST runner có **0call** đến `benjamini_hochberg`. Có unit test cho helper BH
không có nghĩa cả campaign đã kiểm soát FDR. Không có bảng p/q-family cho512
candidate như claim. Không dùng nhãn thống kê thay kết quả của pipeline thực.

P1 của R9 vẫn dùng mean gain>=5đ/lowerCI>0, khác master yêu cầu lowerCI>=5đ.
P2 acquisition/final/mean lấy trung bình; dù worst.maxđã chặn seed có worst-loss
lớn, vẫn thiếu per-seed acquisition/final/mean hoặc full R-matrix để kiểm đầy đủ
mọi seed theo master. Hai candidate hiện FAIL rõ nên chưa tạo một falseGO.

## 5. Finding P1 — reset/attenuation chưa hỗ trợ kết luận tất yếu về kiến trúc

Fixture không train: `subtractive_exact` và `hard_reset` có voltage forward
**maxerr0**, nhưng eligibility lệch **2,5213947**. Cả hai vẫn gọi cùng subtractive
ALIF core. Vì vậy đây là so hai derivative approximations, không phải thử neuron
hard-reset v=0 so với neuron subtractive-reset.

R9 không có lựa chọn stopped-reset `e_v=u_e` đang được Codex kiểm. Không thể dùng
kết quả hard_reset của R9 để bác bỏ stopped-gradient reset. Implementation e-prop
gốc dùng subtractive reset với stop-gradient ở spike path và recurrence
epsilon_v=alpha*epsilon_v+z_pre; không phải phép nhân(1-z)của option R9.
[Source tác giả đã pin](https://github.com/IGITUGraz/eligibility_propagation/blob/efd02e6879c01cda3fa9a7838e8e2fd08163c16e/Figure_3_and_S7_e_prop_tutorials/models.py#L295).
Không gọi một variant SCAR-X là bản sao chính xác full e-prop khi convention
surrogate/refractory/timing/feedback khác nhau.

H3 tên `learned_attenuation` nhưng code chỉ dùng **channel6 biết trước**, threshold
abs>.5 và một attenuation_factor cố định. Không có tham số attenuation được học
qua trải nghiệm. Nó cũng đổi effective sensory forward, không chỉ learning signal.
Thất bại của grid này không chứng minh mọi learned inhibition hoặc ALIF một ngăn
không thể chống nhiễu. Chưa có cơ sở kết luận dendritic2compartment là bắt buộc.

## 6. Finding P1 — raw context trace làm mất dấu của query bipolar

Context head cập nhật `e_ctx=.85*e_ctx+.15*abs(row)`. Key/query4bit của bài4/8/16key
là bipolar ±1,5. Fixture chỉ lật toàn bộ4querybits, context trace cuối **giống
hệt**. Hidden features có thể vẫn khác; không phải cả model tất yếu mù với key.
Nhưng context pathway đang mất thông tin query identity mà nó cần để routing.

Đây là bottleneck encoding/readout cần ablation trước khi quy lỗi cho dung lượng
48neuron. Grid H4 bị cắt80: context_gated có12config2key,12config4key,8config8key,
**0config16key** trong screening. StageE có thử16key nhưng điều đó không biến
screening thành full-factorial khớp ngân sách trên mọi keycount.

## 7. Số cấu hình/lượt/thời gian/RAM cần định nghĩa lại

-512canonical params/config hashes đúng; sau bỏ các tham số vô hiệu theo source,
  còn412model signatures hoặc464model+evaluation signatures. Đây là count của
  signature đã normalize, không phải định lý mọi model trong mỗi nhóm tương đương.
  Ví dụ gatingnone thì đổi threshold không tạo learning rule mới.
- SourceStageB gọi screen_*1536lần theo nhánh task/group, không đồng đều512x2x2.
  H1/H2chạy2tasks/seed; H3/H4/H5chạy một screen khác loại.512aggregate rows không
  đủ để xác minh ledger2.400run theo một định nghĩa chung hoặc mọi run khôngVOID.
- Different groups dùng384episodes/task nhưng H1/H2có2tasks,H5có3skillstream;
  rank tổng hợp không phải so mọi cơ chế trên cùng một bài/budget.
- H5screen gán `frozen_probe_acc=.50`, `probe_gain=acquisition-.50`,
  `causal_revert_drop=acquisition-final`, `ood_acc=final` **không đo các endpoint
  mang tên đó**. H4screen cũng gán ood_acc=direct ID, không có OOD riêng.
  Không được đem proxy có cùng fieldname ra so với actual probe/revert/OOD của H1.
- `uniform_mid` thực tế(.25+.92)/2=.585, không phải.60 như prose. Surrogate widths,
  thresholds và disabled-cascade rates có ignored-parameter duplicates.
- Record runtime sumStageB1221,42s, final search_summary332,76s; prose nói1897s.
  Resume có thể giải thích khác biệt, nhưng cần per-stage/source ledger. Không
  kết luận tất cả thí nghiệm mới đã chạy trong332s hoặc số liệu bị bịa.
- `peak_ram_mb` ở summary là RSS **cuối lượt của một process**, không peak toàn
  campaign/commit/sharedworkers. Protocol còn ghi total6GB, khác mục tiêu2GB của
  hệ SCAR-X; không lấy một finalRSS nhỏ làm chứng nhận đã đạt trần toàn hệ.

## 8. State persistence vẫn chưa đủ cho deployment

RNG không được save/load; fixture xác nhận state PCG đổi sau load. Xóa syn_trace
khỏi state NPZ vẫn được nhận và silently reset momentum về0. Unit roundtrip chỉ
so một số weights không phát hiện lỗi next-learning continuation này. Đây là
lỗi kỹ thuật; không tự suy rằng mọi inference/revert score đều sai vì RNG không
được dùng lúc inference hiện tại. Chưa đưa serializer này vào app.

## 9. Quyết định và thứ tự tận dụng

**Approve ở mức tài liệu/hạ tầng research sau sửa:** commonfactory, consistency
tests cho context predict, explicit no-ceiling gate, manifest/hash theo version,
failed results để tránh lặp các hướng không hiệu quả.

**Chưa approve:** learner/R9serializer vào main, promote candidate, đại sweep mới,
ngân sách6GB, đổi neuron/dendrite hoặc kết luận đã chứng minh giới hạn kiến trúc.
Nếu nghiên cứu tiếp sau confirmation đang chạy: chọn1–2factor có hypothesis rõ,
matched forward/budget/optimizer, raw per-seed predictions/R-matrices và source
digests; phân biệt derivative approximation với membrane reset và readout với
hidden learning. Không lấy candidate ranking đã lỗi cache làm shortcut.

Evidence cuối: `artifacts/gemini-r9-independent-review-v1.json` trỏ run
`20261112T062822Z-gemini-r9-independent-review-v1-9453bf`. Technical evidence tái
sử dụng từ run40df26, source hashes helper/worker đã kiểm trước reuse; không
chạy lại unit tests hoặc quality chỉ để làm đẹp số. AST tính bank trước/sau và
overlap thật trong finalrun; preliminary run40df26 chỉ dùng technical evidence.
Tất cả code/output mới ở worktree forensic của Codex, không sửa source R9 hoặc
source/worktree của confirmation đang chạy. Overall: **NOT_PRODUCTION_READY**.
