# Capacity H64/H128 — kiểm toán kỹ thuật độc lập

Ngày12/11/2026. Reviewer `/root/capacity_independent_review`; không chạy DEV,
không chọn/tune learner, không sửa `capacity.py`, codec, runner hoặc protocol.
Chỉ thêm `tests/test_capacity_independent.py` và tài liệu này. Ponytail không
có trong tool inventory. Main/Hermes/liveSQLite/process/port không bị can thiệp.

**Technical GO để mở DEV đã được người dùng duyệt, chưa quality GO.**

## Quyền và phạm vi

Đã đọc AGENTS, README, PROJECT_MAP, STATUS, proposal capacity và execution
registration `experiments/capacity-h64-h128-approved-v1.md`. Phê duyệt trực tiếp
mới của người dùng cho H64/H128 vượt thông báo “chưa phê duyệt” và giới hạn
scaling lịch sử trong đúng phạm vi research này. Không cho phép main/Hermes,
fresh confirmation, P2, language training hoặc mở rộng tiếp vượt H128.

## Implementation đã đọc và đối chiếu

- `CapacityOwn.step` và `.learn_episode` là chính các method kế thừa của
  `MatchedLocalCredit`, không viết lại phương trình/optimizer.
- `CapacityACR.step`, `.learn_episode`, `.fixed_alpha`, `.reset_episode` là các
  method kế thừa của ACR gốc. Constructor generalized theo H và profile vẫn
  `groups_up`; không thêm task ID hoặc chọn alpha theo accuracy.
- Constructor giữ sensory threshold0,6; gamma linspace0,25..0,92; initial
  mean0/variance0,01, momentum0/energy1e-4; signal PCG64 seed+35100000 và router
  PCG64 seed+43100000. Weights/topology vẫn do SpikingCore tạo. Không thay
  forward hard-ALIF, sparse fan8, D8/O2, R12 hoặc learning rates/clips.
- H48 technical fixtures khớp toàn bộ core arrays, active/auxiliary traces,
  optimizer/normalizer/RNG/counters/loss tại mỗi cập nhật cho cả correct,
  readout-only và shuffled conditions; không chỉ kiểm prediction cuối.
- Trong cùng H, bốn arms có initialization core/normalizer/optimizer exact và
  hard neural forward giống nhau trước cập nhật. H64/H128 chỉ cấp thêm mảng,
  không lấy weights pretrained hoặc chạy model khác trả lời.

## Checkpoint và recovery

Codec giữ format/kind cũ chỉ H48. Hai capacity kinds dùng distinct
`scarx-capacity-research-snapshot-v1`, whitelist H48/64/128, D8/O2/fan8/R12,
fixed gamma/policy/mechanism hash và strict array schema. Shape/dtype/body,
duplicate/member/byte bounds, hash và RNG state được kiểm qua adapter data-only;
không pickle và không import class được chỉ định bởi payload.

Snapshot round-trip và next-update khớp byte cho tất cả kích thước/conditions.
Test độc lập bổ sung cold subprocess thực: lưu sau tick7 của một episode,
khôi phục và chạy **phần đuôi chưa xong trước**, rồi mới học episode tiếp theo;
snapshot sau phần đuôi và sau update đều exact ở H64/H128, cả Own/ACR. Điều
này tránh gọi `learn_episode()` ngay sau load (hàm đó reset episode và có thể
che lỗi lưu neuron/eligibility state giữa episode).

Bounds/unknown-layout/unknown-neuron/large H bị từ chối trước allocation của
core lớn; không âm thầm padding H48 thành128. Signed-metadata tampering đối
với shape/dtype/indices/gamma và unknown optimizer state bị từ chối trong tests.

Một điểm có thể siết về sau: alpha/router_bias của fixed groups_up vẫn là
saved arrays; loader kiểm alpha range nhưng chưa yêu cầu toàn vector alpha
đúng profile cố định. Tick kế tiếp luôn tái tính fixed alpha, nên chưa thấy
ảnh hưởng học trong runner hiện tại; không coi đây là blocker chất lượng.

## Kết quả kiểm chạy thật

```powershell
python -X utf8 -m pytest -q -p no:cacheprovider tests/test_capacity_independent.py tests/test_capacity.py
```

**55 test PASS**, wall4,37s ở lượt reviewer cuối. Trong đó file độc lập có27
test; fixture kỹ thuật cố định, không phải task dataset, không mở DEV bank.
Kiểm thêm bằng source cho biết core `plastic_rows=12` không hạn chế wrapper
learning còn12neuron: optimizer inherited cập nhật **toàn bộ H×16 synapses**.
Fixture xác minh có weights đổi ngoài core.rows và số rows đổi lớn hơn12.

Mảng thực khớp công thức Own=`596H+864`byte và ACR=`944H+864`byte ở cả3H.
Unique trainable weights=`18H+2`; không nhầm stable copies/traces là independent
parameters. Đây là array bytes, không tổng process RAM, checkpoint buffer hay
capacity kiến thức. Native DLL chưa có lúc reviewer xem; parent cần build/run
shared-core CPU/native parity trước quality theo AGENTS. Không có native port
riêng của ACR để tuyên bố ở đây.

## Các điều kiện diễn giải đã được khóa

1. Fixed fan8 làm density8/H đổi khi tăng H; gamma linspace được resample và
   profile chia hai nửa theo normalized neuron position. Báo các covariance
   này, không mô tả H là factor duy nhất hoàn toàn độc lập.
2. Seed ghép cặp giữa H tạo cùng distribution/init rule, không cùng weight
   prefix: generator topology tiêu thụ số random draws khác. Trong cùng H
   initialization giữa arms phải exact.
3. H48 primary scores có thể reuse sau parity. Không dùng training CPU của
   cached dual-stream alpha0 làm cost của Own single-stream. Registration chạy
   mới108equalCPU timing cells trên3H là cách đối chiếu phù hợp hơn.
4. Primary144records gồm48reuse và96new; secondary108new =>252records,
   **204new training jobs**. Three independent seeds, không252independentseeds.
5. Full-revert/refit phải về readout-only cùng H; thêm neuron làm frozen random
   representation tốt hơn chưa phải hidden synapses học tốt hơn.
6. Direct/probe/OOD/order và equalupdates/equalCPU phải báo riêng. Không thay
   gate sau kết quả và không lấy ceiling Order làm representation success.
7. Đây là capacity DEV diagnostic, không test forgetting/P2 hoặc AI đa năng.
   Nếu H128 không tiến bộ vẫn chưa chứng minh architectural ceiling: learning,
   data, objective/readout và optimization budget còn là các giải thích khác.

Không thấy blocker về phương trình, initialization hoặc cold state recovery
trong implementation đã kiểm. Cần source/protocol/resource lock, throughput
preflight, native parity và audit dữ liệu/metrics thực khi có kết quả trước
bất kỳ quyết định quality/promotion nào. Reviewer chưa phê duyệt quality.
