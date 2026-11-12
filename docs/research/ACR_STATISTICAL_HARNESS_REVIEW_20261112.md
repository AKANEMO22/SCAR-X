# ACR — kiểm toán độc lập statistical/postprocess harness

Ngày12/11/2026. Reviewer `/root/acr_final_stats_review`, chỉ đọc source/raw;
không sửa learner, không gọi `learn_episode`, không chạy DEV/confirmation/P2.
Ponytail không xuất hiện trong tool inventory. Phạm vi này không thay review
toán ALIF, endpoint hoặc final raw audit trước đó.

Run đang kiểm: `20261112T121634Z-acr-dev-v1-resume-59ced4`. Các nhận xét dưới đây
được viết khi worker còn chạy; **chưa chấp thuận quality hoặc final completeness**.

## Những phần đã kiểm

- Đọc `experiments/acr-dev-v1.md`, addendum phục hồiv3, runner, learner, study,
  `audit_acr_dev.py`, `audit_acr_mechanism.py` và `render_acr_report.py`.
- Đối chiếu sáu fixture độc lập cho scalar3seed bootstrap và exact sign-flip:
  allzero; allpositiveequal; balancednegative/zero/positive; monotonicpositive;
  mixedunit extremes; meanzero nonuniform.27draws bootstrap có hoàn lại,
  inverse-CDF95% và8signvectors khớp NumPy reference. Nonfinite bị từ chối.
  Exact two-sided p nhỏ nhất0,25; không có mức ý nghĩa confirmation từ3seed.
-220mean metrics AB (11profiles×5environments×4metrics) dùng scalar
  `statistics.mean` khớp `np.mean` trong worker, maxerror0 trên raw thực.
- Grid logic đã deduplicate static50/nominee normpair và state-shuffle alias.
  Trường hợp tối đa là345jobs/375model records; số model tăng do hai trajectory
  của normpair không được tính thành thêm seed. Có thể ít hơn nếu nominee là
  static50 hoặc state, hoặc C không mở; không được bắt buộc ghi375mọi trường hợp.
- Nomination dùng một profile toàn cục trên primaryisolated groups; joint chỉ
  veto. Profile/readout/shuffle condition được lọc riêng. Không có tên task hoặc
  testlabel trong `AdaptiveCreditRouting.route`.
- Broad checks giữ direct65%, mean margin2đ và hidden/readoutgain5đ kèm direct
  causaldrop3đ, endpoint/OOD/protected/jointNI3đ. Shuffle/norm margins lần lượt
 2đ/1đ trên ít nhất2families; regression direct/probe được kiểm riêng. Đây là
  exploratory effect rules, không chuyển thành P1significance/HolmPASS.
- Missingresult chỉ được giải thích đúng hai trajectory của explicit invalid
  normpair. Không để một controlinvalid che cell khác bị thiếu. Positive-radius
  norm failures giữ trạng thái fail-closed; R0 không được gọi plasticitygain.

## Continuation và compute đã mất

Kiểm trực tiếp recursive provenance:

| Copy | Cell giữ lại | File giữ lại | Kết quả |
|---|---:|---:|---|
| f947fb→ac9bba |55|220|parentmanifest và cả hai bản filehash khớp |
| ac9bba→59ced4 |203|812|parentmanifest và cả hai bản filehash khớp |

Tổng1.032so sánh file theo từng link không phải1.032file độc lập:55cell được
giữ qua hai link. V3limit5800,1977465s + observedv2elapsed1884,8037654s +120s
uncertaintyreserve = parentlimit7805,0015119s đúng arithmetic. Thời gian mất
giám sát không có exactdeath timestamp, nên allowance không chứng nhận tổng
walltime thực dưới9.000s một cách tuyệt đối. Báo observedcompute/reserve riêng.

Cell bị mất optimizer state có ít nhất128, nhiều nhất1.024update chưa được
checkpoint. Replay đúng fixedcell/init/data/RNG một lần là ngoại lệ phục hồi đã
ghi rõ; compute đó phải báo là chi phí thừa. Không xem nó như seed độc lập hoặc
0updates. `reused_completed_cells=203` là số cell không học lại, không phải số
cell được huấn luyện mới trong v3.

## Finding/hardening trước công bố cuối

1. Auditor chính đã kiểm mainSHA, resourceobservedlimit/error, result/job/row
   counters và tồn tại144diagnostic/36stressreceipt; cần giữ các kiểm đó ở final.
   Kiểm `mechanism-audit-v1.json` status/counts riêng chưa chứng minh các artifact
   còn nguyên sau audit: finalauditor nên rehash toàn bộ `artifact_sha256` từ
   receipt trước gán `mechanism_stress_verified=true`.
2. `audit_acr_mechanism.py` replay stressprobe bằng chính `refit_predict` mà
   runner dùng. Đây là recomputation/replay, **không phải implementation probe
   độc lập**. Eigen-probe của auditor chính và các SVD kiểm trước mới là đối
   chiếu implementation riêng; diễn giải không nhập nhằng hai loại bằng chứng.
3. Report template còn câu continuationv2 và chỉ historicalpeak phần55cell.
   Trước xuất bản cần bổ sung v3lost128..1024updates,203reuse, uncertaintyreserve
   và observedpeaks của từng segment. Không gọi samplerpeak cũ là finalglobalpeak.
4. Report phải tổng hợp actual alpha/time/neuron, eligibility direction/norm,
   OODstress và counterexamples từ artifact thật; genericmechanismbullets không
   thay yêu cầu này. Diagnosticclones không phải originaltrainingtrajectory.
5. Label `ACR_SPECIALIZED_GAIN` hiện chỉ khi đúng một family rawimprove, phù
   hợp câu chữ protocol. Nếu nhiều rawimprovements nhưng failbroadguards, cần
   báo rõ các tradeoff và thất bại; nhãn no-qualified-gain không được dùng để
   che improvements có thật. `ACR_REPRESENTATION_ONLY` nghĩa chưa có directgain
   đạt margin2đ, không nhất thiết mọi directdelta đều bằng0.

## Giới hạn của nhận xét hiện tại

Chưa có finalresult hoặc đủ normcontrols tại lúc review; không chấm winner
hay qualitydecision. Sourcefrozen và algorithm không thay đổi bởi các kiểm
postprocess. Không có bằng chứng đã giải catastrophicforgetting hoặc tích lũy
kỹ năng task-agnostic từ pooledjoint/independenttask fits; P2 vẫn chưa mở.

**Interim verdict:** không thấy blocker thống kê cho tiếp tục grid đã khóa;
các hardening/diễn giải trên cần xử lý trước finaldelivery. Quality GO chưa có.
Số cập nhật học do reviewer tạo: **0**.

## Final independent review — worker hoàn tất

Final result của59ced4 không lỗi:345jobs/375model records đúng registered grid,
không có missing hoặc invalidnormpair. Finalreview dưới đây chấm raw trực tiếp,
không coi việc auditor của parent chạy thành công là bằng chứng mặc định.

| Kiểm độc lập | Số lượng / kết quả |
|---|---:|
| Prediction entries ID/OOD và3hidden interventions |3.072.000, scalaraccuracy khớp |
| Accepted trajectory history entries |321.539, step/CPU/norm/counters khớp |
| Full checkpoint byte round-trip và RNG receipt |375/375 |
| ALIF literal replay không gọi candidate step/predict |3.000prediction khớp |
| Full-revert refit probe về frozen representation |240/240equal-updatecells exact |
| Actual physical norm control |30pairs/30.720steps,0zero,0invalid |
| Max norm relative error |9,999663298901383e-6 ≤1e-5 |
| Source/peer/dataset hashes |19/2/15khớp |
| Recursive reused file-link comparisons |1.032khớp |
| Mechanism/stress artifact hashes |180/180khớp |
| Supplemental router raw hashes |48/48khớp,192rollouts,0updates |
| Stress predictions scalar rescore |41.472entries/36files khớp |
| Filteredtrace mixture từ loggedalpha |144tapes, maxerror0 |

Manual ALIF replay khởi tạo v/a/spike/features bằng0, dùng hardthreshold và
subtractive adaptive reset đúng core, chuẩn hóa theo savedf_mean/f_var rồi đọc
Wout/bias. Nó không gọi `ACR.step`, `features` hoặc `predict`; chỉ đọc bounded
checkpoint. Schema profile/condition/seed của mỗi checkpoint cũng khớp row.

Đã tái dựng độc lập toàn bộ globalrank, broadchecks và33contrasts (7baseline,
2shuffle,2norm trên3primaryfamilies). Effects, descriptiveCI và exactp khớp
`audit-v1.json`. Nominee toàn cục vẫn `groups_up`; rank là
`(-0.029296875, 0.0021701388888888395, false)`. Stateheuristic không được chọn
riêng cho task nào. Readout/shuffle và modelindex của normpair không bị gộp
nhầm vào các mean primary.

### Kết luận theo gate đã khóa

- `groups_up` đạt Bindingdirect85,026%, XOR98,112%, nhưng Distractor59,245%<65%.
  Chỉ Distractor có rawdirectgain≥2đ so betterendpoint. Không family nào được
  miễn causal/OOD chỉ vì điểm gần100%.
- Hiddenprobe gains so readout: Binding26,237đ, XOR15,169đ, Distractor6,641đ;
  directcausaldrop lần lượt33,594đ/41,016đ/8,203đ. Đây là evidence event-task
  representation/usableoutput hẹp, không một learner lifelong đa kỹ năng.
- Shuffleprobemargin qua3/3; normprobemargin qua2/3, không có meancontrol
  regression vượt3đ. Normmatchedprobeeffects từng seed được kiểm độc lập:
  Binding `[+.009765625,-.03515625,+.009765625]`, XOR
  `[+.029296875,+.0390625,+.134765625]`, Distractor
  `[+.029296875,+.05078125,-.021484375]`. Không gọi đây là significance3seed.
- Broadgate vẫn FAIL: chỉ1causalimprovingfamily, Distractor dưới65%; joint
  sharedcheckpoint regress Binding, Distractor và Order. Jointmean direct của
  groups_up là56,510%/72,591%/52,734%/78,841% theo thứ tự Binding/XOR/Distractor/
  Order. Pooledjoint là trainingmixedIID, chưa phải retentionstreamP2.

**Final verdict: integrity PASS; `ACR_SPECIALIZED_GAIN` đúng protocol.**
Không `ACR_DEV_ADVANCE`, không confirmationproposal đủ điều kiện, không nhận
learner vào main/Hermes. Static/neuronspecific signal được giữ như research
evidence; state-adaptive routing chưa chứng minh broad advantage. Kết quả
không bác bỏ hoặc xác nhận toàn bộ họ thuật toán trong literature.

### Resource, provenance và số lần học thực tế

V3worker1620,74193s < remaininglimit5800,1977465s. Observed ownedtree peakRSS
177.618.944B và committed220.549.120B; measurementerrors rỗng, nooverlimit.
Windows cap receipt ghi enforcedtrue. Samplerflag riêng không đồng bộ với
receipt không được dùng thay số đo hoặc suy ra bytesđãswap/energy.
Main ref vẫn `e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541`, trackedmaincheckout
sạch. Không đọc productionSQLite hoặc tác động processHermes trong review.
Git/main cleanliness không tự chứng minh mọi live state trong máy; phạm vi
đã kiểm là source/ref và researchartifacts.

321.539historyentries là acceptedtrajectory update calls, gồm cả hai learner
của30normpairs. Thêm144diagnostic clone updates đã đăng ký và128..1.024updates
của lostattempt: tổng số học thực đã thực hiện có lowerbound321.811 và
upperbound322.707. Calibration/forwardoracle không được tính thành synaptic
learning. Các lostupdates không tạo thêm seed hoặc evidence quality độc lập.
Tổngactivewall của các đoạn gián đoạn vẫn có phần không được giám sát; giữ
observedtime và uncertaintyreserve riêng, không claim certificate exact9.000s.

Các hardening trước công bố đã thấy được áp dụng: main/resource/countchecks,
rehash180mechanismartifacts/48routerfiles và sửa wording replayprobe. Final
report phải tiếp tục giữ v3recovery/computeoverhead, actualrouter/stress và
counterexamples; không chuyển integrityPASS thành qualityGO.

Số lần gọi learner hoặc cập nhật học do final reviewer tạo: **0**.
