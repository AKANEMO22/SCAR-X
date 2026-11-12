# ACR A/B — kiểm toán độc lập, 12/11/2026

**Endpoint1 thực nghiệm: PASS exact. A/B integrity: PASS. Chưa có quality GO
hoặc chấp thuận production.**

Reviewer thực `/root/reference_audit_resume` kiểm source/raw bằng chương trình
độc lập, không học, không gọi lại learner.learn_episode và không thay tham số.
Ponytail không có trong inventory. Chỉ ghi tài liệu kiểm toán mới này.

Run tiếp nối `20261112T082034Z-acr-dev-v1-resume-ac9bba`, source commit
`4e871eb1f6fb397e9a82e357c373176446342d1c`. Learner/filter/router/protocol gốc
không đổi so source khóa trước gián đoạn. Manifest hash đối chiếu đúng. 55cell
đã hoàn tất từ runf947fb được copy bảo toàn220file hashes, không train lại.
165A/B cells gồm cả55cell tái sử dụng; số entries lịch sử kiểm là168.960update,
không phải168.960lượt học mới do reviewer tạo.

## Endpoint alpha1 trên toàn bộ1.024updates

Đối chiếu12isolated ACR `static_100` với `stop` trong literature reference run
`20261112T080533Z-literature-reference-dev-v1-839e07`, cùng4task×3DEVseed:

| Thành phần đối chiếu | Số lượng | Kết quả |
|---|---:|---|
| Mọi direct/probe prediction và3hidden intervention, ID/OOD |61.440entries|Exact |
| Mọi metric lưu kèm |12cells|Exact |
| Loss từng lượt học |12.288losses|Exact |
| Toàn bộ core arrays và fingerprint |216array pairs|Exact |
| Selected stop e_v/e_a/e_f và mixed e_f |48array pairs|Exact |
| Momentum/energy, normalizer, initial hidden weights và gamma |84array pairs|Exact |
| RNG signal và episode/hidden/tick counters |12cells|Exact |
| Các mảng dữ liệu TRAIN/refit/ID/OOD |96array pairs|Exact |

Hai checkpoint có class/schema khác nhau nên không đòi ZIP byte-identical
giữa hai class. Các trạng thái tương ứng phải exact; inactive own trace của
ACR không được dùng làm tiêu chí khớp stopped baseline. Alpha0/own đã được
kiểm ở vòng trước:12isolatedreferencecells khớp toàn bộ predictions và causal
interventions với ACR alpha0. Endpoint1 lần này bổ sung bằng chứng sau1.024
updates thật, không chỉ fixture ngắn hoặc inference giống nhau trước học.

## A/B grid, matching và scoring

Đủ165cell:11profiles×5environments×3seeds. Profiles là5uniform static,
linear_up/down, groups_up/down, intrinsic và readout_only. Uniform neuron
profile là alias static; không đếm thành evidence độc lập. Seeds chỉ140011,
140017,140023, không dùng12confirmation hoặc R9bank để chọn candidate.

Chấm lại **1.351.680prediction entries** của A/B từ labels/predictions thô,
gồm ID/OOD và3interventions: mọi scalar metric khớp. Mỗi cell có1024history
entries hữu hạn, step1→1024. Metadata task/profile/condition đúng grid; chỉ
readout_only dùng conditionreadout, không nhập signal-shuffle vào candidate.

Joint có256TRAIN+64refit mỗi family, tổng1024TRAIN/256refit, input vẫnD8.
Evaluator lưu family labels riêng, không thêm task channel. Kiểm balanced
family masks và chấm lại mọi joint per-family metric trên512ID/OOD mỗi nhóm.
Một pooled head/probe được fit chung; không có bốn probe riêng theo task.
Joint là shuffled pooled training, chưa phải sequential retention hoặc P2.

## Nomination và điều kiện mở C

Reviewer tái tính mean từ seedrows, ranking và broad guards trực tiếp, không
gọi `nominate` hoặc `broad_checks` của code đang được đánh giá. Kết quả khớp
`stage-ab.json`: nominee **groups_up**, uniform_broad=[], open_state=true.
Ranking groups_up `(worst margin,mean direct margin,static preference)` là
`(-0.029296875,0.0021701388888888395,false)`, đúng ruleđã khóa trên primary
isolated tasks. Joint được xét như guard riêng, không lựa alpha theo task.

| Nhóm | groups_up direct | groups_up probe | OOD direct/probe |
|---|---:|---:|---:|
| Binding |85,03%|88,54%|85,81%/89,00% |
| XOR |98,11%|98,44%|85,22%/90,89% |
| Distractor |59,24%|60,48%|56,05%/57,75% |
| Order |100,00%|100,00%|99,93%/97,98% |
| Joint pooled aggregate |65,17%|73,73%|67,22%/71,08% |

groups_up chỉ có1family đạt điều kiện cải thiện kèm probe/causality; distractor
direct chưa65%. Joint preservation fail ở Binding, Distractor và Order theo
margin−3đ so endpoint tốt hơn trong cùng joint setting. Order isolated100%
không được dùng che các lỗi này hoặc gọi thành representation breakthrough.
Không uniform static.25/.5/.75 nào qua broad guard đã khóa. Vì vậy mở C sau
đủ165A/B cells là đúng nhánh protocol, không đổi threshold sau kết quả.

groups_up là profile cố định hai nhóm neuron, không phải learned router.
Source C vẫn dùng fixed coefficients và intrinsic bias đã đăng ký trước dữ
liệu; không lấy nominee/taskaccuracy để đổi alpha theo từng task. Kiểm route
cho thấy input chỉ có u/theta/spike/adaptation và trace history q. Nhãn chỉ
được dùng trong tín hiệu cập nhật cuối episode, không đi vào route function.
Core forward vẫn một lần/tick, hai eligibility stream riêng và trộn sau lọc.

## Phạm vi phán quyết

C đang chạy theo công thức đã khóa tại thời điểm kiểm này. Chưa có final
nominee, actual-norm controls, shuffle/equalCPU hoặc fullmechanism/OOD audit
cho toàn chiến dịch; tài liệu này không phê duyệt kết quả C chưa hoàn tất.
Ba seed DEV không đủ significance; configuration không phải independentseed.
Không thay P1/P2gate hoặc gọi một pooled checkpoint là lifelong learning.

**Kết luận:** endpoint và A/B harness có bằng chứng hợp lệ; chưa tìm được
static/neuronspecific learner vượt toàn bộ guard. Tiếp tục phầnC/controls đã
đăng ký, giữ mọi negative results, không promotion main/Hermes và không mở
fresh confirmation/P2. Reviewer không điều khiển PID/port và tạo **0learning
updates** trong lần kiểm này.
