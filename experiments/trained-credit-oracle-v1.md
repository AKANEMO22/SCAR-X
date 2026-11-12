# Frozen trained-credit oracle v1 — đăng ký trước tính diagnostic

Ngày12/11/2026. Parent giao phép kiểm cơ chế giới hạn sau reference DEV;
đây KHÔNG phải lượt huấn luyện mới, không chọn/tune candidate và không P2.
Nguồn duy nhất: completed run
`20261112T080533Z-literature-reference-dev-v1-839e07`, source
`a3b79ea66cb6676f8b03a22b0cb907d158e5b2a2`,132cells, không lỗi.
Ponytail không có trong inventory; không main/Hermes/liveSQLite/port8767.

## Khóa phạm vi

Methods: own,stop,eprop_matched,eprop_aligned,braintrace.
Primarytasks: strict_binding,xor,distractor_binding.
DEVseeds:140011,140017,140023; chỉ dùng final **equal_updates, unshuffled**
checkpoint tương ứng (45checkpoint). Bốn episode TRAIN cố định mỗi checkpoint:
indices0,63,511,1023. Tổng180rows; không tăng row/seed/index sau khi xem kết quả.
Không đọc nhãn ID/OOD; nhãn TRAIN đã học chỉ dùng tính tín hiệu loss tại cuối
episode, không cập nhật bất cứ parameter hoặc statistic nào.

## Đại lượng

Freeze toàn bộ hidden/readout weights, f_mean/f_var và optimizer state tại final
checkpoint. Clone model, reset episode và chạy từng row bằng local step để
được endpointfeature f và local eligibility E. Shared endpoint learning signal:

```text
scale=max(sqrt(f_var),.05)
raw=(f-f_mean)/scale; fhat=clip(raw,-5,5)
p=softmax(Wout@fhat+bias)
signal=Wout.T@(onehot(target)-p) * (abs(raw)<5) / scale
local_ascent_direction[j,k]=signal[j]*E[j,k]
full_ascent_direction=signal @ full_surrogate_sensitivity(frozen_core,sequence)
```

Full oracle giữ tất cả recurrent spike sensitivity, gồm self-edges; local
adapter giữ approximation đã công bố. Dấu ascent theo đúng source1_y-p;
đây không phải delta vật lý sau momentum/energy/clipping và không phải gradient
của learner đang thay f_mean/f_var. Không diễn giải khác biệt đó thành lỗi
của optimizer. Không gọi learn_episode hoặc sửa weights/counters/RNG gốc.
Chỉ clone dynamic neuron/eligibility state và total_ticks được thay khi forward;
coreupdates/episodes/hiddenupdates, optimizer, stats và RNG của clone phải giữ.

## Tính đúng, lưu chứng cứ, dừng

Toàn source hash từ reference manifest phải còn khớp; datasetZIP/modelZIP
đối chiếu receipt trước đọc. Protocol và script hashes được ghi manifest mới
trước tính row đầu và kiểm lại sau cùng. Snapshot original model trước/sau
diagnostic phải exactbytes; clone learning-state hashes trước/sau phải exact.
Oracle numerical correctness đã có independent anchored-smooth finite-difference
tests, không thêm một training oracle hoặc BPTT candidate.

Mỗi row lưu cosine, signagreement,normratio, local/fullnorm, frozenloss,
target/prediction, clippingfraction, checkpointSHA, TRAINindex, CPUtime.
RawNPZ lưu signal,local/fullvectors,localeligibility,features/target/loss để
chấm lại metric. Không dùng probe/accuracy để chọn rows. Normzero báo riêng,
cosineundefined lànull, không âm thầm bỏ dòng. Tổng hợp descriptive theo
method/task và seed; không p-value hay chiến thắng quality từ diagnostic.

Reservation riêng một lần; output unique run; max180wallseconds kể từ đầumain,
gồm preflight/I/O. Một numericalthread, process/tree dưới2.000.000.000B,
Windowscap nếu khả dụng và monitorRSS/commit. Kiểmdeadline mỗi row; timeout,
nonfinite/hash/state mismatch => dừng và giữpartial, không tự mở lại/sửa ngưỡng.
Không thay sourcelearner, upstream hoặc raw reference. Không freshconfirmation.

## Câu hỏi và giới hạn

Hướng D-RTRL từng gần fulloracle ở frozeninitialfixtures có còn gần khi
recurrentweights đã học? Creditnorm và hướng của own/stop/e-prop khác ra sao?
Các checkpoint/method khác nhau có Wout/fstats/weights khác nhau nên đây là
within-checkpoint estimator-vsoracle, không phải matched-weights causalcomparison
giữa năm methods. Kết quả không được dùng cứu ACR hoặc thay protocol ACR.
