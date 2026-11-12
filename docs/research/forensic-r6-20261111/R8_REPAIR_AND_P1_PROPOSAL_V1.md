# R8 — đề xuất sửa cô lập trước kiểm chứng P1

**Cập nhật:** phần model-state engineering đã làm bằng adapter riêng, giữ nguyên
learner tác giả; xem `R8_SNAPSHOT_REPAIR_HANDOFF_V1.md`. Chưa sửa learning
controls hoặc chạy P1 mới. Những mục còn lại dưới đây là đề xuất để duyệt. Base R8
`fe472b3b1257cdb264805bf79fd53e5647e3568d`, review độc lập tại
`GEMINI_R8_REVIEW_AND_REPAIRS_VI.md`. Chỉ thị V2III/VIII giữ các giới hạn: không
tự mở cơ chế/thí nghiệm ngoài phạm vi, không fresh confirmation hoặc deploy.

## A. P0 engineering trên bản sao nghiên cứu — ưu tiên đầu tiên

Đã khép phần2/3 và model array accounting của5 trong adapter dữ liệu riêng;
76test qua và cold continuation exact. Chưa coi toàn bộ A đã xong: không migrate
namespace/module tác giả, chưa tạo matched controls hoặc streaming optimizer
controller. Snapshot chứa toàn state model và caller JSON đã cung cấp; không
hứa khôi phục một app/controller chưa triển khai.

1. Đưa candidate vào namespace research/evaluation ở bản sao; một ALIF H48,
   fan8, cùng forward, không đổi serving/main. Giữ checkpoint v1 lịch sử.
2. New versioned snapshot lưu đầy đủ RNG, threshold/gamma/gate policy, config,
   hidden/head/stable/fast/slow/bias, traces, normalizer, counters và pending
   fixture cursor; BF cần persistence mới, không khai báo save/load khi chưa có.
3. Reject shape/dtype/NaN/Inf/object/duplicate/oversized headers/checksum lỗi
   trước allocation; atomic save, whole-state rollback và cold exact continuation.
   Reuse thiết kế bounded wrapper đã kiểm trong codec P0, không import live data.
4. Xây matched control: mỗi forward/trace/gating/optimizer giống candidate,
   chỉ đổi correct/zero/permuted learning signal. Kiểm zero/matched freeze bằng
   exact weights/state và cùng RNG draw count; gate-only/multiscale-only được
   đặt tên riêng, không làm thay đổi ngầm shuffled control.
5. Thu exact update counters và applied norms, cả readout/hidden; array byte
   accounting tính mọi allocation. P0 dùng tối đa8synaptic fixture updates mỗi
   case, không train corpus hoặc chấm accuracy. Không migrate codec/G1/Hermes.

**Cửa qua A:** no-update forward parity, matched trace trước modulation,
corrupt-checkpoint rejection, cold-resume/rollback exact toàn state, sampled
process tree RSS/commit<=2.000.000.000byte và per-process cap; tests không dùng
quality outcomes làm tiêu chí. P0PASS không tự duyệt B.

## B. P1 exploratory DEV — chỉ sau A và phê duyệt riêng

- Không mở confirmation. Chỉ ba seed DEV đã xem140011/140017/140023; ghi rõ reuse,
  không lấy chúng làm bằng chứng tổng quát cuối. Không sweep học suất.
- Ba target: strict_binding, xor, distractor_binding; protected order chỉ đo
  regression, không lấy ceiling100% làm bằng chứng hidden learning.
- Chốt source generator hash và latent schema trước chạy; kiểm codebook injective,
  label oracle, split/raw overlaps và timing/noise. Không dùng bài8keyR8 hiện lỗi.
- Bộ so sánh từng yếu tố: homogeneous-fast, homogeneous-slow, multiscale-ungated,
  multiscale+gate; candidate có frozen/readout-only/zero/permuted-signal matched
  controls. Giữ normalization, readout rule, LR, clip, calibration và mẫu như nhau.
  R35 lịch sử chỉ reference có nhãn khác nếu không match được source thật.
- Ngân sách mỗi task/seed:1024train,256common-refit,512ID,512OOD; calibration chỉ
  từ train, không future-task bank. Report cả equal-update và compute/time cost.
  Khóa thứ tự, RNG streams và maximumupdate trước code được chạy.
- Lưu **mọi** seed/arm metric, actual predictions+labels, trace/weight digests,
  per-update counters, source/normalizer/RNG metadata. Không để trials=[] rồi chỉ
  công bố mean. Refit evaluator không đưa labels/metadata vào neural inference.
- Đo direct, common refit, input/recurrent revert riêng, correct-zero-shuffle
  contrasts, ID/OOD, latency/RAM. Không quy routing gain cho một control đổi trace.

**Ngưỡng giữ nguyên:** representation lowerCI>=5đ trên>=2/3bài hợp lệ, direct>=65%,
positive readout-only contrast với correction đã đăng ký. Protocol chi tiết phải
khóa primary contrasts/multiplicity trước chạy; bản đề xuất này chưa là quyền
train. Không gọi DEV thành formal confirmation. Thiếu controls/raw là chưa kết
luận, không PASS. Tối đa hai vòng có protocol; không lợi ích thì STOP/PIVOT.

Giới hạn đề xuất: CPU một numerical thread, một job tại một thời điểm, mỗi job
<=120giây và toàn DEV<=30phút. Nếu timing preflight dự báo vượt, dừng lập dự toán,
không tự cắt mẫu/epoch hoặc nới budget. Chưa có số đo đáng tin để hứa thời lượng.

## C. P2 và triển khai vẫn khóa

Chỉ đề xuất sau P1: no future calibration, không task ID/boundary oracle; cascade
so EMA+anchor, replay-only và combo, đủ acquisition/final/mean/worst trên mọi
seed với matched budget. Giữ65%/65%/3đ/5đ, không thay bằng “giảm forgetting5đ”.
Không tăng neuron hoặc ngụy tạo autonomous reasoning. Product QA, unknown,
vision/screenshot và8hresource acceptance là các cửa riêng chưa được giải quyết
bởi P0/P1 này. Main merge, model live và fresh confirmation luôn cần duyệt riêng.
