# Đề xuất kiểm history của local update — cần duyệt trước triển khai/chạy

Trạng thái: chưa code learner biến thể, chưa train. [Diagnosticfc84eb](../docs/hybrid/CREDIT_DIRECTION_DIAGNOSTIC_20261114.md)
đã cho vị trí sai khác hướng; không chứng minh biến thể dưới đây học tốt hơn.
Master§4.1 yêu cầu mô tả rule/state/tín hiệu và duyệt trước chạy rule mới.

## Phạm vi giữ nguyên

Chỉ nhánh nghiên cứu; ALIF D64/H128/FP32, topology/readout/error transport/
eligibility/gamma hiện tại, encoder predicate và head relevance hai lớp.
Không Transformer thay core, backward/BPTT ở learner, VLM/LoRA training, scale,
main merge, researchcheckpoint vàoHermes hoặc sửa source lượt8h đang chạy.

Tín hiệu `d_t = signal_t × eligibility_t` và readout update giữ nguyên. Chỉ
thay bước trộn history ở hidden. Không thêm đường lỗi mới giữa neuron hoặc
qua thời gian, không dùng surrogatecontrol gradient để cập nhật synapse.

## Năm đối chứng, khóa trước kết quả

1. Frozen cùng forward/input, không học.
2. Readout-only hiện có.
3. Currenthidden: `h_t=.9h_prev+.1d_t`, `E_t=.99E_prev+.01d_t²`,
   `u_t=clip(.008h_t/(sqrt(E_t)+1e-4),-.01,.01)`.
4. Current-only: `h_t=.1d_t`; cùng energy và mọi clip/learning rate của arm3.
   Đây là rule khác, chưa được coi tốt hơn hoặc biological hơn.
5. Current-only khớp magnitude: tại state thực của arm5, tính hypothetical
   current-rule delta và current-only direction; dùng scalar không âm để norm
   hidden update sau coordinate/weightclip khớp current-rule norm trong1%.
   Tìm scalar bằng bisection cố định tối đa20bước, chặn256; không thêm target
   hoặc kết quả test. Nếu zero support/clip khiến không khớp, ghi mismatch và
   dừng attribution, không âm thầm đổi mức norm hoặc ngưỡng.

Weightclip±3 như cũ. Arm5 dùng norm hypothetical tại chính statearm5, không
copy modelstate của arm3. Các trajectory vẫn khác; magnitude matching không
là đối chứng hoàn hảo về representation/history. Giữ raw delta và norm-error.
State thêm: không có mảng ngoài shape đang có; checkpoint envelope mới phải
phân biệt thuật toán/encoder/arm, không được live loader nhận như checkpoint cũ.

## P0 bắt buộc trước DEV

Đối chiếu công thức từng bước trên mảng fixture hữu hạn; firststep giốngcurrent
vì oldtrace0; oldtrace khác0 tạo delta khác đúng thiết kế; energy/readout không
bị đổi lén. CPU/CUDA parity, có delta ở input/recurrent thật, zero-norm handling,
strict loader/schema/checksum/algorithmversion, save-load/RNG/replay/nextupdate
và rollback exact trên clone. Không dùng20stage labelbank ở P0. P0≤5phút,
≤100attemptedupdates mỗiarm/seed, tổng≤600 bao gồm tests/reruns,2seed kỹ thuật
260313/260337; giữ failedreceipt nếu fail, không reset budget qua script khác.

## DEV nếu P0 qua và người dùng duyệt cả phần này

Fresh owned bank: cùng generator SQL/Python dualverification nhưng namespace
`historyDEV-v1/` trong source RNG, làm owner/record/time/splits mới trước model
output. Không reuse/retune original032f30 finalDEV. Ba neural seed260213/260237/
260271;5arms,20stages×16new+16replay=640scored calls/cell, same schedule/replay
RNG/labels/caseorder. 200probe(50/family),400finalDEV(100/family); đây làDEV,
chưa freshconfirmationTEST. Check encoded aliases/opposite labels trước học;
có exact opposing-label aliases thì dừng data-integrity, không chọn mẫu theo điểm.

Giữ balanced labels/four predicate families. Không dùng taskboundary/ID/gold
vàoencoder, không calibration theo probe/final. Frozen không mutation nhưng
forward calls tương đương. Thời gian thực khác phải công bố, không chỉ gọi
“matched compute”. Cap120s cumulativeLEARN/cell và40phút wholeDEV; samples,
RAM/VRAM/serialization/replay/overhead đo riêng,512MiBphysicalVRAMreserve.

Gate không đổi: acquisition sau giới thiệu từngfamily và final từngfamily≥65%
ở từngseed, meanforgetting≤3pp/worst≤5pp sau20stages; candidate gain≥5pp so
đối chứng mạnh nhất và lowerpairedseed/familyCI95%>0. Hidden phải hơnreadout-
only; nếu không thì chưa chứng minh lợi thế hidden. Không chọn một seed tốt.
Bootstrap fixedseed26021499,10000resamples, source/family grouping công bố.

Hidden-reset/commonhead trên clone và shuffle label kiểm attribution chỉ khi
có acquisition signal, có budget riêng trước chạy; P0/magnitude alignment
không thay qualitygates. Candidate chỉ có screening signal, chưa promote trước
đối chứng nhân quả/retention và confirmation khác được duyệt riêng.

Nếu acquisition vẫn gầnchance hoặc jointgate fail: dừng familyhistoryscreen,
không tăng epoch/modelsize hoặc nới65/3–5/5pp. Ưu tiên encoder/binding và data
learnability; không tự thử thêm grid momentum hoặc áp vàolive.

## Đầu ra

Một run/P0 và một run/DEV duy nhất nếu được duyệt, protocol/source/datamanifests
trước output, raw predictions/curves/loss/delta/normchecks, fullstate/coldresume,
auditor độc lập đường tính và localhosttrace thật. Giữ tất cả lỗi/budgetdeviation.
Không nâng cấp khả năng ngôn ngữ/video chỉ từ bài predicate tổng hợp này.
