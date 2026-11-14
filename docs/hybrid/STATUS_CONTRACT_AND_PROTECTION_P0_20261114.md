# Mã trạng thái, giới hạn bảng recovery và prototype bảo vệ synapse

## Không huấn luyện trên một bảng không thể đạt mục tiêu đã khóa

Chỉ giải thích đầy đủ mã outputstatus làm2/16nhánh qua rubric yếu trước đó0/16,
trên đúng8ownedDEVcases×2actions;16nativecalls/25,016s. Inputdiff đã kiểm chỉ
append contract, actualstate/RNG/nguồn/model/schema/budget giữ nguyên.
Không sửa original196043 hoặc biến đây thành SNNlearning.

[Protocol](../../experiments/hybrid-recovery-status-contract-diagnostic-v1.md),
[rawbad3a2](../../runs/20261114T030616Z-recovery-status-diagnostic-v1-bad3a2/result.json),
[independentstate/inputaudit](../../runs/20261114T030616Z-recovery-status-diagnostic-v1-bad3a2/independent-audit-v1.json).
Cả2nhánh qua rubric vẫn xin credentials trong message; không dùng làm nhãn
“an toàn/đúng mọi nghĩa”. Các nhánh khác có statuskhác, lặp deniedcall, hoặc
message hỏi nhưng không qua phrasecheck; số2không là semanticaccuracy.

Trên original24ca196043, prototypecontrol dùng đúng báo cáo lỗi đầu tiên:
exception≠None→show_schema; không lỗi→finish. Nó qua11/24ca; oracle được quyền
chọn bất kỳ action **đã đo** nào cũng chỉ qua12/24. Margin tối đa4,17pp,
không đủ gate5pp đã khóa. Không có lý do chạy thêm epoch để ép SNN “thắng”
trên bảng này.

[Finitebound audit](../../runs/20261114T025511Z-recovery-outcome-bank-dev-v1-196043/finite-bank-bound-v1.json)
giữ percase/control/actionoutcomes. Control được xây sau khi xem dữ liệu,
**không phải baselineđăngkýtrước trên heldoutTEST**; cận chỉ mô tả chính bảng
action/budget này, không chứng minh SNNkhôngthểhọc hoặc mọiRAGtốt hơn.
Không tạo nhãn mới hoặc thay ngưỡng để hợp thức hóa một thử nghiệm.

## Prototype cô lập nối importance/anchor vào cập nhật thật

Audit trước đã chỉ ra currentHybrid importance/stable_* chưa tham gia learn.
PrototypeP0 mới ở evaluation dùng lại heuristic magnitude/anchor từ
PlasticCore lịch sử, áp lên actualdelta sau localcredit đang có; không thay
ALIF, recurrence, eligibility hoặc đường W_out.T×error. Thêm importance cho
readout/bias vì đây là chỗ currentHybrid chưa có protection state riêng.

[Registration công thức](../../experiments/hybrid-magnitude-protection-prototype-p0-v1.md),
[implementation](../../src/scarx/evaluation/magnitude_protection_p0.py).
Trình tự dùng omega/anchor **trước update**, attenuate/anchor/clipping, rồi
tăng magnitudeimportance; EMAstable rate0,02 mỗi64attempts. Frozen/readout-only
giữ giới hạn cập nhật của mode. ExtraFP32importance4128byte; checkpoint fixture
~683,9KB gồm replay/traces/stats/RNG/stable và metadata.

Đây là **thuật toán prototype khác**, không giả định chỉ sửa serialize:
importance giờ ảnh hưởngdelta. OuterZIP_STORED khóa format/constants/checksums,
parser chặn extra/duplicate/badtype/dtypefloat64, currentliveloader không nạp.
Không plug vào automaticruntime/consolidation, không thay liveweights/main.

P0numeric đã quan sát:

- 3 seed × 32 cập nhật số học; bước đầu với importance bằng 0 khớp base trên các
  fixture, trong toleranceFP32≤3e-7 đã khóa.
- Importanceinput/recurrent/readout/bias tăng thực, omega20 attenuate normdelta
  output xuống dưới2% ở điều kiện anchor=weightbefore.
- EMAstableout giữ nguyên tới63, thay đổi đúng64.
- ColdGPUrestore và bước negative/replay tiếp theo khớpbyte; frozen/readout-only
  không lén cập nhật hidden. Tất cả neuralnumerics trên CUDA.

[Runbacf0f](../../runs/20261114T032017Z-magnitude-protection-p0-bacf0f/result.json),
[technicalrerun7fcb49](../../runs/20261114T032234Z-magnitude-protection-p0-7fcb49/result.json).
Không có taskdataset, accuracycurve, unseen skill hoặc retentionclaim ở đây.

## Giới hạn kỹ thuật và sai lệch ngân sách

Hai recordednumericrunners dùng175updates/lượt, tổng350, chưa tính unitchecks.
Registration ghi≤300total mà không cấp riêng lượt táikiểmserialize. Theo cách
đếm tíchlũy chặt, đã vượt cap; [reviewgiữriêng](../../runs/20261114T032234Z-magnitude-protection-p0-7fcb49/aggregate-budget-review-v1.json).
Không gọi toàn campaign là protocol-compliantresearchPASS hoặc xóa lượt đầu.
Các property số học quan sát được vẫn có nguồn nhưng không biến thành kết luận
khoa học. Dừng numericreruns; kiểm hồi quy tách khỏi phép đo taskquality.

Magnitudeimportance chưa đo causalusefulness, có thể bảo vệ cả thay đổi sai.
Monotonicimportance và anchor có thể làm mất plasticity. f_mean/f_var vẫn thay
đổi nhưbase; readout/hidden drift chưa mặc định được sửa. Prototypelearn chưa
được tích hợp transactionguard/rollback của gateway, nên không thể đem vào
Hermes chỉ vì checkpoint unitcheck qua.

Muốn đánh giá năng lực phải đăng ký riêng freshDEV: baselinecurrentlearner,
uniformmagnitude-matchedlrcontrol, anchor-only, importance-only, replay/statistics
controls, readout/hidden/shuffled/reset; cùng nguồn/lượt/time/memory.
Giữ acquisition,≥5pp/CI và mean≤3/worst≤5pp20stage, cùng plasticityreserve.
Không dùng196043 vì bankbound đã biết không đủ margin.

## Trạng thái sản phẩm và goal

Model live vẫn Qwen0.8B/Whisperbase/ALIFCUDA,5updates thật trước lab; không
nhập prototype hoặc nhãn trạng thái chưa xác minh. Lượt8h554c19 vẫn phải
đối chiếu PID/source/finalreceipt, không coi design/ticks là hoàn tất.
Quyết định: **PIVOT vẫn giữ; có prototype để kiểm cơ chế, chưa có bằng chứng
hidden SNN hơn RAG hoặc continuallearning đủ tiêu chí.**

## Kiểm tra hồi quy và lỗi transport giữ lại

Một full-suite run ghi243pass/1fail do WinError10053 ở nonce HTTPfixture.
Chưa mặc định đó là lỗi neuron hay firewall. Sourceaudit thấy rejectionpath
gửi503 khi POSTbody vẫn chưa đọc; đóng socket kiểu này có thể gây reset trước
khi client nhận response. Regression mới đỏ trên original ở điều kiện unreadbody.
Đường audit nay chỉ drain Content-Length≤64KB, sockettimeout1s, không chunked/
unboundedbody và không gọi handler bên dưới cho request khôngđúngnonce.
Test11pipelinechecks qua sau sửa; không yếu nonce hoặc đổi route/capability.
Đây là sửa harness đọc/framing, không phải chứng minh nguyên nhân mọi10053.
Source8h không chứa harnessfile này và vẫn khớp hash.

Sau sửa:245Python checks,12Node checks và realHTTPreply fixture qua. BảnJUnit
failed trước sửa được giữ riêng với bản sau; không rerunmodel hoặc đổi metric
để chuyển một taskresponse từFAIL thànhPASS. Auditors là đường tính/kiểm lại
khác do cùng orchestrator viết/chạy, chưa phải peerreview khoa học bên ngoài.
