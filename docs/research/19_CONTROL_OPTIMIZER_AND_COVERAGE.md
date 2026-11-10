# P1: tách optimizer và độ phủ synapse trong đối chứng

Ngày2026-11-09. **DIAGNOSTIC_ONLY**. Đây không là một local candidate, không mở
lại family local-classifier đã NO-GO và không đưa BPTT vào lõi chính.

## Vì sao phải kiểm?

[Báo cáo18](18_LOCAL_CLASSIFIER_FINAL_AND_LEARNABILITY.md) cho thấy cùng ALIF nhỏ
có thể học binding bằng BPTT tốt hơn local rule. Nhưng control đồng thời đổi
tín hiệu credit, optimizer và độ phủ cập nhật hidden; không thể quy mọi chênh
lệch cho một yếu tố. Audit này giữ nguyên task loss/BPTT để tách hai yếu tố sau.

[Protocol](../../experiments/p1-control-factor-audit-v1.md) khóa trước:3freshseeds,
3tasks,5conditions,2.048examples/trial, tổng45trials. Cùng initialization,
dataset/order, ALIF48/K8/R12/D8/O2 và train-only normalized main head.

Readout luôn dùng local clipped delta hiện có. Hidden control dùng Adam hoặc
SGD(LR0,002), toàn48rows hoặc12rows luân phiên. Full BPTT/globalnormclip1 được
tính trước rowmask; inactiveweights và vector moments được phục hồi chính xác.
Adam scalarstep/bias-correction clock vẫn là global, không là per-row Adam.
Các nhánh này nằm riêng trong `evaluation/controls.py`; serving không import.

## Kết quả trên mẫu chưa học

[Run gốc](../../runs/20261109T125432Z-p1-control-factor-audit-v1-c52f61/report.md)
lưu toàn bộ predictions, weights/head, losses/ΔW và paired effects.

| Bài | Hidden frozen | Adam / toàn bộ | Adam / 12rows | SGD / toàn bộ | SGD / 12rows |
|---|---:|---:|---:|---:|---:|
| Order | 95,18% | 100,00% | 100,00% | 97,79% | 96,61% |
| XOR | 87,76% | 100,00% | 99,87% | 93,75% | 89,45% |
| Strict binding | 65,10% | 82,42% | 63,28% | 61,07% | 62,76% |

Order có ít headroom; điểm100% không là bằng chứng hội thoại/suy luận đa năng.
Audit không có candidate để chọn và không mở finaltest/confirmation. Các gate
representation/forgetting của master được giữ nguyên, không coi bảng này là GO.

## Điều audit thực sự cho thấy

Trong thiết lập đã khóa, bài binding nhạy với cả optimizer và coverage:

- Giữ allrows, Adam hơn SGD21,35đ; seed-bootstrap95% thăm dò13,67…29,69đ.
- Giữ Adam, allrows hơn rotating12rows19,14đ; khoảng17,58…21,09đ.
- Giữ rotating12rows, Adam hơn SGD0,52đ; khoảng−7,42…10,16đ.
- Giữ SGD, allrows thấp hơn rotating1,69đ; khoảng−7,03…2,73đ.

Không có quy luật “Adam luôn tốt” hoặc “cập nhật nhiều luôn tốt”; có tương tác.
Order/XOR không bộc lộ coverage gap lớn của Adam, còn binding thì có. Vì vậy
không dùng bài dễ làm đại diện cho học quan hệ hoặc normalchat/vision.

Chỉ3development seeds; p sauHolm12primary contrasts đều1,0, độ phân giải exact
sign-flip quá thô để chứng nhận hiệu quả quần thể. Bootstrap từ3seed là mô tả
thăm dò, không biến thành confirmation độc lập. Nonsignificance không chứng
minh không có hiệu ứng. Không tuneLR hoặc mở thêm budget sau khi thấy kết quả.

## Không được kết luận gì?

- Không kết luận optimizer là nguyên nhân duy nhất của local-rule failure.
  Tất cả unfrozencontrols ở đây vẫn có full task credit/BPTT; chưa thử tương
  đương tín hiệu local dưới cùng effective step và coverage.
- Không kết luận mọi SGD thua Adam. Cùng numericalLR khác effective step:
  trênbinding, meanL2ΔW_in Adam-all1,420 so với SGD-all0,0636; recurrent2,078
  so với0,0779. Khoảng cách lớn về mức cập nhật là một phần của treatment.
- Không nói rotatingcontrol tiết kiệm CPU/RAM: nó vẫn tính full graph/gradient
  rồi mask. Mỗi episode có768scheduledhiddenwrites(all) hoặc192(rotating),
  còn số weights đổi ròng được log riêng, không đồng nhất với số lần ghi.
- Không nhập weights/backward vào model live hoặc mở lại local-classifier
  bằng tên optimizer mới. Family đó vẫn dừng sauhai vòng.

## Tính đúng, tài nguyên và quan sát

Tests kiểm inactiveweights/moments, frozenhidden và forward parity. Mọi trial
kiểm Torch feature bằng NumPy trênheldout; final core/head save-load giữ đúng
arrays và probabilities. Control checkpoint không cam kết optimizerresume.

Lượt mất khoảng302s, peakRSS751MB, committed khoảng1,25GB; CPU1thread,
khôngQwenđồngthời. Windowsper-processcap2GB và monitorprocess-tree giữ nguyên;
sampling không là aggregatehardcap. Live model có một core, counterupdates
giữ241347 trong toàn lượt, không nhận candidate/controlweights.

Localhost đã quan sát process đang sống,7/45và39/45trials, actualspikes,
loss/ΔW512examples/block và heldouttensorpredictions. Kết quả tiếp tục tự hiện
sau khi xong, không cần người xem bấm chạy hoặc chọn model. Qwen không được
chạy trên các tensor tests, trường thiếu được ghi đúng; bản đo dialogue cũ giữ.

## Hệ quả cho nghiên cứu tiếp theo

Theo P1, một giả thuyết local khác phải kiểm riêng chất lượng teacher/eligibility,
mức bước hiệu dụng và coverage trước khi quy mọi lỗi cho credit assignment.
Theo P2, tăng mức cập nhật còn phải chứng minh acquisition cùng retention;
đẩy ΔW lớn có thể phá kỹ năng cũ, không tự là lợi thế lifelong.

Đây là ràng buộc thiết kế cho protocol mới, không là phê duyệt vòng3 của family
đã dừng. Không nâng neuron/tham số, mở10Mtoken hoặc dùng replay/RAG thay yêu
cầu synapse học kỹ năng. Mục tiêu normalchat/vision của SCAR-X vẫn chưa đạt.
