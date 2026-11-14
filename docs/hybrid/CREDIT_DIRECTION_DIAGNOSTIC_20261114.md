# Credit tới hidden còn tồn tại; lịch sử cập nhật cần được kiểm riêng

**Chưa đổi learning rule hoặc model live.** Kiểm toán mới cho thấy trên các
mẫu đã chọn, local credit có cùng hướng đáng kể với đối chứng surrogate qua
thời gian. Nhưng ở checkpoint cuối, momentum và tiền điều kiện làm hướng cập
nhật thực tế khác nhiều hơn so với gradient của riêng mẫu đang xét. Đây là
một giả thuyết cụ thể cần ablation, không phải kết luận đã tìm ra nguyên nhân
duy nhất của quên hoặc bỏ momentum sẽ làm model giỏi lên.

## Phép đo thực

[Protocol](../../experiments/hybrid-credit-direction-diagnostic-v1.md),
[raw fc84eb](../../runs/20261114T064319Z-credit-direction-diagnostic-v1-fc84eb/result.json),
[phân rã data-only](../../runs/20261114T064319Z-credit-direction-diagnostic-v1-fc84eb/history-decomposition-v1/independent-data-audit.json).

Tám mẫu train cũ: dương/âm đầu tiên của mỗi nhóm predicate. Ba seed; hai trạng
thái initial và currenthidden640update. Tổng48case, không dùng thêm mẫu để học,
không chọn theo output hoặc dùng finalDEV để fit. Mẫu và seed không phải48lần
kiểm chất lượng độc lập. Các gate65%/3–5pp/5pp trước đây giữ nguyên.

Actual LocalALIFCUDA tạo96rollout có tracking/spike tape. Đối chứng riêng dùng
TorchCUDA tính48backwardpasses qua thời gian, logistic surrogate khớp với slope
đang dùng. **Backward chỉ nằm trong `evaluation/controls.py`, không ở SNN chính.**
Không optimizer step, thay trọng số, normalization fit hoặc replay. Forward
dùng hard-spike tape quan sát được; state sai lệch tối đa1,91e-6 trong ngưỡng
đã khóa, probability/loss cũng khớp. Surrogate không phải đạo hàm thật của hàm
phát xung gián đoạn; không gọi nó là chân lý tối ưu hóa toàn hệ.

Việc tách kiểm tra số học khỏi benchmark học cũng xuất hiện trong
[repo e-prop chính thức của tác giả](https://github.com/IGITUGraz/eligibility_propagation/blob/master/README.md):
họ cung cấp các script đối chiếu factorization/BPTT và hardcoded/autodiff riêng.
Phép đo SCAR-X này không tái lập các định lý hoặc điểm số của công trình đó;
gamma, momentum và normalization của SCAR-X vẫn là heuristic riêng cần kiểm.

## Hướng cập nhật khác ở đâu?

Cosine trung vị so với surrogate direction;1 nghĩa cùng hướng,0 trực giao,
âm nghĩa ngược hướng theo thước đo này. **Đây không phải accuracy.**

| Thành phần | Initial,24case | Final640,24case |
|---|---:|---:|
| Local direction `signal × eligibility` |0,867|0,724|
| Sau trộn lịch sử `0,9trace_cũ + 0,1direction` |0,867|0,204|
| Delta thực tế sau energy/clip |0,848|0,139|
| Chỉ direction hiện tại, cùng energy hiện có, chẩn đoán |0,848|0,554|

Ở final640,4/24accumulated-trace direction và6/24actual delta có cosine âm.
Phân rã tái tạo delta hiện hành từ trace/energy của parentcheckpoint và vector
đã lưu, sai số trong2e-7. Không forward hoặc backward thêm trong audit này.
Clip coordinate trung vị0%, thay đổi bởi weight-bound cũng0 ở48case đo được;
chưa có căn cứ quy sự khác hướng ở các case này cho chạm trần±3 hoặc clip rộng.

Counterfactual bỏ tử số lịch sử giữ nguyên energy denominator giúp tách ảnh
hưởng số học của history. Nó chưa là một learner khác: nếu thay rule thật,
mọi trọng số/energy và dữ liệu nhận về sau sẽ khác. Không triển khai biến thể
này hoặc lấy cosine tăng làm bằng chứng retention/transfer tốt lên.

Tín hiệu surrogate đối với8tick query đầu vẫn khác0: tỷ lệ norm query/record
trung vị khoảng0,573initial và0,988final. Không thể kết luận đơn giản rằng core
đã mất toàn bộ đường credit từ query. Norm không chứng minh nó học đúng binding
hay áp dụng được cho dữ liệu mới; [common-head/counterfactual results](HIDDEN_BINDING_AND_COMMON_PROBE_20261114.md)
vẫn chưa đạt lợi thế tổng quát đáng tin cậy.

## Vì sao chưa gọi momentum là lỗi?

Gradient của một mẫu không đồng nghĩa hướng tốt nhất cho mọi kỹ năng cũ và mới.
Momentum có thể chủ ý duy trì tín hiệu từ trải nghiệm khác; chuẩn hóa từng
synapse cũng làm đổi góc vector. Mẫu ở đây là các trải nghiệm cũ, không phải
toàn phân bố hiện tại. Chưa đo đối chứng bỏ history trong học20chặng, khớp độ
lớn update, acquisition, retention và hidden-reset. Do đó chỉ xác định được
**history/preconditioning là vị trí phát sinh phần lớn sai khác hướng trong
các mảng đo**, chưa xác định tác động nhân quả của nó lên điểm chất lượng.

Ưu tiên tiếp theo là ablation có protocol mới: current rule, current-only và
đối chứng khớp magnitude, cùng encoder/data/replay và readout-only. Không tăng
quy mô, kéo epoch hoặc thay các threshold cũ để tạo PASS. Encoder/binding vẫn
phải được kiểm riêng; sửa credit một mình chưa đảm bảo representation tốt.
[Đề xuất ablation cụ thể, chưa chạy](../../experiments/hybrid-update-history-ablation-proposal-v1.md)
giữ5arms/3seed và gate cũ; cần duyệt rule mới theo master§4.1 trước triển khai.

## Integrity và tài nguyên

Giữ hash nguồn, source-byte bundle, parentcheckpoint,48local/tape/controlarrays
và mọi score. Auditor khác đường tính lại prospective delta/cosine/sign/norm;
đối chiếu weights/indices/normalization với checkpoint. Đây là cùng orchestrator,
chưa peerreview bên ngoài; auditor không tự tái lập backward calculus hoặc
eligibility từ đầu. Initialtrace0/energy1e-4 theo constructor được ghi rõ.

Toàn lượt12,438s; child control1,625s,0optimizer step. Peak RSS process-tree
nghiên cứu~1,737GB, privatecommit~2,462GB; không phải RAM toànHermes/VLM. GPUmode
không caphostRAM, giữ nativeprecision và kiểm reserve vật lý. PDH lấy mẫu có thể
bỏ sót đỉnh ngắn. Không dùng các con số này làm benchmark chi phí training.

Unit control cùng forward và source-state bất biến đã qua. VLM39372, ALIF live
và lượt8h554c19 không thay bởi diagnostic. Các số đo hiện ở localhost với actor
orchestrator, tách khỏi bộ đếm cập nhật được chấp nhận của model đang phục vụ.
11test đối chứng liên quan và17test Node giao diện qua; không xem số test là
thêm seed hoặc tăng độ chắc chắn của kết luận học kỹ năng.
