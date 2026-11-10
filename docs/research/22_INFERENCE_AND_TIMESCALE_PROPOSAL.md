# P2: chế độ phát xung và đề xuất time constants theo neuron

Ngày 09/11/2026. **Phần A đã đo; phần B là đề xuất và diagnostic view, chưa là
neural core đã huấn luyện hoặc tính năng trong Hermes.**

**Kết quả diagnostic đã đóng:** `STOP_TIMESCALE_VIEW_SCREEN_V1`, không mở
training hoặc deployment từ view-screen này. [Run60views](../../runs/20261109T144816Z-p2-timescale-view-audit-v1-689d02/report.md)
giữ mọi checkpoint bất biến. Baseline offline-full probe47,45%; leak-spread
47,47%; all-spread47,28%; all-mean54,11%. Mean-tau control tốt hơn spread trong
các so sánh chính, nhưng vẫn dưới65%. Không có bằng chứng heterogeneity đem
lại lợi ích đủ dùng ở đây. Không tune thêm grid hoặc nhận thành quả paper làm
số đo SCAR-X. Không bác bỏ khả năng học với dynamics khác trong nghiên cứu
được đề xuất riêng; chưa tạo learner hoặc thay quy tắc score đã khóa.

Lượt mất35,8s, peakRSS78,3MB. Baseline vector/scalar khớp bitwise sau khi giữ
đúng phép làm tròn complement của scalar `1-feature_decay`; test bao gồm
decay vectors và mean-tau kiểm soát. Các nội dung đề xuất bên dưới giữ như
lịch sử thiết kế, không là danh sách tính năng đã triển khai.

## A. Bỏ nhiễu suy luận chưa cứu được endpoint

[Protocol](../../experiments/p2-inference-dynamics-audit-v1.md),
[run](../../runs/20261109T143806Z-p2-inference-dynamics-audit-v1-2c180b/report.md).
Giữ nguyên 6 checkpoint offline-joint,3seed; 72 tổ hợp temperature0/0,1/0,4,
MC1/8, prefix6/12. Không đổi weights, RNG, calibration hoặc chọn model mới.

| Offline-joint đã học | Head cũ tick12 | Refit tick6 | Refit tick12 |
|---|---:|---:|---:|
| ALIF xác định, một rollout | 27,73% | 60,42% | 50,16% |
| Temperature0,1, MC8 | 28,49% | 66,52% | 56,35% |
| Temperature0,4, MC8 | 30,49% | 54,47% | 48,11% |

Không endpoint nào đủ65%. MC8 ở temperature0 chỉ lặp cùng một trajectory,
nên diagnostic thực thi1lần và ghi đúng effective rollouts. Original head với
calibration cũ không được refit rồi giả thành đầu ra model. Prefix6 không là
adaptive halting đã học; nó chỉ có ít bước hơn trong một input nhân tạo.

Động học0,4 khớp bitwise hàm stochastic cũ;0 khớp ALIF xác định hiện có. Không
phát hiện sai khác implementation ở các reference này. Kết quả chỉ hỗ trợ
nghiên cứu tiếp về state/representation, không deployment hoặc P2pass.

## B. Đề xuất trong phạm vi LIF/ALIF hiện có

Liên kết master: P1 representation/temporal state, P2 acquisition trước
protection, §4 các cơ chế nhớ nhiều thang thời gian. Hypothesis: một quần thể
có hằng số thời gian khác nhau có thể giữ thông tin ở nhiều độ trễ tốt hơn
quần thể có một hằng số chung. Chưa coi đó là nguyên nhân đã được xác nhận.

Nguồn đối chiếu: [Perez-Nieves và cộng sự2021, trang tác giả](https://neural-reckoning.org/pub_heterogeneity.html).
Công trình nguồn dùng các cách huấn luyện khác, gồm surrogate gradients và
FORCE; không sao chép training/backward vào SCAR-X hoặc gọi đây là tái lập.

Đề xuất giữ một ALIF population, H48/K8/R12, cùng synapses/weights và input/output.
Thay hệ số decay vô hướng bằng vector cố định theo neuron cho membrane,
adaptation hoặc filtered-spike feature. Không thêm tầng neuron vật lý/model
thứ hai; đây cũng không phải ba chế độ Fast/Deep/Idle của controller.

Thay đổi tiềm năng: thêm3×48 số float32(576byte) làm metadata dynamics.
Nếu về sau huấn luyện với leak khác nhau, forward synaptic sensitivity phải
lọc theo leak của chính postsynaptic neuron; trace chung cũ không đủ. Cần
protocol/checkpoint/parity/locality riêng trước khi tạo learner đó. Hiện chỉ
simulated inference view trên immutable weights, không sửa local score rule,
không mở vòng3 của các family đã dừng.

## C. Phép kiểm diagnostic trước huấn luyện mới

So homogeneous gốc, spread riêng membrane/adaptation/feature và homogeneous
mean-tau control tương ứng, cùng combined/correlated/independent spread.
So control mean-tau để không gán lợi ích của decay chậm hơn cho heterogeneity.
Tất cả views dùng cùng storedweights,MC8,temp0,4,prefix12 và dữ liệu nguồn;
nhãn chỉ refit diagnostic head, không vào neural forward.

Nếu có dấu hiệu đủ chất lượng, đó chỉ là cơ sở cho protocol fresh-seed khác;
không nhập weights/view vào runtime hoặc thay gate P2. Nếu không, giữ thất bại
và không tìm kiếm time constants vô hạn. Goal chat/vision bình thường vẫn mở.
