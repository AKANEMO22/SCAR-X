# Giới hạn không gian credit của ACR — phân tích, chưa phán quyết quality

ACR giữ một forward ALIF và hai local eligibility streams độc lập. Ở mỗi
neuron j, mỗi trace có D+fan=16 thành phần. Với một alpha_j duy nhất, trace
trộn nằm trên đoạn thẳng nối hai vector trong không gian16chiều:

```text
E_j(alpha) = E_own_j + alpha_j*(E_stop_j-E_own_j), 0<=alpha_j<=1
local_ascent_j = m_j * E_j(alpha)
```

Đây là hạn chế biểu diễn credit, không phải khẳng định model không thể học.
Một router hoàn hảo vẫn không tùy ý tạo bất cứ vector cập nhật16chiều nào
tại cùng frozen state/signal. Hai stream khác nhau có thể tạo bias hữu ích;
trộn chúng có thể cải thiện optimization hoặc generalization dù không gần
full-gradient hơn. Kết quả thực nghiệm mới quyết định ích lợi trên nhiệm vụ.

Với loss trên readout features, full surrogate ascent cho row synapse j có
dạng tổng qua output features của mọi neuron k:

```text
full_j = sum_k m_k * d(feature_k)/d(weight_row_j)
diagonal_local_j = m_j * local_sensitivity_j
```

Cross-neuron recurrent pathways k!=j bị bỏ trong cả own và stopped hiện tại.
Scalar mixing không nói chung khôi phục được mọi thành phần bị bỏ đó. Nó có
thể xấp xỉ một số hướng hoặc cải thiện representation bằng quá trình học,
nhưng không có định lý rằng state router sẽ giải temporal credit assignment.
Không dùng oracle alpha/future signal/test accuracy để điều khiển candidate.

Điểm khác với e-prop reference cũng cần phân biệt: graph resetspike-detached
trên SCAR-X adaptive threshold giữ `-z*beta*e_a_old` nhưng bỏ `-theta*z_e`.
Stopped hiện tại bỏ cả hai; own giữ cả hai. Trộn filtered traces có lịch sử
riêng không đồng nhất với giữ đúng một cạnh Jacobian của graph tham chiếu.

BrainTrace local-block adapter khớp feature filter và giữ within-neuron state
coupling. Nó vẫn cắt recurrent mixing. Oracle full-sensitivity trong nhánh
literature là control toán, không candidate và không learner sản phẩm. Những
phương pháp này đã có nền tảng trong literature; việc viết adapter NumPy không
tự tạo tính mới. So accuracy phải giữ init/data/readout/optimizer, rồi tách
actual-update norm, direction, OOD và causal representation.

Reference DEV đã cho phản ví dụ đối với suy luận “credit gần oracle => direct
skill cao”: alignment tốt không ngăn regression Binding. Chưa xác định duy
nhất lỗi ở estimator, source objective/normalizer, conditioning hay trajectory
của optimizer. ACR đang kiểm matched physical norms để tách một phần confound;
kết quả đó không tự giải quyết các phần còn lại.

Phân tích này không thay protocol/gates hoặc chọn profile theo task. Chưa cho
phép kết luận H48 đã chạm trần, literature thất bại, ACR thành công hay không
cần BPTT cho mọi kỹ năng. Evidence hiện tại chỉ áp dụng các bộ event tasks và
numerical controls đã công bố. Retention/continualP2 còn là một phép đo riêng.
