# 07 — Giới hạn của Lifelong Adaptive Intelligence

Định hướng đã bỏ mục tiêu1M context. Câu hỏi trung tâm: tiếp tục học và giữ kỹ năng hữu ích trong capacity/RAM cố định, không phải giữ toàn bộ lịch sử.

## Có thể học tiếp mà không huấn luyện lại toàn mạng?

Có thể cập nhật một phần synapse, dùng trace tại chỗ, consolidation và replay nhỏ. Prototype đã thực hiện kiểu update đó. Tuy nhiên, khả năng cập nhật weights không đồng nghĩa credit assignment đủ tốt hoặc representation mới đủ mạnh. Cần kiểm tra ba việc độc lập: học được điều mới, giữ điều cũ, còn khả năng học sau thời gian dài.

Local three-factor learning có nền tảng lý thuyết và thực nghiệm riêng: [Frémaux/Gerstner](https://pmc.ncbi.nlm.nih.gov/articles/PMC4717313/), [reward-based learning](https://neuronaldynamics.epfl.ch/online/Ch19.S4.html). SCAR-X dùng biến thể heuristic nhỏ, không được lấy thành công của paper làm kết quả của mình.

## Capacity hữu hạn

N synapse với b bit/weight và state phụ hữu hạn chỉ có số trạng thái hữu hạn. Không thể cam kết giữ chính xác số kỹ năng/ánh xạ độc lập tăng vô hạn. Task có cấu trúc chung có thể chia sẻ representation; task ngẫu nhiên hoặc mâu thuẫn không có lợi thế nén đó. Muốn học dài hạn cần ưu tiên, quên có chủ đích, tái sử dụng hoặc đưa một phần tri thức ra storage. Storage bên ngoài không giải quyết tự động việc chọn đúng bằng chứng hay học representation.

Consolidation mạnh bảo vệ quá khứ nhưng giảm số bậc tự do để học mới. Synapse importance chỉ tăng sẽ dần đóng băng mạng; importance decay trả lại plasticity nhưng có nguy cơ quên. Không có một hệ số được bảo đảm tốt cho mọi phân phối thay đổi.

## Credit assignment

Reward scalar báo kết quả cuối cùng nhưng ít thông tin về neuron/kết nối nào gây ra lỗi. Eligibility kéo dài giúp nối hành động và phản hồi, đồng thời tăng nhiễu và interference giữa trải nghiệm. Block-local update tiết kiệm RAM nhưng giảm số synapse nhận credit mỗi episode. Clipping kiểm soát độ trôi nhưng làm thay đổi estimator.

No BPTT là ràng buộc thuật toán. Có thể truyền tín hiệu học bằng cơ chế khác, nhưng phải công bố đường đi của nó. Một projection của toàn bộ error về mọi neuron không được âm thầm đổi tên thành thuần synapse-local. Phase2.1 chỉ dùng scalar modulation tại hidden; SI control chỉ có error tại readout.

## Representation và capacity có ích

Reservoir ngẫu nhiên có thể tạo feature đủ cho bài đơn giản. Readout học tốt không chứng minh hidden đã học. Cần freeze hidden, reset/refit head bằng dữ liệu độc lập, so với initial hidden, đảo modulation và phá temporal state. Nếu XOR hay quan hệ khó vẫn không tách được, replay nhiều hơn chỉ ôn lại một representation thiếu thông tin.

## Forgetting khác loss of plasticity

Forgetting: kỹ năng cũ giảm. Loss of plasticity: mạng mất khả năng học điều mới, kể cả có đủ dữ liệu. [Dohare et al.](https://www.nature.com/articles/s41586-024-07711-7) nghiên cứu vấn đề thứ hai và continual backpropagation với thay một phần neuron kém hữu ích. Kết quả paper không chứng minh SCAR-X có thể học vô hạn; kiểm thử finite stream và probe cuối stream chỉ là bằng chứng hữu hạn.

## Importance và selective learning

[Synaptic Intelligence](https://arxiv.org/abs/1703.04200) ước lượng vai trò tham số từ quỹ đạo học. SCAR-X có hai đối tượng khác nhau: importance heuristic từ độ lớn tín hiệu cục bộ, và control SI tại readout. Activity lớn không đảm bảo hữu ích; học từ nhãn sai có thể làm một synapse sai trở nên rất “quan trọng”.

[Metaplasticity trong BNN](https://www.nature.com/articles/s41467-021-22768-y) và [MESU](https://arxiv.org/abs/2504.13569) đưa ra cách điều chỉnh khả năng thay đổi synapse. Chưa có tái lập matched benchmark của những hệ đó trong repository, nên chưa kết luận vượt chúng.

Không thể luôn phân biệt phản hồi đúng/sai chỉ từ confidence của model. Thông tin nguồn, kiểm chứng bên ngoài và sự nhất quán nhiều lần là dữ liệu thêm. Trong stream thí nghiệm, reliability metadata chỉ tương quan với chất lượng, có false positive/negative; selective learning được so với random rejection. Mọi lợi ích phải được phân tích theo nguồn thông tin này.

## Mức độ so sánh được phép

Các control chạy trên cùng48neuron và bounded replay giúp kiểm chứng cơ chế ở mức prototype. Control SI chỉ áp dụng output; control BPTT có stochasticity và compute khác phải ghi rõ. Chưa có evidence để quảng bá hơn continual-learning SOTA, khả năng general language hay trí tuệ như người. Quyết định phải dựa trên báo cáo Phase2.1, không dựa vào sự giống sinh học.
