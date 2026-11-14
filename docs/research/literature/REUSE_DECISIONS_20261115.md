# Literature-first intake — quyết định kế thừa, 15/11/2026

**Không mở thí nghiệm mới trong lượt này.** Quy tắc vận hành nằm duy nhất trong
[master plan §0](../../RESEARCH_PLAN.md#0-literature-first-bắt-buộc-trước-nghiên-cứu-mới).
Tài liệu này ghi nguồn đã tra, điều biết được và quyết định hiện tại. Paper và
README là bằng chứng về cơ chế/claim của tác giả; chúng không là bằng chứng
SCAR-X đã đạt hiệu năng đó. Không cài framework hoặc nhập trọng số/code upstream.

Đã đọc paper/metadata ở arXiv, PMLR và nguồn tác giả; đọc README, phần code liên
quan qua GitHub connector. Nature chuyển qua trang cookie nên không đọc được
toàn bài bằng công cụ ở lượt này; không giả là đã đọc full text. Hồ sơ paper/
license/source đã có từ12/11 được giữ và tái sử dụng. Một số repo mới chỉ đọc
README cùng paper, không gọi đó là kiểm toán toàn implementation.

## Đối chiếu theo nút thắt

| Nguồn chính | Điều có thể kế thừa | Giới hạn đối với SCAR-X và quyết định |
|---|---|---|
| [e-prop, repo tác giả](https://github.com/IGITUGraz/eligibility_propagation), [paper](https://www.nature.com/articles/s41467-020-17236-y) | Eligibility traces và learning signals; repo tách kiểm chứng factorization/hardcoded khỏi bài đo chất lượng. Đã xem ALIF source và README hiện tại. | **REUSE** hồ sơ/phương trình/kiểm số đã có. Signal/optimizer/forward SCAR-X khác nên không nhập điểm paper. Không chạy lại toàn suite chỉ để chứng minh trace tồn tại. |
| [BrainTrace](https://github.com/chaobrain/braintrace), [pp-prop source](https://github.com/chaobrain/braintrace/blob/fafa25753b743a2e2d5b47209c4ee947abc02cfb/braintrace/_algorithm/pp_prop.py), [SNN tutorial](https://brainx.chaobrain.com/braintrace/tutorials/snn_online_learning.html) | D-RTRL và trace factorization pp-prop có giả định, VJP window và decay cần công bố. Code online không tự đồng nghĩa mọi đường tín hiệu đều địa phương. | **REUSE / DEFER**: upstream hiện vẫn commit `fafa257…` đã pin12/11; D-RTRL blob cũng khớp. Tái dùng kết quả cũ; chưa nhập compiler/JAX hoặc đổi rule. pp-prop chưa là giải pháp đã chứng minh cho hidden skill/retention của SCAR-X. |
| [DECOLLE paper](https://arxiv.org/abs/1811.10766), [code neuron](https://github.com/nmi-lab/decolle-public/blob/master/decolle/base_model.py) | Local auxiliary losses và tách state qua thời gian là hướng tham khảo cho representation. | **REFERENCE / ADAPT có điều kiện**. Đoạn code đã đọc có nhánh dynamics cũ giữ tương thích; không copy mù. PyTorch/local surrogate implementation không phải ngoại lệ cho backward toàn mạng trong core. Không tuyên bố đã tái lập DECOLLE. |
| [Synaptic Intelligence](https://proceedings.mlr.press/v70/zenke17a.html), [optimizer tác giả](https://github.com/ganguli-lab/pathint/blob/86c5e07c5603d6a44c2f53585c19ee70773ef15c/pathint/optimizers.py#L105) | Importance theo đường đi cập nhật và penalty giữ kỹ năng. | **REUSE ý tưởng, không copy training rule**. Code đọc được dùng gradient loss; biến thể local sẽ là thuật toán khác cần kiểm riêng. Không coi importance lớn hoặc forgetting thấp khi không học được gì là thành công. |
| [Reflexion paper](https://arxiv.org/abs/2303.11366), [repo](https://github.com/noahshinn/reflexion) | Feedback và reflective text trong episodic memory có thể cải thiện hành vi mà không đổi trọng số LLM. | **BASELINE bắt buộc**: cùng VLM/tool/evidence/budget. Reflection là đề xuất, không oracle sự thật; không quy gain do nhớ văn bản cho synapse. README yêu cầu API không phải cấu hình được phép tự đưa vào runtime local. |
| [ExpeL paper](https://arxiv.org/abs/2308.10144), [repo tác giả](https://github.com/LeapLabTHU/ExpeL) | Gom trải nghiệm, rút insight và truy xuất khi suy luận mà không cần parametric update. | **REUSE pattern / BASELINE**. So sánh với memory/experience tốt, không chỉ bare LLM hoặc BM25 yếu. Không tự chạy lại ALFWorld/WebShop hoặc sao điểm model lớn sang Qwen0.8B. |
| [Self-RAG paper](https://arxiv.org/abs/2310.11511), [repo](https://github.com/AkariAsai/self-rag) | Retrieve/critique được học bằng reflection tokens và training LM. | **REFERENCE / DEFER**: không có sẵn trong frozen VLM chỉ vì thêm prompt; chưa phù hợp nhánh hiện không train VLM/LoRA tự động. Không tải model7B/13B hoặc dùng nó thay core ALIF. |
| [Graphiti](https://github.com/getzep/graphiti), [temporal edge code](https://github.com/getzep/graphiti/blob/main/graphiti_core/edges.py#L263) | Episode provenance; phân biệt observation/reference time với thời gian fact có hiệu lực. | **REUSE data-model concepts** cho memory; không coi graph memory là neural skill. Chưa thêm Neo4j/FalkorDB/embeddings/service mới. Raw episode/provenance cũng không mặc định là sự thật. |

## Những phép đo không cần chạy lại ngay

- [Reference132 cells](../LITERATURE_REFERENCE_RESULTS_20261112.md) và
  [đối chiếu ACR](../ACR_VS_LITERATURE_20261112.md) đã có source/init/data audit,
  kết luận mixed. BrainTrace main hiện pin `fafa25753b743a2e2d5b47209c4ee947abc02cfb`,
  commit date07/11, trùng snapshot12/11; blob D-RTRL `0b1e2111dc1e6aac50eebdc236d1c58ac096200e`
  cũng khớp cache cũ. Không có lý do rerun toàn bộ chỉ vì xem lại cùng repo.
- [Current SNN–RAG DEV](../../hybrid/RAG_SNN_AND_VRAM_20261114.md) chưa có hidden
  advantage. Bộ procedural có strong-rule ceiling không dùng làm chỗ chạy thêm
  epoch để “thắng”; lỗi ngân sách gốc vẫn giữ nguyên.
- [Recovery bank hữu hạn](../../hybrid/STATUS_CONTRACT_AND_PROTECTION_P0_20261114.md):
  trên24ca đã ghi, control11/24 và oracle12/24 chỉ cách4,17pp, dưới gate5pp.
  Không train selector trên chính bank/action contract đó để cố đạt gate.
- [Information-path audit](../../hybrid/INCIDENT_INFORMATION_AND_BENCHMARK_AUDIT_20261114.md)
  đã có ba probe khác nhau đi vào cùng input. Kiểm lại source/artifact có thể
  đủ; không cần học nhiều seed để xác nhận phần thông tin đã bị loại bỏ.
- Đọc checkpoint P0 đã có cho thấy sáu mảng H128×72 FP32 (`e_v/e_a/e_f`,
  trace, energy, importance) tổng **221.184 byte**. Đây chỉ là cận trên nếu bỏ
  được cả sáu mảng, **không phải mức tiết kiệm pp-prop đạt được**. Tối ưu nhóm
  buffer này không giải quyết vài GB VRAM còn trống; chưa phải ưu tiên memory.

## Khoảng trống nào mới đáng kiểm chứng?

1. Chỉ rõ đường tác động: SNN chọn nguồn/action nào, executor nào thực thi,
   outcome nào được kiểm độc lập. Foreground web route bằng runtime không được
   tính là neural policy đã học. Không có causal path thì chưa cần benchmark dài.
2. Input phải phân biệt trạng thái mà task cần; action/verifier phải có khả năng
   tạo kết quả hữu ích. Khóa source, tránh future/gold leakage, kiểm alias và cận
   baseline/oracle trước khi cập nhật synapse.
3. Sau đó mới xét một giả thuyết local credit/representation/retention cụ thể,
   matched rule/RAG/experience/frozen/readout/hidden và ablation cần thiết.
   Giữ acquisition, gain≥5pp/lowerCI>0, mean forgetting≤3/worst≤5pp. Không tăng
   quy mô hoặc lấy replay/retrieval tốt hơn làm chứng cứ hidden learning.
4. Một minimal compatibility/parity check cho phần thật sự chuyển sang ALIF có
   thể cần thiết. Nó không tái lập nguyên benchmark paper, không tự mở DEV và
   không sửa [history P0 FAIL](../../hybrid/HISTORY_P0_RESULT_20261115.md).

Mẫu intake cho protocol tương lai: **vấn đề → paper/code@ref → kết quả đã biết
và artifact local → khác biệt SCAR-X → REUSE/ADAPT/VERIFY_GAP/DEFER/DROP_DUPLICATE
→ giả thuyết còn lại → minimal budget/đối chứng/điểm dừng → authorization**.

Đối với bản sửa nguồn web/review đang hoàn tất: đây là kiểm hồi quy đường dữ liệu
của app, không thử một learner mới. Provenance/task binding phải đúng trước khi
bất kỳ cơ chế học nào nhận trải nghiệm đó; không lấy một câu self-critique của
VLM làm nhãn đúng. Không có nghiên cứu SNN-vs-RAG mới được khởi chạy trong lượt này.
