# Phase 2.1 — đăng ký trước khi chạy

Ngày 2026-11-08. PIVOT cơ chế học, giữ H=48, fan-in=8, plastic rows=12. Không tối ưu native/iGPU, không tải model/corpus. Cấu hình: configs/phase21.json. Kế hoạch này và source snapshot đi cùng mỗi run, không sửa ngưỡng theo kết quả.

## Giả thuyết và audit

H1: shared readout/interference là nguồn forgetting đáng kể. Audit toàn bộ15 trial Phase2, tính diagonal acquisition, final, max-to-final forgetting và signed after-learning-to-final; kiểm hash/checkpoint. Re-run3 seed cũ, thay head/hidden ở cuối bằng snapshot sau context đầu để đo counterfactual task0. Kiểm replay coverage theo(context,label), hoạt động spike và feature correlation. Đây là chẩn đoán cơ chế, không phải test xác nhận mới.

H2: STDP cũ không đủ credit assignment. Thử local score eligibility tại neuron stochastic ALIF: presynaptic voltage traces ×(spike-probability)/temperature, tích lũy qua đoạn, nhân scalar log-likelihood advantage ở cuối. No backward through network/time. Có clipping và block rotation nên không tuyên bố estimator không chệch của toàn thuật toán.

H3: metaplasticity giảm interference nhưng có thể ngăn học. Accumulate importance từ magnitude local update, giảm learning rate tại synapse quan trọng, anchor về stable weights. Đây là heuristic mới lấy cảm hứng complex synapses, không phải tái lập Synaptic Intelligence sử dụng gradient.

H4: replay cân bằng context/label + consolidation cải thiện retention với budget cố định. Phải báo acquisition và final accuracy; forgetting thấp do không học không được tính thành công.

## Bài representation/temporal

Ba bài binary,12 bước,8 input, cue noise/jitter: XOR hai thuộc tính; thứ tự A/B với cùng bag-of-events và ký hiệu cuối giống nhau; binding hai key/value rồi query key. Output class cân bằng. Split train/probe/test/OOD bằng seed riêng; OOD tăng khoảng cách thời gian trong cùng12 bước. Input không chứa nhãn; noise mới chưa đủ để gọi semantic generalization, chỉ là synthetic feature/temporal relations.

Học512 episode/condition. Sau đó freeze hidden, huấn luyện readout ridge mới từ128 episode disjoint và đo256 test. Probe learned vs initial hidden dùng cùng dữ liệu/ridge=0.01. Đánh giá thêm end-to-end, temporal permutation/reset-memory lesions và delayed OOD. Probe chỉ là phép đo representation, không nhập trở lại model.

Development: seed71,73,79 ×4 rules ×3 tasks. Chọn rule hidden (stdp/score/score_meta) theo mean probe gain so với initial; tie chọn thứ tự stdp,score,score_meta. Readout là matched forward control, không được chọn làm hidden winner. Chỉ chọn một lần.

Confirmation:8 seed mới, selected rule vs readout-only và selected rule với scalar modulation bị đảo dấu độc lập từng episode (negative control). Mọi stochastic condition dùng cùng initial weights, noise schedule và readout training. Evaluate labels không tham gia cập nhật hidden hoặc chọn rule. Không tune lại sau mở test.

## Replay/consolidation

Giữ stress20 context Phase2, mỗi context permutation riêng,64 fresh+32 replay slots/stage. No-replay dùng32 fresh repeats để cân bằng update count. Buffer luôn được cấp phát128 padded samples, cùng dtype/dung lượng; không có extra memory bank theo task.

Development:3 seeds ×3 replay policies(none,reservoir,context/label-balanced) ×3 consolidation(none,EMA,meta), hidden dùng score rule cố định để tách yếu tố. Same96 updates/stage, khởi tạo và dữ liệu. Chọn cấu hình có acquisition không thấp hơn reservoir+EMA quá3pp trên development; trong số đủ điều kiện, ưu tiên final accuracy rồi forgetting rồi thời gian. Nếu không ai đạt, giữ baseline và báo không có winner hợp lệ.

Confirmation:8 seed mới ×selected replay/consolidation, reservoir+EMA reference, selected với hidden frozen, selected với consolidation tắt. Nếu condition trùng nhau vẫn ghi riêng như identity check. Test64/context, cùng samples suốt20 chặng. Báo full accuracy matrix; forgetting max-to-final và signed BWT; acquisition mean diagonal. Không rollback trên test.

Budget training được đối chiếu ở hai chế độ: (a) cùng96 cập nhật/stage; (b) cùng thời gian train/stage lấy median thời gian mỗi stage của reference development (cố định trước confirmation), với tối đa96 cập nhật. Không tính evaluation/probe vào train time. Báo actual updates/time và chênh lệch; không gọi equal-updates là equal-wall-time. Candidate và reference chạy tuần tự, không tranh CPU giữa trials.

## Tiêu chí giữ cố định

- Học mới: gain>=10pp vs frozen, báo riêng; không đủ để gọi representation thành công.
- Representation: probe gain learned-vs-initial có cận dướiCI95%>=5pp trên ít nhất2/3 tasks; end-to-end accuracy>=65%; lợi ích selected-vs-readout dương. Holm-adjusted paired sign-flip p<=.05 cho3 primary task probe contrasts. Temporal lesions/OOD hỗ trợ diễn giải, không thay primary criteria.
- Retention: mean forgetting<=3pp, worst<=5pp cho mọi confirmation seed; mean diagonal acquisition>=65%, final không dưới acquisition quá3pp. Threshold cũ không bị nới.
- Cải thiện tương đối: giảm mean forgetting ít nhất5pp (cận dướiCI95%), acquisition noninferiority margin3pp. Kết luận tốt hơn baseline cần cả hai, kiểm ở update/time budgets; không tính thành đạt ngưỡng tuyệt đối.
- Bootstrap theo seed độc lập10000 resamples; sign-flip exact với8 seeds; không bootstrap từng noisy example như seed độc lập. Tất cả contrast raw được công bố, primary endpoints xác định trên đây.
- Save/reload model, importance, traces, active block, reward baseline và RNG phải tái lập bước học tiếp theo. Replay buffer/RNG cũng lưu được.
- RAM<=2e9; preallocated nhỏ, monitor RSS/commit; không GPU. Dừng nếu quá budget, NaN, lỗi persistence hoặc wallclock>3h cho một run. 8h stress không thuộc vòng này.

## Quyết định và tài nguyên

CPU NumPy, một process nghiên cứu; RAM dự kiến<1 GB commit cho mạng/buffer nhỏ, trần2 GB. Thời gian dự kiến5–30 phút tuỳ audit/evaluation, hard experiment deadline3h; số đo thực tế quyết định báo cáo. Không tăng số neuron/tham số.

GO chỉ cho vòng nghiên cứu nhỏ kế tiếp nếu representation và retention tuyệt đối đều đạt trên unseen seeds và không fail integrity. PIVOT nếu còn tín hiệu học nhưng ít nhất một gate chưa đạt. NO-GO cho rule/cấu hình nếu không có lợi ích đáng tin và không học được; không suy ra mọi SNN bất khả thi. Không mở language training ngay cả khi gate vòng này đạt.

Nguồn cơ chế: [Three-factor review](https://pmc.ncbi.nlm.nih.gov/articles/PMC4717313/), [reward-based learning](https://neuronaldynamics.epfl.ch/online/Ch19.S4.html), [Synaptic Intelligence](https://arxiv.org/abs/1703.04200), [experience replay](https://arxiv.org/abs/1811.11682). Đây là cơ sở thiết kế, không phải chứng nhận kết quả SCAR-X.
