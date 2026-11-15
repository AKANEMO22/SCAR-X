# SCAR-X — Kiểm toán P1 và thực nghiệm chuyển kỹ năng S+P → S

Bạn là Codex, phụ trách nghiên cứu độc lập, triển khai thí nghiệm cô lập và kiểm toán kết quả cho repository:

`AKANEMO22/SCAR-X`

## MỤC TIÊU CUỐI

Tìm bằng chứng thực nghiệm xem một mạng ALIF nhỏ có thể:

1. Học kỹ năng mới trong vùng Plastic (P).
2. Truyền kỹ năng hữu ích từ teacher S+P sang student S bằng local eligibility.
3. Giữ kỹ năng cũ và mới khi student không còn truy cập P.
4. Reset và tái sử dụng P để học kỹ năng tiếp theo.
5. Thực hiện những việc đó trong tài nguyên cố định, không tăng neuron, head hoặc archive theo từng kỹ năng.

**Không được tuyên bố đã giải lifelong learning chỉ vì một bước transfer thành công.**

Kiến trúc hiện tại là SCOPE-ALIF, một adaptation của Progress & Compress, không phải kiến trúc mới đã được chứng minh.

Mục tiêu phiên này là tìm một kết quả **có khả năng xác nhận hoặc bác bỏ cơ chế truyền kỹ năng**, không phải cố tạo kết quả PASS.

---

## PHẦN A — BẮT BUỘC ĐỌC VÀ KIỂM TOÁN KẾT QUẢ CODEX TRƯỚC

### A1. Xác minh đúng repository và commit local

Worktree dự kiến:

`C:\Users\hachimi\Downloads\model agent\scar-x\local\worktrees\scarx-scope-alif-design-20261115`

Commit local:

`6e08a361`

Base nghiên cứu trước đó:

`df80b1b5fb9b19acc96def0a26f4e3f7d1bb6353`

Kiểm tra thực tế:

- `git status`, `git log`, `git show`, SHA của commit và trạng thái worktree.
- Không giả định local commit đã push.
- Không checkout đè lên thay đổi chưa commit.
- Không ghi vào `main`, Hermes hay bất kỳ checkpoint live nào.

### A2. Đọc nguyên văn ba artifact P1

**Báo cáo:**

`docs/research/SCOPE_P1_PREFLIGHT_RESULT_20261115.md`

**Protocol:**

`experiments/scope-p1-teacher-preflight-20261115.md`

**Raw:**

`runs/20261115T165305Z-scope-teacher-preflight-v1/result.json`

Tìm thêm runner, audit scripts và source manifests mà ba artifact tham chiếu.

Nếu thiếu artifact, phải báo chính xác file nào thiếu và dừng mọi bước cần nội dung đó. Không dựng lại dữ liệu rồi nhận là raw gốc.

### A3. Kiểm toán độc lập kết quả 144 forward

Codex trước đó báo:

- `PREFLIGHT_NO_GO`.
- 144 forward CPU.
- 0 learning updates.
- Teacher được dựng sẵn có thể giải task pulse-gap.
- Student tồn tại lời giải trong phạm vi representability.
- Readout-only có feature collision.
- Thời gian preflight/audit khoảng 0,44 giây.
- Tổng cận trên peak RSS hai worker khoảng 66 MB.

Teacher terminal output được báo:

- Nhãn 0 → `0.472428`
- Nhãn 1 → `0.527572`

Hãy kiểm toán lại từ raw, không lấy các con số trên làm kết quả đã được xác minh độc lập.

Cần xác định:

1. Hai số đó là xác suất của lớp nào, có phải softmax hai lớp hay không.
2. Mỗi case có teacher output nào khác hai giá trị nói trên không, ở độ chính xác đầy đủ.
3. Với từng nhãn, phương sai có thực sự bằng 0 trong giới hạn số học không.
4. Input pulse timing/amplitude có thực sự tác động đến neuron với trọng số khác 0 không.
5. Teacher có thực sự cần P để phân biệt nhãn không.
6. Student representability witness có đúng với topology và weight bounds không.
7. Readout-only feature collision có được kiểm bằng dữ liệu thực không.
8. Có input leakage, hardcoded answer, oracle codec hoặc reuse TEST nào không.
9. Số forward, thời gian, RSS và training updates có khớp raw/log không.
10. Có sai lệch giữa protocol khóa trước và implementation không.

Viết audit bằng mã độc lập, read-only với raw.

Không được chạy thêm 144 forward hoặc rerun benchmark nếu chỉ cần đọc raw và tính lại kết quả.

### A4. Kiểm chứng phép quy kết teacher

Nếu đầu ra teacher chỉ là hàm của nhãn:

\[
T(x)=g(y)
\]

thì baseline label-only có thể tái tạo soft target đó.

Nếu student update dùng:

\[
\delta=r-p_C
\]

và \(r=T(x)\), hãy chứng minh cụ thể khi nào label-only baseline cho **cùng update** với teacher arm.

So sánh trong cùng điều kiện student weights, normalization, update order, optimizer state, eligibility và clipping.

Phân biệt:

- Teacher giúp nhờ thông tin ngoài label.
- Teacher giúp nhờ cách biểu diễn tốt hơn.
- Teacher chỉ biến đổi soft label.
- Teacher chỉ thay đổi độ lớn gradient/loss.
- Teacher có lợi ích nhờ compute hoặc supervision lớn hơn.

Nếu không phân biệt được, giữ nguyên:

`PREFLIGHT_NO_GO`

Tuyệt đối không dùng GPU để huấn luyện lại thiết kế này.

---

## PHẦN B — KIỂM TRA NHỮNG GÌ ĐÃ ĐƯỢC THỬ

Đọc:

- `docs/RESEARCH_PLAN.md`
- `docs/STATUS.md`
- `docs/research/SCOPE_ALIF_ARCHITECTURE_CANDIDATE_20261115.md`
- `docs/research/SCOPE_ALIF_CAPACITY_RELEASE_ANALYSIS_20261115.md`
- `docs/research/SCOPE_ALIF_CAPACITY_PRIOR_ART_20261115.md`
- `src/scarx/neural/core.py`
- `src/scarx/hybrid/learner.py`
- `src/scarx/hybrid/cuda_alif.py`
- `src/scarx/evaluation/readout_eligibility.py`

Đối chiếu các kết quả âm của SCAR-X: Phase 2, ACR, R5–R13, local credit, readout-only, shared-head, R13.2 và P0/P1 đã có.

Đối chiếu prior art:

- Progressive Networks.
- Progress & Compress.
- HLOP-SNN.
- e-prop/local eligibility.
- Các nghiên cứu distillation hoặc temporal feature transfer cho SNN gần nhất.
- ROW khi liên quan tới functional substitution và capacity release.

**Không được lặp lại một bài toán đã có kết luận chỉ bằng cách đổi tên hoặc thêm seed.**

Tránh chạy lại:

- XOR hoặc OR một tick.
- MOVE hai vật, hai vị trí.
- P0 readout sign với cùng dữ liệu.
- P1 terminal-logit-only.
- EMA, HLOP và những sweep đã STOP.
- Bài teacher đã đạt trần accuracy mà không còn headroom.

Nếu phát hiện thí nghiệm mới đã được thử với giả định tương đương, ghi `DUPLICATE_STOP` và giải thích bằng artifact cụ thể.

---

## PHẦN C — TÌM ĐÚNG MỘT GIẢ THUYẾT MỚI

Giả thuyết ưu tiên:

**Thông tin biểu diễn theo thời gian từ vùng P của một teacher đã có kỹ năng có giúp student S học hidden representation tốt hơn tín hiệu verified-label đơn thuần hay không?**

Đây chỉ là giả thuyết ưu tiên, không được mặc định đã đúng.

Có thể xem xét:

- Temporal spike trajectory của P.
- Filtered temporal features của P.
- Tín hiệu eligibility/credit gắn với sự kiện có căn cứ.
- Một phép alignment nhỏ từ teacher features sang student features, nếu định nghĩa được nguồn tín hiệu và chi phí.

Nhưng KHÔNG được tự động chọn spike trace chỉ vì nó phức tạp hơn logits.

### Preflight mới phải kiểm tra

**P1 — Nguồn thông tin và attribution**

Với các mẫu có cùng verified label, kiểm tra teacher signal có biến thiên có cấu trúc, có liên quan tới tín hiệu cần học hay không.

Phải so với label-conditioned controls, trong đó tín hiệu được sinh từ nhãn hoặc tráo trong cùng lớp.

Biến thiên khác 0 là điều kiện chẩn đoán, chưa đủ chứng minh hữu ích.

Teacher trace là hàm của input nên không được tùy tiện tuyên bố chứa thông tin Shannon mới vượt toàn bộ input gốc. Hãy kiểm định lợi ích về *biểu diễn và learning dynamics*.

**P2 — Teacher competence**

Teacher phải giải được task trên dữ liệu chưa dùng để xây hoặc học teacher.

Nếu teacher hand-engineered hoặc oracle-coded, công bố rõ. Một teacher dựng sẵn chỉ hỗ trợ thử cơ chế; không chứng minh P của SCAR-X đã tự học kỹ năng.

Không được coi teacher accuracy cao nhờ nhãn được đưa vào codec là thành công neural.

**P3 — Student feasibility**

Kiểm xem mạng S với cùng:

- Số neuron.
- Topology.
- Fan-in.
- Precision.
- Input encoder.
- Output contract.
- Những trọng số thực sự được phép học.

có tồn tại lời giải cho union old+new skill không.

Phân biệt:

- `REPRESENTATION_NO_GO`: không có bằng chứng mô hình student biểu diễn nổi.
- `LEARNER_NO_GO`: có lời giải nhưng quy tắc local không tìm được trong budget.
- `ATTRIBUTION_NO_GO`: không thể quy lợi ích cho teacher P.
- `DATA_NO_GO`: task thiếu thông tin hoặc không có headroom.

Không được trộn các nguyên nhân.

**P4 — Tác động nhân quả thực sự**

Teacher-derived signal phải tạo ra một dự đoán có thể bác bỏ.

Ví dụ, giữ nguyên verified labels và student, nhưng tráo teacher trace giữa các mẫu cùng lớp hoặc can thiệp P theo cách đã định trước. Nếu kết quả học không thay đổi, chưa có bằng chứng teacher-specific benefit.

Không được chọn ablation sau khi xem accuracy.

**P5 — Generalization có thật**

Không dùng biến held-out vô nghĩa với trọng số luôn bằng 0.

Test phải chứa biến thiên thực sự tác động lên dynamics, ví dụ khoảng cách sự kiện, tổ hợp thời gian, thứ tự sự kiện hoặc cấu trúc nhiễu nhân quả.

Không được dùng cùng một tập nhỏ để vừa chọn cơ chế vừa tuyên bố OOD.

Nếu bất kỳ điều kiện thiết yếu nào FAIL, dừng trước training.

---

## PHẦN D — NẾU PREFLIGHT PASS, ĐĂNG KÝ MỘT EXPERIMENT NHỎ

Bạn được phép chuẩn bị và thực hiện một thí nghiệm nghiên cứu cô lập, không triển khai production.

Trước lần learning update đầu tiên, phải khóa:

- Giả thuyết.
- Dataset generator và source hash.
- Train/DEV/untouched TEST split.
- Seeds.
- Teacher construction/training.
- Student initialization.
- Kích thước S/P.
- Local learning equations.
- Arms và controls.
- Learning updates.
- Metric và confidence interval.
- Tài nguyên tối đa.
- Quy tắc dừng.
- Quy tắc kiểm tra và rollback.

Đề xuất ngân sách tối đa:

- 3 paired seeds.
- 24 arm-runs **tổng cộng**, không phải 24 cho mỗi điều kiện.
- 30 phút tổng thời gian chạy thí nghiệm, kể cả profiling và audit.
- Peak RAM riêng của thí nghiệm dưới 2 GB.
- Không thay đổi precision, số neuron hoặc khoa học gates để cứu kết quả.

Có thể đăng ký bốn arm chính, mỗi arm qua ba seed và hai điều kiện khởi tạo dấu readout, tổng 24 arm-runs:

**A — Label-only local ALIF:** học với verified feedback, không dùng teacher signal.

**B — Teacher temporal transfer:** cùng input và verified feedback, thêm tín hiệu từ P theo cơ chế đã khóa.

**C — Matched teacher control:** giữ phân bố và chi phí tín hiệu teacher nhưng phá quan hệ sample-specific, chẳng hạn tráo trace trong cùng lớp.

**D — Readout-only:** hidden frozen, học readout trên cùng dữ liệu, encoder và output contract.

Nếu thấy một bộ đối chứng khác mạnh hơn, phải giải thích và đăng ký thay thế **trước training**, không thêm arm sau khi xem score.

Mọi khác biệt về supervision, teacher pretraining, compute và tham số phải được tính và báo cáo. Nếu không thể match tuyệt đối, trình bày hai profile chi phí thay vì gọi so sánh là công bằng hoàn toàn.

### Quy tắc học

Candidate phải giữ ALIF forward và eligibility-based local learning đã quy định.

Không dùng BPTT/RTRL/backward toàn mạng làm learner chính. Được dùng phương pháp khác như diagnostic upper bound nếu tách riêng, có ngân sách và không lấy kết quả đó làm thành tích local ALIF.

Phải ghi rõ physical update cuối sau:

- Smoothing.
- Energy normalization.
- Clipping.
- Weight bounds.
- Trainability masks.

Không thay đổi hidden, normalization, recurrence hoặc readout ngoài phạm vi đã công bố.

---

## PHẦN E — TẬN DỤNG PHẦN CỨNG MÁY TÔI

Máy là laptop Windows 11, cấu hình đã từng được mô tả có CPU 14 nhân/20 luồng và 32 GB RAM. GPU chưa được xác minh chắc chắn.

**Không tin cấu hình kể lại; tự kiểm tra trực tiếp.**

Kiểm tra:

- CPU chính xác, physical/logical cores.
- RAM thực và RAM còn trống.
- GPU, compute capability, VRAM tổng và khả dụng.
- CUDA driver/runtime.
- CuPy hoặc backend CUDA đã cài.
- Tiến trình Hermes, llama.cpp và các dịch vụ GPU khác.
- Nhiệt độ hoặc throttling khi các bộ đo sẵn có hỗ trợ.

Không tắt Hermes, chiếm VRAM đang phục vụ, thay driver hay cài lại CUDA để chạy thí nghiệm nhỏ.

### Chọn backend tối ưu thực tế

Benchmark một workload ALIF đúng phương trình và shape dự kiến trên CPU và GPU, tính cả transfer, kernel launch, synchronization và peak allocations.

Đối với H48/H128 nhỏ, CPU có thể nhanh hơn GPU. Nếu CPU nhanh hơn, dùng CPU.

Nếu CUDA nhanh hơn và VRAM đủ an toàn, sử dụng GPU cho phép đo neural; CPU thực hiện I/O và audit.

Không chuyển một thuật toán tuần tự thời gian thành batch song song sai semantics.

Có thể batch nhiều episode độc lập nếu reset và thứ tự cập nhật vẫn đúng hợp đồng.

Không biến benchmark backend thành một sweep neural không được đăng ký.

Giữ tổng chi phí và ngưỡng tài nguyên đã khóa. Nếu thiếu VRAM hoặc có nguy cơ ảnh hưởng Hermes, dùng CPU hoặc dừng.

---

## PHẦN F — GATES SAU TRAINING

**G0 — Preflight và attribution**

Nếu teacher signal không phân biệt được với label-only, STOP trước training.

**G1 — Teacher qualification**

Nếu teacher chưa có năng lực thật trên task hoặc có leakage, STOP.

**G2 — Student transfer**

Đánh giá từng arm trên DEV chưa dùng học.

Đặc biệt kiểm:

- Lợi ích của teacher P so với A và C.
- Hidden learning so với readout-only.
- Kết quả trên cả hai điều kiện khởi tạo dấu readout.
- Hiệu quả trên trường hợp mới, không chỉ training exact-match.
- Old valid skill retention.
- Causal ablation.
- Số update và chi phí thực.

Nếu lợi ích không rõ hoặc chỉ đến từ label smoothing, không tuyên bố teacher transfer.

**G3 — Hard removal of P**

Chỉ nếu G2 đủ điều kiện, tạo checkpoint nghiên cứu mới rồi kiểm student S-only.

Không cho S đọc:

- Weights P.
- Spike/state P.
- Teacher logits.
- Hidden teacher targets khi inference.
- Teacher archive/RAG chứa đáp án.
- Những output trung gian không thuộc serving contract.

Reload checkpoint trong một tiến trình độc lập và kiểm tra lại old/new skills.

Không reset P trong Hermes.

**G4 — Plastic reuse**

Chỉ nếu G3 PASS, chạy một chu kỳ P mới đã được đăng ký trước, có dữ liệu kỹ năng tiếp theo độc lập.

Kiểm acquisition mới và retention qua lần chuyển trước.

Một chu kỳ thành công không được tuyên bố là lifelong learning.

Không tự động mở rộng lên 20 stages nếu protocol chưa khóa và ngân sách không đủ.

### Ngưỡng nghiên cứu SCAR-X

Không thay đổi:

- Acquisition ≥65%.
- Hidden gain ≥5 điểm phần trăm so với đối chứng mạnh phù hợp.
- Cận dưới CI95% của gain lớn hơn 0.
- Mean forgetting sau 20 chặng ≤3 điểm phần trăm.
- Worst-group forgetting ≤5 điểm phần trăm.

Nếu số seed quá ít để đánh giá CI có ý nghĩa, ghi `INCONCLUSIVE`.

Không dùng TEST cũ hoặc tăng seed sau kết quả để tạo PASS.

---

## PHẦN G — BẢO TOÀN KẾT QUẢ VÀ AUDIT

Mọi kết quả phải có:

- Source SHA và dirty status.
- Protocol hash trước training.
- Dataset/seed hashes.
- Số neural forward và learning updates.
- Raw predictions và labels.
- Loss/accuracy theo arm và seed.
- Hidden weight deltas thực.
- Trace/eligibility signal diagnostics.
- Ablation outputs.
- Checkpoint hashes.
- CPU/GPU time.
- Peak RSS/VRAM.
- Trạng thái STOP/PASS/INCONCLUSIVE.

Audit độc lập phải đọc raw để tự tính lại metric mà không huấn luyện.

Không được chỉ chép lại các trường `passed` từ runner.

Giữ kết quả âm vĩnh viễn trong báo cáo nghiên cứu; không sửa protocol cũ để hợp thức hóa một run mới.

Nếu có sự cố, ghi rõ số update đã chạy, trạng thái đã lưu và phần chưa kiểm.

---

## PHẦN H — QUYỀN HẠN VÀ ĐIỀU KIỆN DỪNG

Được phép:

- Đọc repo và kết quả hiện có.
- Kiểm toán P1 bằng mã.
- Tạo tài liệu và thí nghiệm cô lập trong nhánh/worktree `research/`.
- Chạy thí nghiệm nếu mọi preflight PASS và protocol đã khóa.
- Lưu raw, báo cáo và checkpoint nghiên cứu.

Không được phép:

- Sửa `main`.
- Sửa Hermes hoặc checkpoint live.
- Reset P của model đang phục vụ.
- Merge/deploy hoặc tự promote model.
- Push nếu chưa được yêu cầu.
- Nâng budget, số seed, neuron hoặc precision.
- Tiếp tục chạy sau một kết quả âm đã đủ bác bỏ giả thuyết.

Nếu thí nghiệm không khả thi, **dừng đúng chỗ và nêu kết quả âm**.

---

## PHẦN I — BÁO CÁO CUỐI CÙNG BẰNG TIẾNG VIỆT

Trả lời rõ từng câu:

1. Kết quả P1 commit `6e08a361` đã được kiểm toán đến mức nào? Có sai sót gì không?
2. Teacher terminal-logit-only có đúng là không phân biệt được label-only baseline không?
3. Tín hiệu teacher P mới có vượt qua preflight attribution không?
4. Codex đã chạy training thật hay dừng trước training? Chính xác bao nhiêu forward/update?
5. Hidden ALIF-local có vượt readout-only cùng ngân sách không?
6. Loại hoàn toàn P và restart có giữ được kỹ năng không?
7. Chu kỳ tiếp theo có tái sử dụng P thành công không?
8. Tổng tài nguyên và thời gian thực tế là bao nhiêu?
9. Có phát hiện cơ chế mới hay chỉ adaptation prior art?
10. **SCAR-X đã giải được bài toán tự học liên tục chưa?** Trả lời YES, NO hoặc NOT YET, có bằng chứng.
11. Quyết định cuối: `GO`, `PIVOT`, `STOP` hoặc `INCONCLUSIVE` và đúng một bước kế tiếp đáng làm.

**Nguyên tắc cuối:** Tôi muốn khám phá một cơ chế học hiệu quả cho SCAR-X, không cần báo cáo đẹp. Kết quả FAIL có bằng chứng đáng giá hơn PASS do dữ liệu sai, đối chứng yếu hoặc dùng quá ngân sách.

Hãy bắt đầu bằng việc đọc và kiểm toán ba artifact P1 gốc. Không bỏ qua bước này để chạy thẳng thí nghiệm mới.