# 05 — Thiết kế và tái lập

Cập nhật docs 2026-11-09. [RESEARCH_PLAN.md](../RESEARCH_PLAN.md) là master plan duy nhất. File này là hướng dẫn thực nghiệm, giữ mô tả Phase2 làm lịch sử. **Không có protocol hoặc lần chạy Phase2.2 nào được phê duyệt bởi lần cập nhật docs.**

**Phê duyệt tiếp theo Product-first:** người dùng đã cho triển khai MVP V0.1, local learning tương tác và kiểm thử sản phẩm; [ADR0005](../decisions/0005-product-first-mvp.md) thay phạm vi docs-only. [Acceptance scenario](../../experiments/mvp-v01-acceptance.md) ghi rõ demo và build iteration, không đổi research gate. Giữ bản ứng dụng hoạt động gần nhất khi nghiên cứu rule chưa đạt.

## Roadmap hiện tại: Phase 2.2 đề xuất

| Thứ tự | Mục tiêu / điều kiện mở | Nguồn tiêu chí chính |
|---|---|---|
| P0 | Khép checkpoint `plasticity-v1`, transaction learner/replay/RNG, verification evidence và RAM; trước consolidation tự động | [P0](../RESEARCH_PLAN.md#p0--khép-an-toàn-kỹ-thuật-trước-consolidation-tự-động) |
| P1 | Ưu tiên hidden representation và local credit assignment; loại shortcut và tách readout contribution | [P1](../RESEARCH_PLAN.md#p1--representation-và-credit-assignment) |
| P2 | Ưu tiên readout stability và importance gắn usefulness; đo học mới cùng giữ cũ | [P2](../RESEARCH_PLAN.md#p2--readout-stability-và-importance-theo-mức-hữu-ích) |
| P3 | Từ nhiều trải nghiệm tới quy luật áp dụng được khi retrieval tắt | [P3](../RESEARCH_PLAN.md#p3--chuyển-nhiều-trải-nghiệm-thành-kỹ-năng) |
| P4 | Offline consolidation thủ công, verifier/budget/rollback và matched extra-training controls | [P4](../RESEARCH_PLAN.md#p4--offline-consolidation-có-đối-chứng) |
| P5 | Stream20→50→100 giai đoạn theo gate, plasticity reserve, không tăng capacity | [P5](../RESEARCH_PLAN.md#p5--độ-bền-và-capacity-sau-20--50--100-giai-đoạn) |

Mỗi mục trong master đã ghi mục tiêu, giả thuyết, code, dữ liệu, baseline, metric, pass/fail, quyết định, rủi ro và phụ thuộc. Không thay mục tiêu neural bằng tối ưu replay. Chưa chọn thuật toán/topology mới.

## Kiểm soát attribution và tính công bằng

So cùng core initial seed và budget giữa frozen hidden, readout-only, full local, modulation control, replay/no-replay và retrieval-off. Probe head mới phải có cùng thuật toán và train split; so cả direct readout để tránh nhầm probe mạnh với core giỏi.

Phải kiểm cả thuật toán học và task validity: oracle chẩn đoán dữ liệu, chống leakage/bag-of-events shortcuts, positive control có thể học; một bài mà mọi control đều fail chưa phân biệt được credit rule. Với BPTT control, khớp stochasticity/forward khi mục tiêu là so learning rule, hoặc công bố rõ sự khác nhau.

Count tổng CPU training/consolidation, sample presentations, actual updates và peak buffer bytes; so equal-update và equal-time riêng. Verifier overhead và guard time báo riêng nhưng vẫn nằm trong tài nguyên toàn session. Trọng số sau retrieval không đổi không được gọi là học synapse.

## Đăng ký và quản lý kết quả

Trước chạy Phase2.2 phải có phê duyệt triển khai và protocol mới khóa data/seed mới, budget, primary contrasts, CI/correction, thresholds và stop rules. Không sửa `experiments/phase2-v1.md`, `phase21-v1.md` hoặc raw run sau khi thấy kết quả. Tối đa hai vòng điều chỉnh hợp lệ cho cùng giả thuyết; không có lợi ích thì NO-GO cho rule đó. Chi tiết giới hạn3h/run, 2GB và không mở rộng48neuron theo master.

Guard chỉ dùng development data, không final test; thuật toán benchmark không bật rollback để che forgetting. Đường vận hành cần transaction đầy đủ và phải báo cả rejected-update rate. Không dùng verified flag hoặc model confidence như nhãn thật mặc định.

## Hồ sơ đã hoàn thành Phase 2.1

[Catalog](../../artifacts/phase21-summary.json) định danh chính xác các run: representation/replay factorial; no-task-boundary noisy stream; strict binding audit; BPTT control riêng; reproducibility. [Báo cáo08](08_PHASE21_RESULTS.md) có số liệu và giới hạn. Không rerun để thay điểm lịch sử.

## Thực nghiệm Phase 2 — hồ sơ lịch sử

- `experiments/phase2-v1.md`: 5 condition ×3 seed, association/delay/reversal/20-context continual stream.
- `experiments/bptt-control-v1.md`: 3 seed, cùng forward/topology/initialization, surrogate BPTT CPU; cập nhật mọi weight, khác ngân sách update với block-local.
- `experiments/runtime-smoke-v1.md`: NumPy/native overhead, 1M byte SQLite lookup và vòng learn/checkpoint ngắn.

Core local không import PyTorch. BPTT control là module độc lập chạy trong process riêng. Không import checkpoint control vào SCAR-X.

Các control đã chạy: frozen; readout-only; unmodulated STDP; reward-modulated STDP; analog recurrent readout learning; surrogate BPTT. Model ít RAM pretrained và RAG chung vẫn chưa chạy; chưa có ngôn ngữ phù hợp để đối chiếu chất lượng và gate tăng quy mô đang thất bại.

## Chống lẫn dữ liệu

Training seeds và test seeds độc lập theo protocol. Generalization trong v1 là noise mới/delay mới của cùng lớp tín hiệu, không phải nghĩa mới. Continual task có context cue riêng để tránh yêu cầu ghi nhớ hai nhãn mâu thuẫn cho cùng một input. Reversal không có cue được báo riêng như đánh đổi học mới/quên cũ.

Reservoir replay giữ tối đa128 mẫu, lấy32 mẫu/chặng; chỉ dùng train. EMA consolidation sau chặng. Không dùng rollback trong bài đo forgetting để không che thất bại của thuật toán. API người dùng có development guard riêng.

## Lệnh tái lập

```powershell
python -m scarx benchmark --config configs/phase2.json
python -m scarx.evaluation.controls
python scripts/build_native.py
python -m pytest -q
python scripts/runtime_check.py --seconds 15
```

Mỗi run có ID UTC, cấu hình, seed, phiên bản dependency, hash code/protocol, raw trials và kết luận. Run xác nhận có source.zip. File result không sửa khi code thay đổi; tạo run mới.

## Gate lịch sử và NO-GO vẫn hiệu lực

Rule hiện tại không đạt hidden-contribution/retention → dừng 10M/30M/100M model và corpus lớn. Việc dừng là kết quả của protocol, không phải thiếu quyền thực thi.

Giới hạn cũ từng cho tối đa12 cấu hình/round và24h/screen. Với Phase2.2 đề xuất, dùng budget nghiêm hơn3h/run theo master và số condition khóa trước, không tự mở tìm kiếm. Không dùng checkpoint thương mại hoặc đổi sang hybrid để vượt gate. Các protocol đã chạy giữ nguyên budget/định nghĩa của chúng.

## Backlog dài hạn và mục đã ngừng làm mục tiêu

600+ tác vụ đa năng, feedback thật, load8h, iGPU, energy, quantization và native executable độc lập vẫn chưa được mở trong vòng này. Phase2.1 đã thử noise giả lập20%, không phải xác nhận mọi mức noise hoặc phản hồi thực tế.

**Deprecated như mục tiêu hiện hành:** length ladder8K→1M subword. Các benchmark/context data lịch sử vẫn giữ nguyên, có thể làm đối chứng khi có lý do nghiên cứu cụ thể, không thành gate kiến trúc. Thử15s không thay cho8h. Unit tests đúng implementation không chứng minh AI giỏi.
