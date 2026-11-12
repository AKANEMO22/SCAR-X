# Codex — START HERE: SCAR-X Independent Forensic Audit

Bạn tiếp nhận dự án **SCAR-X**, repository https://github.com/AKANEMO22/SCAR-X. Tôi giao cho bạn kiểm toán khoa học và kỹ thuật các chiến dịch nhiều agent (DeepSeek, Grok, Gemini/Antigravity) R1–R6D, không phải xác nhận họ đã đúng. **Bắt buộc đọc toàn bộ tệp đính kèm** `SCARX_CODEX_FULL_RESEARCH_HANDOFF_2026-11-11.md` trước khi làm việc; đây là report chi tiết gồm lịch sử, commit, protocol, nguồn raw, per-seed metrics, code-level bugs, luận điểm đối kháng và execution plan.

## Nguyên tắc

- Không tin Grok/DeepSeek/Gemini, cũng không tin bất cứ con số nào của bản bàn giao nếu chưa tự so source + raw + test. Bản handoff chứa cả kết quả người khác tự báo cáo, kết quả JSON được tính lại, và kết luận phân tích tĩnh. Phân biệt rõ.
- Giữ đúng một recurrent LIF/ALIF SNN với local synaptic learning, CPU-first ≤2 GB. Không thay bằng Transformer/pretrained LLM/LLM proxy/RAG. BPTT/autograd chỉ là control cô lập; không vào core production. Không thay đổi `main`, live checkpoints hoặc dữ liệu của người dùng.
- Không tự merge, chạy large training, đổi neuron architecture, mở P2 pilot hoặc thay quy tắc học chưa được duyệt. Được lập branch/worktree audit, tái lập historical runs, kiểm lỗi, dựng evaluator và soạn protocol R6E/R7 để xin duyệt.
- Raw history bất biến. Không sửa old `result.json`, không xóa failed run, không đổi threshold sau kết quả, không gán PASS cho gate `None`/missing.

## Việc phải làm theo đúng thứ tự

1. Kiểm chứng repo HEAD, branches, worktrees, `AGENTS.md`, `docs/RESEARCH_PLAN.md`, manifests và historical sources. Baseline main: `495dab2f4bac0180e4d1cd28674391f055ccf09d`.
2. **Audit R6D mới:** nhánh `research/deepseek/r6d-audit` SHA `4bacd7b2c8a1df7ab47e55f266f2b6bf0e193893`, draft PR #12. Raw 8 seeds n16k `both_A=0.91782`, n1024 `.68728`, paired difference +.23054, matched unique−repeated +.15076; stress distractor `both_A=.55023` FAIL. Kiểm nguồn và tự tính lại.
3. Cẩn thận: `scripts/r6d_matrix.py` ghi `G4/G5/G8 = null`, chưa có completed aggregate `confirmation_pass`. `r6d_controls.py` dùng head training order khác, dẫn tới controls ref `both_A=.95035` vs matrix `.91782`. Phân biệt đúng R5B original normalization và R6D 'R5B-style fixed calibration'. Kiểm mọi gate, test perturbations và sample-level oracle.
4. **Audit Grok R6B:** nhánh `research/grok/r6-causal-readout-repair` SHA `c39e9db7771582d9933af402a209917328b71119`; reported head `.667617`, ridge `.985806`, NoClip improvement ~.00982 but order regression .1875; runner duplicate overwritten definitions và toy test dùng class shadow thay candidate thật. Độc lập tái lập từ actual predictions/counterfactual pairs.
5. Kiểm kê các vòng khác R1–R6 theo hồ sơ, chú trọng invalid time shift, post-hoc protocol deviations, readout-vs-hidden attribution, Gemini DC-ALIF strict_binding win nhưng order/pair_binding regressions và P2 forgetting fail.
6. Dựng independent reference evaluator + mutation tests; khóa comparator/protocol trước bất kỳ fresh confirmation. Không dùng những seed đã từng xuất hiện làm 'fresh'.
7. Viết `BASELINE_INVENTORY.md`, `FORENSIC_AUDIT.md`, independent tests, `CLAIM_LEDGER.json`, `R6D_GATE_INTEGRATION.json`, `R5B_R6B_REPRODUCTION.md`, `PROTOCOL_R6E_DRAFT.md`, `RESULTS.md`, cùng final full SHA. **Protocol R6E/R7 là đề xuất chờ người dùng duyệt, KHÔNG chạy tự động.**

## Cách kết luận

Mỗi claim phải có `CLAIM -> SOURCE + SHA -> RAW -> INDEPENDENT TEST -> COUNTEREXAMPLE -> VERDICT -> LIMITS`. Dùng `PASS`, `FAIL`, `INCONCLUSIVE`, `INVALID_HARNESS`, `NOT_TESTED`. Báo cáo cả seed xấu nhất và CI bootstrap trên **seed độc lập**, không gộp các sibling làm mẫu độc lập. Nhắc rằng frozen hidden + trained readout không có bằng chứng hidden neuron đã học được representation.

Bắt đầu bằng việc tóm tắt inventory và 5 rủi ro lớn nhất đã kiểm tra từ **code thực**; không viết plan chung chung thay cho audit. Không được nói công việc đã hoàn tất khi chưa chạy hoặc chưa có artifacts.
