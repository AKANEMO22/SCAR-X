# SCAR-X — trung tâm kết quả nghiên cứu

**Review đề xuất người dùng15/11:** [Neural Operator Lattice / Functional Recomposition](NEURAL_OPERATOR_LATTICE_REVIEW_20261115.md).
Giữ câu hỏi compatibility, yêu cầu sửa spec trước triển khai. Đối chiếu DNAR,
NCA/BPTT, boundary-firstPC và ROW negative compilation; đếm lại savedR13.2,
không gọi model hoặc đổi ALIF/gate. [Nguồn và phạm vi](literature/sources/operator-lattice-review-20261115.json).

**Tự phản biện15/11:** [cận toán, phản ví dụ và các nghiên cứu không cần lặp](ARCHITECTURE_SELF_CRITIQUE_20261115.md).
Hạ ưu tiên gói kiến trúc B; tách quan sát, dự đoán và điều chưa biết. Chưa có
lượt chạy mới hoặc bằng chứng một kiến trúc mới đã thắng.

**Nghiên cứu sơ bộ kiến trúc15/11:** [ma trận và phản biện](ARCHITECTURE_DESIGN_SPACE_20261115.md),
[nguồn có commit/hash/phạm vi đọc](literature/sources/architecture-design-space-20261115.json).
Đây là tài liệu thiết kế trước thí nghiệm; chưa có learner, điểm benchmark hoặc
phát hiện “kiến trúc mới đã thắng”. Master vẫn quản lý cửa duyệt bước tiếp theo.

**15/11 localALIFGPU:** [report/source/figure](../hybrid/LOCAL_CREDIT_AND_RETENTION_20261115.md)
giữ3seed representationgain cóđiềukiện, forgetting vàRAGlimits. Signederror
common-headDEV78,33vsfrozen62,50%, rawPCA64cùngreadout79,72%; chưa superiority.
Importance/slowanchor chỉ1/3seedqua jointgate, khôngscale/promote hoặcTESTretune.

**Quy tắc15/11:** bắt buộc qua [literature-first gate của master](../RESEARCH_PLAN.md#0-literature-first-bắt-buộc-trước-nghiên-cứu-mới)
trước nghiên cứu mới. [Bảng kế thừa hiện tại](literature/REUSE_DECISIONS_20261115.md)
ghi source/code đã đọc, quyết định dùng lại và những lượt không cần chạy lại;
không tạo thêm roadmap hoặc cấp quyền mở experiment.

Cập nhật13/11/2026. Đây là điểm bắt đầu duy nhất để tra kết quả, thay vì mở
từng branch hoặc suy từ một báo cáo tác giả. [Master plan](../RESEARCH_PLAN.md)
quản lý định hướng; các protocol và số đo lịch sử giữ nguyên.

**Hybrid13/11 mới:** [implementation/giới hạn](HYBRID_IMPLEMENTATION_20261113.md),
[vận hành](../hybrid/README.md). Qwenfrozen+localALIF, chatHermes thường,
trace8775; engineeringMVP, chưa scientificPASS. Source G1 và activepeer không đổi.

| Cần tìm | Mở |
|---|---|
| Đã thử những gì, đi theo nhánh nào? | [Sơ đồ thí nghiệm](EXPERIMENT_MAP.md) |
| Kết quả chi tiết, bằng chứng và giới hạn | [Sổ thí nghiệm](EXPERIMENT_LEDGER.md) |
| Tra theo mã, tác giả, trạng thái bằng script | [Catalog JSON](experiment-catalog.json) |
| Báo cáo/protocol gốc Grok, DeepSeek, Gemini | [Nguồn gốc lưu trữ](archive/README.md) |
| Source SHA, tag phục hồi từng branch/worktree | [Danh mục archive](../maintenance/20261113/branch-archive.json) |
| Đã dọn gì và giữ gì? | [Bàn giao cleanup](../maintenance/20261113/README.md) |

## Những điều đã kiểm được

- Hidden synapses có thể cải thiện representation trên các bài event cụ thể;
  **thay đổi trọng số không tự động là học có ích**. Direct, probe, causal revert
  và shuffled controls được báo riêng.
- R8 confirmation12seed có integrityPASS nhưng chỉ XOR đủ gate, nên
  `P1_FAIL_RESEARCH_ONLY`; không dùng kết quả đó để chọn lại candidate.
- ACR hoàn tất375modelrecords: `ACR_SPECIALIZED_GAIN`, chưa một credit rule
  đủ tốt trên nhiều bài và joint checkpoint.
- H128Own cải thiện direct ở các bài đã đo: Binding87,30%, XOR99,41%,
  Distractor69,40%. Ba task fit checkpoint riêng, không chứng minh một model
  giữ đồng thời mọi kỹ năng. Gate tăng thêm learned-vs-frozen là
  `CAPACITY_ONLY_SIGNAL`; có disclosure hiệu ứng trần của contrast XOR.
- Literature adapters/tests và snapshot/rollback đã có kiểm kỹ thuật; chưa
  chứng minh hết readout overwrite hoặc lifelong retention.

## Điều chưa được kết luận

Chưa có bằng chứng đã đạt học liên tục task-agnostic theo acquisition/final65%,
meanforget≤3đ và worst≤5đ từng seed; chưa AI ngôn ngữ/vision đa năng hoặc ngang
model thương mại. Replay giúp một số lượt nhưng không đồng nghĩa đạt gate.
Không suy thắng/thua một ngành nghiên cứu từ một prototype H48/H128.

**R13-family đang chạy (ảnh ban đầu R13.1) và được người dùng bảo vệ.** Worktree
`C:\Users\hachimi\Downloads\model agent\scar-x-gemini-r7`, nhánh
`research/gemini/r13-1-spike-composition-vs-small-llm`, các file chưa commit,
download/cache và process của nó không thuộc cleanup. Không chấm kết quả R13.1
trong catalog hoàn tất. R13 cũ ở commitf4272ba là một snapshot khác, không
phải kết quả mới của R13.1.

## Cách đọc trạng thái

`VERIFIED_SCOPED` nghĩa là phạm vi ghi rõ đã được kiểm, không phải approval
production. `AUTHOR_RECORD` là số/claim từ source tác giả chưa tái kiểm đầy đủ.
`RETRACTED`/`INVALID_CONTROL` được giữ để không lặp lỗi. `IN_PROGRESS_EXCLUDED`
không được archive/xóa khi đang chạy. P0 technicalPASS không là qualityPASS;
probe/refit tốt không thay direct output hoặc retention.

## Cách lưu các lượt mới

1. Đăng ký protocol/seed/budget trước scores, một runID duy nhất trong `runs/`.
2. Lưu raw/labels/predictions/checkpoint/RNG/resource/provenance; báo cả aborts.
3. Thêm report và entry catalog, nêu clearly authorclaim hay independentaudit.
4. Kết thúc/review rồi mới hợp nhất research code; không tự promote vào serving.
5. Archive tag theo SHA trước khi xóa branch; active worktree là ngoại lệ bắt buộc.

```powershell
python scripts/research/query_catalog.py list
python scripts/research/query_catalog.py search replay
python scripts/research/query_catalog.py show CAPACITY-H64-H128
python scripts/research/query_catalog.py verify
```

Historical rerun cần checkout đúng tag/SHA và source-byte-bundle nếu có. Các
hash của nguồn cũ không được nới chỉ vì main đã hợp nhất hoặc đổi đường dẫn.
Không rerun một runID đã tồn tại, không xóa reservation để ghi đè bằng chứng.
