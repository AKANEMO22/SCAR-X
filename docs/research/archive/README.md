# Nguồn nghiên cứu lưu trữ

227tệp báo cáo/protocol/metric receipts từ27nhánh nguồn được chép nguyên byte.
[source-index.json](source-index.json) ghi sourceSHA đầy đủ, SHA256, branch
nguồn, localpath và immutableGitHubURL cho từng phiên bản. Mỗi đường dẫn/tác
giả chỉ có một bản đọc chính để tránh nhân nhiều bản sao; các version khác
vẫn được truy cập bằng SHA/tag, không bị xóa khỏi lịch sử Git.

- [Grok](grok/README.md)
- [DeepSeek](deepseek/README.md), gồm toàn bộ reports R7round1–43 hiện có.
- [Gemini/Antigravity](gemini/README.md), gồm errata và source snapshotR13 cũ.
- [Sổ đánh giá chính](../EXPERIMENT_LEDGER.md), đọc trước khi dùng số từ bản gốc.

**Bản gốc không mặc định đúng.** Có report tự gọi PASS hoặc giữ số đã bị
retract; chúng được giữ nguyên để audit, không biến thành kết luận của main.
Metricfile ở đây là evidence tác giả, không independentlyscoredprediction
archive trừ khi sổ thí nghiệm nêu rõ phép kiểm. Git tags giữ toàn bộ original
code/raw/checkpoint; peer learner chưa kiểm không được nhập vào serving.

R13.1 ongoing bị loại khỏi import. Chỉ đọc source commitf4272ba của R13 trước
đó qua Gitobject, không đọc/copy các file R13.1 chưa commit, không thay
worktree/branch/cache/process đang dùng. Thư mục archive này không phải nơi
để ghi kết quả tiếp diễn của R13.1.

Đường dẫn tương đối bên trong bản gốc giữ nguyên và có thể dẫn sang source
không mirror ở đây. `source-index.json` luôn có URL source SHA để mở đúng cây
gốc. Không sửa báo cáo cũ chỉ để làm link đẹp rồi làm mất checksum.

```powershell
git fetch origin --tags
git worktree add --detach local/reproduce-r12 archive/2026-11-13/branches/research/gemini/r12-replay-lite-continual-memory-20261112
```

Tạo checkout reproduction mới; không checkout/reset live R13.1, không ghi đè
run/reservation cũ. Private ignored local state chỉ có trong recoveryZIP local,
không tự đưa cache/model/session của người dùng lên GitHub.
