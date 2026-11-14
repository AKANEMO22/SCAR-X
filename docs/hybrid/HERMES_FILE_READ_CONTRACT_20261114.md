# Sửa một nguồn phản hồi sai trước khi dạy SNN

**Đã sửa và cài đường đọc file mặc định của Hermes:** file bị khóa đọc không
còn được báo là file rỗng0byte. Đây là sửa tính đúng của công cụ, không phải
bằng chứng model đã tự hiểu nguyên nhân hoặc học được kỹ năng mới.

## Phát hiện từ thực thi thật

[P03ce157](../../runs/20261114T051425Z-file-contract-p0-v1-3ce157/result.json)
thực thi14read_file qua dispatcher và public observer hooks của Hermes, dùng
HERMES_HOME/project cô lập. Không VLM, neural forward/training, web hoặc sửa
quyền/ACL. Mã model_tools trong fixture khớp source đang cài lúc kiểm.

File rỗng thật và file24byte đang được giữ bằng exclusive read handle đều trả
**cùng một JSON**: content rỗng, size0, lines0, không error, hint “File is empty”.
Observer đều nhận statusok. File24byte được tạo bởi harness và không ghi/xóa
trong lúc đọc; sau khi đóng handle bytehash/nội dung còn đúng. Không thể suy
nguyên nhân từ hai packet giống nhau bằng cách huấn luyện SNN thêm.

Nguồn lỗi ở parser compound read: dữ liệu kích thước hoặc số dòng không parse
được bị thay bằng0. Trạng thái của phần cuối pipeline cũng có thể che lỗi bước
đọc phía trước. Bổ sung một nhãn metadata không sửa được dữ liệu đã mất, nên
không triển khai classifier metadata ban đầu hoặc coi reported success là truth.

Các kiểm tra khác giữ được: file text có cụm từ “Permission denied” vẫn đọc
thành công; partial read có truncated; ba lần thay nội dung tạo ba digest khác.
Fixture định làm binaryNUL được tool đoán UTF16BE và hiển thị có ký tự thay thế:
đây là giới hạn heuristic cần đánh giá riêng, không lấy ý định fixture làm một
oracle phổ quát về encoding.

Auditor ban đầu có lỗi giả định độ dài/newline Windows. Raw kết quả không đổi;
giữ auditv1 và correction receipt, [auditv2](../../runs/20261114T051425Z-file-contract-p0-v1-3ce157/independent-contract-audit-v2.json)
dùng actual24byte. Không sửa nhãn/chỉ số model hoặc lấy lần audit lỗi làm PASS.

## Thay đổi và kiểm chứng

[Protocol sửa lỗi](../../experiments/hybrid-hermes-file-read-contract-repair-v1.md),
[engineering acceptance3aeff8](../../runs/20261114T052552Z-hermes-file-contract-repair-v1-3aeff8/engineering-acceptance-v1.json).

- Parser từ chối byte-size/line-count thiếu hoặc âm, sample read thất bại,
  sample rỗng trái với size và metadata0byte mâu thuẫn với content.
- Tách xử lý compound response vào sibling nhỏ `tools/file_operations_read_contract.py`.
  Facade giảm40dòng; không thêm tool/schema, service, quyền hoặc dependency.
- 9ca regression lỗi trên base và qua trên candidate; hai control đọc hợp lệ
  vẫn qua. Lần red đầu harness dùng sai sentinel đã được giữ và loại khỏi bằng
  chứng regression; redv2/green dùng sentinel thật của lời gọi.
- Thực thi14ca với bản sửa đầu,3ca sau refactor và3ca cold-import trực tiếp mã
  đã cài: tổng20calls đúng budget. File rỗng thật vẫnok; file khóa trảerror;
  partial read giữ nguyên; source có câu giống thông báo lỗi không bị diễn giải
  thành lỗi công cụ. Không thêm call để đổi kết quả sau khi đã chấm.

Runner chuẩn `scripts/run_tests.sh`,4file,1worker, không retry:

| Mã | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Base |104|19|9|
| Candidate |114|9|9|

11test hợp đồng mới đều qua.9failure còn lại cũng nằm trong baseline:3case
search path/MSYS hoặc gitignore,4case kỳ vọng POSIX file modes trênWindows,
2case symlink tới `/dev/zero`. Không sửa/ẩn chúng trong lượt này. Một failure
background-review/dedup chỉ có ở baseline; chưa quy việc nó biến mất cho patch.
**Không gọi toàn bộ upstream suite là xanh.** Lượt bare-pytest thăm dò trước
đó được giữ riêng; kết luận đối chiếu dùng canonical per-file logs.

## Đã cài ở đâu và khôi phục thế nào?

[Installer](../../scripts/install_hermes_file_read_repair.py) yêu cầu acceptance
receipt và SHA nguồn đúng phiên bản; dừng nếu gặp source/sibling khác hoặc sửa
đồng thời. Không thay package môi trường, model, config, session hay neural state.
Tệp hiện cài tại `%LOCALAPPDATA%/hermes/hermes-agent/tools/`:

- `file_operations.py`:SHA256 `68bb3ef6fb07c12384641f1359a7744dbf4814e1d84a79b7b20407d2bad3dee9`.
- `file_operations_read_contract.py`:SHA256 `e846b861d9a2f672f47f026b72aab2b4799109b93cef55f9720309192add5a2f`.

Backup nguyên byte và receipt nằm trong
`local/hermes-file-read-contract-backups/20261114T054057412253Z/` của worktree.
Khi cần hoàn tác: đối chiếu current hashes với receipt, khôi phục
`file_operations.py.before` bằng atomic replace, bỏ sibling do patch tạo nếu
hash còn đúng, rồi nạp lại backend. Không ghi đè source đã được updater/agent
khác thay; phải review lại phiên bản mới. Chưa thực thi rollback trên live.

[Bằng chứng cài/nạp](../../runs/20261114T052552Z-hermes-file-contract-repair-v1-3aeff8/installation-and-activation-v1.json):
cold imports đúng sourceHermes đã cài và3calls thực qua. Gateway cũ drain sạch,
PID mới37992; Desktop tự kết nối lại và hiển thị Backendready. Không dùng memory
introspection để giả xác nhận từng module đang import. SCAR-X gateway35036 và
nativeVLM39372 giữ nguyên; liveALIF vẫn5updates tại lúc kiểm.

## Những việc còn mở

Cập nhật sau lượt này: [bản nâng cấp read-paths v2](HERMES_READ_PATHS_V2_20261114.md)
đã sửa/test/cài sequential và status pipeline trên GitBash. Kết quả và SHA ở
các mục trên giữ ý nghĩa lịch sử của lượt3aeff8; source hiện tại xem báo cáo v2.

1. Kiểm riêng fallback sequential và lỗi đầu pipeline bị tiến trình sau che:
   sửa hiện tại tập trung parser compound/default, chưa chứng minh mọi backend
   và race đọc nguồn đều có thông báo đúng. Không gán engineering acceptance này
   cho toàn bộ I/O hoặc tất cả lỗi quyền truy cập.
2. SNNserving vẫn không nhận đủ error/probe outcome trong goal_sequence. Sửa
   công cụ là điều kiện cần; chưa phải nối được learned recovery policy.
3. Hidden advantage, học từ sai rồi chuyển kỹ năng và retention vẫn PIVOT theo
   [các chẩn đoán đã đo](HIDDEN_BINDING_AND_COMMON_PROBE_20261114.md).
4. Lượt8h554c19 vẫn độc lập với sửaHermes; source và PID41904 giữ nguyên. Đủ thời
   gian engineering cũng không thay quality/video/lifelong gates.

Các kết quả được đăng ở [localhost](http://127.0.0.1:8775/?after=7357&tab=trace)
với actororchestrator; không tính các ca file thử thành câu trả lời model hoặc
neural updates. Main và thí nghiệm R13 không thay đổi.
