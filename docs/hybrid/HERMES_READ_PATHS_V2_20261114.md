# Đường đọc dự phòng và lỗi pipeline của Hermes

**Đã sửa, cài và kiểm mã đã cài trên Windows/GitBash.** Khi bước đọc `sed`
thất bại nhưng `cut` phía sau thành công, Hermes nay trả lỗi đọc thay vì giả
một trang rỗng. Đường sequential cũng từ chối metadata kích thước/số dòng
không hợp lệ. Đây là sửa công cụ cung cấp bằng chứng cho SCAR-X; chưa phải SNN
tự hiểu nguyên nhân lỗi hoặc tự cải thiện.

## Bằng chứng trước và sau

[Protocol đã khóa](../../experiments/hybrid-hermes-read-paths-v2.md),
[run778239](../../runs/20261114T061928Z-hermes-read-paths-v2-778239/manifest.json),
[kiểm toán kết quả](../../runs/20261114T061928Z-hermes-read-paths-v2-778239/engineering-acceptance-v1.json).

Sáu ca dùng dispatcher `read_file` và public observer hook thật, HERMES_HOME và
file riêng. Harness chủ động làm lỗi framing để vào sequential, hoặc đưa hàm
shell `sed` trả17 vào một subshell cô lập. Các lệnh kiểm file/sample/page/count
còn lại vẫn chạy thật. Không gọi model, web hoặc gửi nhãn vào ALIF.

| Ca | Bản trước | Bản sửa |
|---|---|---|
| Compound đọc text | Có nội dung, statusok | Giữ nguyên |
| Compound, `sed` trả17 | `1\|`, statusok | Error, statuserror |
| Sequential đọc text | Có nội dung, statusok | Giữ nguyên |
| Sequential, `sed` trả17 | `1\|`, statusok | Error, statuserror |
| Sequential file rỗng thật | Empty0byte, statusok | Giữ nguyên |
| Sequential file bị khóa | Error, mô tả terminal unavailable | Error trung tính: nguồn hoặc backend chưa đọc được |

Đây là fault injection được công bố, không phải model tự tạo trajectory và tự
phát hiện lỗi. File rỗng hợp lệ khác với một lần đọc chưa được thiết lập.
Mỗi nhóm chạy6calls: baseline, candidate và cold-import sourceHermes đã cài;
tổng18dispatcher calls, đúng budget. Kết quả6ca installed khớp candidate.

Một lần chạy candidate dừng ở assertion path của harness trước khi gọi công
cụ: nguồnHermes thật được dùng nhưng reader là overlay được cấp rõ ràng.
Đã sửa assertion chỉ áp dụng khi không có overlay, giữ original receipt0calls.
Auditor đầu đọc thiếu ID test chứa khoảng trắng trong tham số; dừng trước
acceptance, giữ bản cũ và correction. Không thay dữ liệu/candidate để sửa điểm.

## Code và bảo trì

Giữ facade và một sibling nhỏ duy nhất, không thêm service/tool/schema hay
dependency. [Helper](../../integrations/hermes/file_operations_read_contract.py)
chứa các phần:

- Parser compound kiểm trạng thái sample, page, byte-size, line-count và tail;
  vẫn nhận frame cũ hai status, builder mới gửi năm status.
- Pipeline dùng `pipefail` nếu shell hỗ trợ, đặt trong subshell để không làm đổi
  shell option của phiên terminal gọi nó. Test thực kiểm shell option không rò.
- Sequential kiểm size/sample/page/count/tail và dừng khi chưa xác định đọc được.
  Không biến lỗi probe thành missing hoặc một nguyên nhân quyền truy cập cụ thể.

Facade reader giảm thêm53dòng so với bản sửa trước. Ruff C901 không có finding
ở ngưỡng20 cho helper mới. [Patch đúng phiên bản](../../integrations/hermes/file_read_paths_patch.py)
và [installer chung](../../scripts/install_hermes_file_read_repair.py) chỉ thay
khi cả facade/helper khớp SHA trước sửa; có backup cả hai, kiểm byte trước atomic
replace. Không đè sửa của updater/agent khác. Source và patch từng phiên bản nằm
trong run tương ứng, không sửa manifests/raw cũ.

Source cài hiện tại ở `%LOCALAPPDATA%/hermes/hermes-agent/tools/`:

| File | SHA256 |
|---|---|
| `file_operations.py` | `89328ad2cba667107dc12125fdee163a41739dfb85f5651d14a4cf678dd03cfc` |
| `file_operations_read_contract.py` | `39207c292c33dab410ac7ed519fea086d9822b80e0061bd004ab6ead8c43f190` |

Backup bản trước và receipt:
`local/hermes-file-read-contract-backups/20261114T063305090004Z/`.
Khôi phục cần so currentSHA với receipt rồi atomic-restore cả hai `.before`;
không ghi đè source đã thay bởi người khác. Chưa rollback live chỉ để thử.

## Test và lỗi nền được giữ nguyên

23test liên quan qua, gồm11test compound trước đây và12test mới.11test mới
thất bại trên phiên bản trước; một test shell-option control vẫn qua. Runner
chuẩn `scripts/run_tests.sh`, một worker, không retries, môi trường riêng.

| Lượt canonical,5file | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Trước nâng cấp |115|20|9|
| Sau nâng cấp |125|10|9|

Sau khi loại11failure thuộc các hợp đồng mới đã sửa,9failure cũ chung còn lại
vẫn là đường dẫn/MSYS, quyền POSIX và `/dev/zero` trênWindows. Candidate còn một
failure background-review/dedup: cùng ID đã fail trong baseline canonical của
lượt sửa trước3aeff8; chạy tập trung riêng trên cả bản trước/sau lần này đều qua.
Đây là bằng chứng test phụ thuộc thứ tự/process, chưa xác định nguyên nhân duy
nhất; **không xóa log, không nhận đã sửa dedup, không gọi fullsuite xanh**.
Không có failure mới chưa được đối chiếu theo điều kiện đã khóa.

## Trạng thái chạy và giới hạn

[Installed validation](../../runs/20261114T061928Z-hermes-read-paths-v2-778239/installation-and-activation-v1.json)
ghi đúng đường dẫn và hash module vừa import,6calls thật. Hermesgateway cũ
drain sạch, PID mới47768 tạo sau cài; Desktop log xác nhận kết nối lại ready.
Không memory-introspect worker để giả chứng minh từng import. SCAR-Xgateway35036,
nativeVLM39372 và trạng thái ALIF live vẫn giữ nguyên tại lúc kiểm.

Chỉ xác nhận các đường shell GitBash đã thử. Shell không hỗ trợ pipefail, remote
backend, encoding khó và file thay đổi đồng thời chưa có bảo đảm toàn diện.
Phép đọc gồm nhiều probe không trở thành snapshot nguyên tử. Quyền hoặc lỗi
được báo bởi công cụ vẫn cần đối chiếu; API thành công không tự là nhãn đúng.

Phần SNN còn thiếu ở đường goal input/error outcomes và executor học phục hồi;
[hidden common-probe](HIDDEN_BINDING_AND_COMMON_PROBE_20261114.md) chưa cho
lợi thế tổng quát đáng tin cậy. Không tăng neuron, train VLM/LoRA, đổi ALIF hay
promote weights nghiên cứu trong lượt này. Main/R13 và source của lượt8h554c19
không đổi; dùng monitor/final receipt riêng để đánh giá độ bền.
