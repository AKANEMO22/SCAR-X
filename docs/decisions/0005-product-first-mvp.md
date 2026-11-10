# ADR0005 — Product-first và SCAR-X MVP V0.1

Ngày2026-11-09. Nguồn: yêu cầu người dùng trong tệp `ee0bb112-d1fa-472f-8931-bf892e36fe9b/pasted-text-1.txt`, phê duyệt triển khai MVP, sửa tích hợp và kiểm thử.

## Thay đổi so với quyết định trước

Người dùng thay workflow research-first bằng **Build → Run → Interact → Learn → Test → Improve → Repeat**. ADR0004 docs-only là phạm vi lịch sử, đã được thay bằng quyền triển khai MVP. Các giới hạn research không được dùng để chặn xây một sản phẩm thử nghiệm yếu nhưng trung thực.

Giữ SNN/ALIF/local learning/CPU-first và mục tiêu2GB; không pretrained LLM/cloud API/model lớn. Giữ ngưỡng forgetting3pp/worst5pp và kết quả Phase2/2.1 trong hồ sơ nghiên cứu; **không coi việc ứng dụng chạy được là đã đạt các ngưỡng đó**. NO-GO cho tăng quy mô vẫn hiệu lực.

## V0.1 thực hiện

- Workbench web chỉ bind127.0.0.1, Python stdlibHTTP, frontend tĩnh không CDN, NumPy/psutil hiện có.
- Dùng LifelongLearner score_ema và PlasticCore48ALIF/864synapse trọng số. Không thay learning rule/topology để làm demo.
- Tác vụ A/B có cấu trúc; prediction lấy neural softmax mean4stochastic passes. Rule dùng kiểm nhãn/chấm điểm; memory chỉ lưu/truy xuất ở màn riêng.
- Checkpoint bundle gồm learner/plasticity-v1, replay, RNG/counters; SQLite transaction commit cùng trải nghiệm/history/active pointer. Failed update khôi phục object trước đó; process interruption nạp generation cuối đã commit.
- LocalJob Windows giới hạn committedmemory của process2GB nếu attach thành công; batches/body/arrays/time bounded và monitor hiển thị số thật. Đây không phải cap toàn bộ RAM browser/OS, không đo điện năng.
- Feedback builtin-rule-v1 kiểm target; human confirmation không tự biến thành evidence. Không dạy natural-language facts hoặc lấy output của model làm nhãn thật.
- Score suy giảm sau học mới vẫn hiển thị; không auto-rollback vì chưa qua research forgettinggate. Rollback cho lỗi state/persistence/resource hoặc do người dùng chọn checkpoint.

## Giới hạn và bài học build

Demo đầu256update chỉ tăngA14,06pp, không đạt target demo20pp. Build iteration2 dùng hai đợtA rồi hai đợtB, giữ rule/encoder và threshold; ghi rõ thay budget demo trong acceptance protocol, không gọi đó là nghiên cứu xác nhận mới. Đầu ra này chưa chứng minh hidden representation, generalization khái niệm mới hoặc solved continuallearning.

Save/load tiếp tục đọc legacy checkpoints mà không sửa artifact/run cũ. Không initGit hoặc di chuyển cấu trúc. Run QA dùng data-dir riêng và có report tái lập; hướng dẫn vận hành duy nhất là README, roadmap duy nhất là masterplan.
