# SCAR-X Hybrid 1.0.0-rc.1

Bản cập nhật RC cho SCAR-X Hybrid **đã cài trên máy Windows/Hermes hiện tại**.
Đây là gói code nhỏ, tái sử dụng model/runtime đã có; không phải bộ cài độc lập
cho máy mới. Không chứa venv, trọng số model, hội thoại hoặc checkpoint cá nhân.

## Dùng thử

1. Giải nén vào một thư mục riêng. Giữ file ZIP và SHA256 để đối chiếu.
2. Chạy `Verify-Package.ps1`, sau đó `Install-Hermes.ps1` để cập nhật.
   `Install-Hermes.ps1 -Check` chỉ kiểm tra, không sửa source hay dừng dịch vụ.
3. Mở chat Hermes bình thường, tiếp tục chọn model **Scarx Hybrid** hiện có.
   Model ID `scarx-hybrid`, một gateway tại localhost:8775. Tab Debug chỉ quan sát.

Installer từ chối nếu lượt 8 giờ bảo vệ còn chạy hoặc chưa có receipt/audit hợp lệ.
Không dừng test để vượt điều kiện. Không xóa/move các thư mục dữ liệu nghiên cứu.
Gói chuẩn bị trước test kết thúc vẫn là candidate, chưa phải bản đã nghiệm thu.

Model đang dùng: Qwen3.5-0.8B nguyên BF16/F32, Whisper-small FP32, ALIF H128 FP32.
Các tính toán neural trên GPU hiện có; I/O, SQLite và điều phối trên CPU.
Không quantization/pruning/hạ precision hoặc tự bật LoRA.

## Khôi phục

`Rollback-SCARX.ps1` phục hồi code trước cập nhật khi file vẫn khớp SHA đã ghi.
Không ghi đè SQLite, xóa journal hoặc hoàn tác trải nghiệm người dùng phát sinh
sau cập nhật. Checkpoint đầy đủ được sao lưu riêng ở thư mục private
`local/hybrid-v1/release-backups/`; không nằm trong gói.

`Start-SCARX.ps1` / `Stop-SCARX.ps1` điều khiển đúng gateway đã đăng ký, có kiểm
PID/create-time/cmdpath; không dừng Hermes hay các tiến trình khác.
Nếu installer báo source/dependency/pin Hermes không tương thích, dừng và giữ
thông báo, không sửa manifest hoặc xóa dữ liệu người dùng để ép chạy.

## Phạm vi RC

Hai sửa chính: truy xuất toàn câu hỏi trong giới hạn 6.000 ký tự và loại
action verify khi checker không hỗ trợ. Chúng là sửa runtime, không phải bằng
chứng SNN đã học kỹ năng mới. Cửa nghiên cứu/retention không được thay.

Xem `KNOWN_LIMITATIONS.md`, `MANIFEST.json`, `ASSETS.json` và `DEPENDENCIES.json`.
SHA/version/codeoverlay có trong manifest; không xuất token hay cấu hình bí mật.
