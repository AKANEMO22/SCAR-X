# Bản lưu SCAR-X trước dự án mới

Ngày 16/11/2026. Các nhánh bên dưới đã được đẩy lên GitHub trước khi làm
sạch nội dung được Git quản lý trên main. Lịch sử Git được giữ nguyên;
main mới là một commit tiếp nối, không phải force push hoặc xóa lịch sử.

| Bản lưu | Commit |
|---|---|
| [Main SCAR-X cũ](https://github.com/AKANEMO22/SCAR-X/tree/archive/scarx-main-20261116) | `37749041ef45d54951558fdeb0c3695616b47cb5` |
| [Nghiên cứu đến lúc chuyển dự án](https://github.com/AKANEMO22/SCAR-X/tree/archive/scarx-research-20261116) | `62e1f6b6069305b780162159ebeaac367a6dc53f` |

[Bàn giao nghiên cứu](https://github.com/AKANEMO22/SCAR-X/blob/archive/scarx-research-20261116/docs/research/ARCHIVE_HANDOFF_20261116.md)
ghi rõ kết quả hoàn tất, negative results và phần chưa chạy. Các trạng thái
FAIL/STOP chưa được chuyển thành PASS; không có model nghiên cứu được deploy.

Nhánh, tag và worktree cũ vẫn tồn tại. Model, runtime và dữ liệu riêng trong
các thư mục bị Git bỏ qua được giữ trên máy; chúng không tự trở thành nội
dung đã upload. Đặc biệt `local/` chứa các worktree và runtime cần bảo toàn.
Hermes backend đã được tắt theo yêu cầu; việc chuẩn bị main không bật lại nó.

Để xem main cũ trong một thư mục riêng mà không thay đổi main hiện tại:

```powershell
git worktree add ..\scarx-legacy archive/scarx-main-20261116
```

Main hiện tại chỉ chứa khung chuẩn bị và tài liệu điều hướng archive.
Yêu cầu dự án mới sẽ xác định cấu trúc và kiến trúc tiếp theo.
