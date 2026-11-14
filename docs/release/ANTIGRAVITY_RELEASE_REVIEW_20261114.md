# Antigravity CLI — review phát hành ngày 14/11/2026

User yêu cầu tận dụng CLI để tăng tốc. Đã xác nhận CLI `agy` 1.5.8 và gọi đúng
`--model gemini-3.8-flash-high --effort high`; không dùng low hoặc tự hạ model.
Ponytail không được cung cấp trong danh sách tool của phiên này.

| Job | Kết quả thực tế | Cách sử dụng |
|---|---|---|
| `120312Z…ff0c07` | Review rộng timeout 5 phút, không có kết luận | Không tính là code review hay approval |
| `121409Z…df6833` | Hoàn thành checklist thiết kế; nhận diện model high | Dùng để rà các cửa release, xác minh từng finding |
| `123212Z…d7f629` | Review code timeout 3 phút, không có kết luận | Không tính là approval; kiểm tra code độc lập tiếp tục |

Chỉ gửi source hoặc mô tả công khai, không gửi hội thoại, video, token, dữ liệu
nghiên cứu hay checkpoint. Raw job/log ở `local/`, không nằm trong gói phát hành.

## Đối chiếu độc lập

- Chặn restart/sửa source trước khi worker kết thúc: yêu cầu đúng. Tuy nhiên
  36 file được **khóa bằng protocol/hash**, chưa có bằng chứng Windows báo
  `ERROR_SHARING_VIOLATION` như CLI suy đoán.
- Truy xuất bị cắt ở **16 từ**, không phải 16 ký tự. Hai candidate P0 đã có
  audit; gói phải khớp recipe chính xác, không chỉ chứa chữ VERIFIED trong báo cáo.
- Hai module readout hiện là NumPy, không có Torch. Import closure vẫn cần kiểm
  trên gói giải nén ngoài checkout; không mô tả đây là lỗi Torch đã tồn tại.
- Allowlist, helper màn hình/search, loại dữ liệu riêng, kiểm PID/create-time
  và rollback chỉ code là các yêu cầu hữu ích, đã đưa vào implementation/tests.
- Kết luận `blocked` của review checklist không có nghĩa goal bị blocked hoặc
  sản phẩm đã được chấm. Cửa live acceptance vẫn phải qua sau khi cài.

## Bằng chứng kiểm tra code

`tests/test_hybrid_rc_package.py` kiểm dữ liệu riêng/path/ZIP, tamper SHA,
worker đang chạy, lỗi giữa overlay, code rollback giữ trải nghiệm mới, từ chối
sửa tiếp và transport checkpoint. Kiểm thêm exact P0 recipe/audit/candidate.

Kết hợp hai bộ P0 shadow: **25 passed**, không train, không gọi neural/VLM mới,
không thay source khóa của lượt 8 giờ. JUnit private:
`local/hybrid-rc-safety-and-p0-20261114.xml`. Kết quả fixture an toàn cập nhật
không thay cho nghiệm thu state neural và chat Hermes thật sau cài đặt.
