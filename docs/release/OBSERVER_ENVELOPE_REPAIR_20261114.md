# Sửa observer bỏ qua công cụ đọc file

Mục đích là sửa pipeline đang dùng trong sản phẩm; không mở benchmark nghiên
cứu hoặc quy tắc học mới. Lượt 8 giờ vẫn đã hủy, không chạy lại.

## Lỗi và tác động

`agent/turn_api_request.py` của Hermes truyền `request` tạo bởi
`ApiRequestHooksMixin._api_request_payload_for_hook`: `{method, body}`.
Observer SCAR-X đọc `request.tools`, nên `file_tool` bị false dù model được
cấp `read_file`. Nhánh chẩn đoán đọc lại file trong CWD project vì thế không
chạy khi tool lặp lỗi. Các fixture cũ truyền tools ở root đã bỏ sót hợp đồng này.

Đã sửa observer đọc `request.body.tools`, vẫn hiểu fixture cũ và từ chối quyền
probe nếu envelope/tools không hợp lệ. Không thay phân quyền, model, ALIF,
reward, cập nhật synapse hoặc trạng thái riêng.

## Kiểm chứng

- 27 unit/contract checks qua, gồm đúng envelope, tối đa một probe trong turn,
  chống bật probe từ payload sai và các hành vi incident/feedback trước đó.
- `scripts/check_hermes_observer_contract.py` chạy process riêng với Hermes
  đang cài, **ApiRequestHooksMixin thật, discover_plugins, invoke_hook,
  handle_function_call** và thư mục project tự tạo. Không giả kết quả đọc file.
- Bản trước: 3 lần đọc `wrong/README.md` trả not-found, 3 public observations,
  **0 probe**. Bản sửa: cùng chuỗi đọc lỗi, **1 probe** đọc được README ở đúng
  project. Probe vẫn ghi `semantic_resolution_verified: false`.
- Không gọi VLM, web hoặc học mới. HOME/project/quan sát của fixture ở vùng
  `local/observer-envelope-validation/20261114T134617760288Z`, không gửi hoặc
  trộn vào hội thoại người dùng. Failed unit collection ban đầu được giữ log,
  bản cuối qua sau khi sửa tên tham số fixture trùng tên dành riêng của pytest.
- Phát lại đúng 3 kết quả đọc lỗi qua `ToolCallGuardrailController` đang cài,
  với ngưỡng SCAR-X hiện có (exact failure 3): lần gọi thứ tư bị `block`, không
  được thực thi. Đây là kiểm bộ chặn bằng dữ liệu thật, không phải một lượt
  model tự suy luận hoặc bằng chứng đã học tránh lỗi.

[Biên bản máy đọc được](evidence/OBSERVER_ENVELOPE_REPAIR_20261114.json).

## Triển khai

`scripts/install_hermes_observer_envelope.py` kiểm exact before SHA, backup và
atomic CAS đúng một file plugin. Chỉ reload gateway Hermes, giữ worker VLM CUDA
49308 và gateway SCAR-X 52192. Không đổi config hoặc dùng restart để giả cải tiến.

Đây là sửa **khả năng quan sát/chẩn đoán ở runtime**. Việc tìm được nguồn có
thể đọc chưa chứng minh hiểu mục đích file, tự học nguyên nhân hoặc áp dụng kỹ
năng sửa lỗi lâu dài. Các kết luận hidden SNN/retention trước đó không thay đổi.
