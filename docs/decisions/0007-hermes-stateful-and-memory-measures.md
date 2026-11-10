# ADR0007 — Hermes thích nghi với SCAR-X, phép đo trí nhớ riêng

2026-11-09. Người dùng yêu cầu sửa Hermes theo SCAR-X, không sửa neural system
để thỏa giả định LLM. Lỗi thật: Hermes chặn model có context4096 trước khi gọi
backend, vì guard 64K phục vụ agent có prompt/tool lớn.

Quyết định: thêm chế độ named-provider `stateful_chat` vào Hermes. Giữ input
G1≤2048ký tự và buffer host4096, không khai khống64K. Chế độ loại agent scaffold,
tools, search controller, memory prompt, fallback và MoA trên đường gửi; giữ
transcript ledger. Binding theo model+endpoint, provider agent khác không opt-in
vẫn bị guard như cũ. G1 vẫn một core, CPU-first, ≤2GB; không đổi neural code,
weights, học cục bộ, topology hay ngưỡng research.

Real AIAgent/SessionDB/HTTP/SSE đã kiểm; không chỉ kiểm model picker. Source
backup và diff nằm trong integration; checkout Hermes có nhiều thay đổi sẵn nên
chỉ vá anchor cụ thể, không reset. Đã restart backend/desktop để nạp module mới
và kiểm hai lượt trong cùng phiên, không lỗi64K; footer Backend state và popover
input/memory chưa đo. Lượt tiếng Việt còn sai/lặp; sửa tích hợp không chứng minh
general chat. Renderer build/deploy có backup riêng, không đổi Electron main/preload.

Context length ngừng là mục tiêu trung tâm, không có nghĩa input/trạng thái/
capacity là vô hạn. Dùng [bảng đo memory/learning](../research/09_MEMORY_LEARNING_MEASURES.md):
quality theo tuổi/interference, working-memory horizon, retention/forgetting,
new-skill acquisition, transfer, capacity có điều kiện và resource costs.
Các giá trị mới chưa đo; không gộp thành điểm trí thông minh giả. Giữ gate cũ.
