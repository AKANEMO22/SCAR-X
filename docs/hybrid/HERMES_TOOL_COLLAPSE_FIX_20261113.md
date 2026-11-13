# Hermes — đóng kín nhóm công cụ13/11/2026

Người dùng báo một dải chữ/log vẫn lộ dưới summary sau khi thu gọn.

Nguyên nhân ở`ToolRun` trongHermes`fallback.tsx`: ngay cả khi persistedOpen=false
(người dùng đã đóng), điều kiện`live && !expanded` vẫn mount`ToolRunTicker`.
Ticker là một cửa sổ1dòng cố ý để xem hoạt động live, nên nó cắt một phần row
thành dải chữ “hé” thay vì ẩn hoàn toàn. Đây không phải dữ liệu model tự sinh.

Sửa: chỉ cho livepreview tự động khi người dùng chưa explicitclose. Bấm đóng
ẩn toàn bộ ticker và toolrows, vẫn giữ summary; khi có log mới cũng không
tự mở lại. Bấm mở lại vẫn có đủrows. Liveness, summary, approvals và delegation
không bị tắt. Không sửaCSS toàncục hoặc xóa nội dunglog.

Patch`integrations/hermes/patches/manual-tool-collapse-v1.patch` giữ cả source
và regressiontest. Test được chạy đỏ trên bản cũ: sau close vẫn có DOMticker.
Patch dùng zero-context unifieddiff (`git apply --unidiff-zero`); đối chiếu đúng
sourcehash trong receipt trước áp lại. Reverse-check trên source đã sửa qua.
Sau sửa25toolgroup/delegate/chrometests qua; Prettier và scopedESLint qua.
Renderer build dùng toolchainHermes pinnedNode26.12.1/pnpm12.15.0, không đổi
dependencies hoặc ghiđè những sửaPython từ tích hợpSCAR-X trước đó.

`scripts/install_hermes_collapse_renderer.py` chỉ cài renderer ở bản desktop
đang chạy, backup trước từng fileđổi, giữ mọi hashedassets cũ.4filesđổi;
Electronmain/preload/buildmetadata giữ nguyênhash. Reloadrenderer bằngCtrl+R,
không xóa session. NativeGUI xác nhận nhóm“Used7tools”
đóng chỉ cònsummary, không dòngchild bịhé. Liveupdatebehavior được kiểm trong
test; không tạo modeljob/benchmarkgiả để kiểm một lỗiUI.

Rollback: dùng`local/hermes-collapse-renderer-current.json` lấybackup và restore
`index.html`/`renderer-manifest.json`; assetcũ vẫn ở đó. Sourceoriginal backup
ở`local/hermes-tool-collapse-current.json`. Source/rootHermes trước sửa là
3b42157858; các thay đổi`agent_init.py`/`tool_search.py` sẵn có không bị thay.
Không đổiSCAR-X weights, main hoặc các thí nghiệmpeer đang chạy.

Kiểmtra sau reload phát hiện guard4GB đã dừngAIworkers trong thời gian thao tác
giao diện. Đã dùng lifecyclehelper có kiểmidentity để khôi phục riêng gateway
Hybrid; trạng tháireadytrue,3.718GB/trần4GB. Không nới guard hoặc bỏ receipt
vi phạm cũ. Bổ sung`runtime_recovery.json` ghi lần khôi phục, tách khỏi receipt
deploymentUI ban đầu; không tuyên bố model chạy liên tục không bị ngắt.
