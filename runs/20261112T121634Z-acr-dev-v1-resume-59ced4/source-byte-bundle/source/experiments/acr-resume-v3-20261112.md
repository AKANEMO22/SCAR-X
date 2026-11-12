# Phục hồi sau mất worker lần hai — ngoại lệ hành chính công khai

Người dùng yêu cầu “tiếp tục”. Phiên tool cũ không còn worker PID2124 hoặc
observer; không kill/can thiệp process khác. Runac9bba giữ203cell hoàn tất và
1folder `distractor_binding-140023-random_alpha-equal_updates` chỉ có initial
checkpoint. Log cuối cùng ghi128updates của cell đó; không có final checkpoint,
history hoặc result để resume optimizer. Không mô tả đây là0updates.

Đăng ký ngoại lệ phục hồi hẹp đối với câu “không restart incomplete cell” trong
resume cũ: replay đúng cell thiếu state MỘT LẦN từ initialcheckpoint byte-exact,
cùng seed/data/order/source/RNG/alpha và1.024updatebudget. Không thay candidate,
không chọn outcome, không dùng kết quả đã thấy để chọn hyperparameter. Giữ
nguyên folder/log cũ; coi128 là lowerbound và1.024 là upperbound số update bị
mất. Đây là compute thừa vì gián đoạn, không independent trial/seed mới.
Nếu cell phục hồi lỗi kỹ thuật/numerical, lưu lỗi và không tự chạy lại nó.

Copy203completecells812files byte-exact sang unique v3run/reservation, skip
mọi calibration/forward/learn của chúng. Manifest khóa parentmanifest và812file
hashes; auditor kiểm recursive provenance55→203 và file initial của lostcell.
Giữ toàn bộ scientificprotocol/gates; sourcelearner/readout/optimizer không đổi.
No freshconfirmation/P2/main/Hermes/liveSQLite/port8767.

Thời gian v3 không reset ngân sách: từ v2limit7805,0015119s trừ last observed
elapsed của v2 và120s uncertainty allowance (bằng perjob wall ceiling), gồm
timer/cap từ trướcpreflight/copy. Timestamp snapshot là lowerbound, không phải
thời điểm chết được giám sát; allowance không chứng nhận exact tổngwall9000s.
Báo riêng observedactivecompute và reserveduncertainty, không claim exact
globalcap của quãng mất giám sát. V3 cap<=remaining,2GB RSS/commit/tree và
120s/job vẫn enforce. Không giảm samples/seed/gates để chạy nhanh hơn.
