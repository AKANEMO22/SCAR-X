# Tiếp tục DEV sau gián đoạn credits — không thay protocol khoa học

Người dùng gửi chỉ thị literature-first và “tiếp tục”. Worker950718bd của run
20261112T073944Z-acr-dev-v1-f947fb đã mất do phiên bị gián đoạn, không còn PID.
Audit độc lập xác nhận55cell/55folder complete,55checkpoint và419.840prediction
entries hợp lệ; không có job đang học dở. Không xóa/reset reservation cũ.

Giữ nguyên experiments/acr-dev-v1.md:10ABprofiles, mở4Cprofiles chỉ sau AB nếu
không staticuniform nào đạt broadchecks; globalnomination, matched physicalnorm,
shuffle, equalCPU, mechanism/stress, cùng3DEVseed. Không confirmation/P2/main.
Chỉ tiếp tục sau literature-reference đã đủ72primarycells để đối chiếu equations
và kết quả. Reference không tạo broaddirect/P1pass trên dữ liệu hiện có; không
phát minh router mới hoặc chỉnh alpha/coefficient/threshold để cứu ACR.

Runner mới có tên riêng run_acr_resume.py, source original không sửa. Nó:
- xác minh source manifest gốc và PID đã chết;
- tạo reservation/run mới, giữ parentmanifesthash/sourcecommit;
- copy55cell nguyên byte, skip toàn bộ học/calibration/forward của các cell đó;
- tái tạo generators để so exact với frozenarray gốc rồi copy datasetbytes;
- chỉ train cell chưa có; không tự ghi đè hoặc restart incomplete cell;
- giữ equations/optimizer/RNG/init và toàn bộ tiêu chí đã đăng ký trước scores.

Budget cũ150min không reset: lấy9000s trừ snapshotelapsed1132,9984881s và thêm
60s dự phòng shutdown chưa quan sát; còn7807,0015119s maximum active worker.
Thời gian worker chết không phải thời gian compute. Historicalpeak135,35MBRSS/
178,63MBcommit là lowerbound snapshot, không tự gọi finalpeak. Resume đo riêng
Windowscap/RSS/commit cả observer8771 mới, không can thiệp8767 hoặc Hermes.

55cell cũ được đánh dấu reuse ở manifest, không tính là55trainingjob mới. Final
audit cần chấm đủ combinedgrid và kiểm copiedhash; missing/invalid controls
chặnadvance. Nếu runtime budget hết, giữpartial và báo đúng, không mở lượt khác.
