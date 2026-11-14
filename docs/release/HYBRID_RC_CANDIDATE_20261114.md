# Hybrid RC — gói đã chuẩn bị ngày 14/11/2026

**Đây là biên bản RC1 tại thời điểm chuẩn bị.** Sau đó người dùng yêu cầu hủy
test 8 giờ và ưu tiên sản phẩm; xem [RC2 activation](RC2_ACTIVATION_20261114.md).
Giữ số liệu/hash RC1 bên dưới làm lịch sử, không dùng trạng thái đang chạy cũ
làm điều kiện chặn yêu cầu mới.

**Đã tạo và kiểm gói code; chưa cài hoặc nghiệm thu live.** Thí nghiệm small
mixed 8 giờ `32cd13` vẫn chạy. Không dừng worker/native để ép phát hành.

## Artifact

- File: `artifacts/releases/SCAR-X-Hybrid-1.0.0-rc.1+2f1fc57c0f59.zip`.
- Dung lượng: **787.328 byte**, 48 file payload; thêm manifest, hướng dẫn,
  installer/start/stop/verify/rollback và asset/dependency pins.
- SHA256: `cee4df090421ed02622f7157768f53e2a677d0980f677a091ad8eec76dd7f035`.
- Git SHA nguồn gói: `2f1fc57c0f590673ebf0274a77f3daa0a1418329`.
- Phạm vi: overlay cho deployment Windows/Hermes đang cài. Không chứa model,
  runtime binary, venv, SQLite, token, checkpoint, hội thoại hoặc dataset.
- Chứa đúng hai candidate runtime đã audit: full-query retrieval và verifier
  capability guard. Source đang phục vụ chưa được thay.

## Kiểm chứng đã hoàn tất

1. **27 safety/P0 checks qua**: ZIP/path/SHA/tamper, bảo vệ worker, lỗi giữa
   overlay, rollback giữ dữ liệu mới, từ chối sửa tiếp, checkpoint transport,
   exact recipe/audit và hành vi hai candidate. Hai ca giả mạo manifest có SHA
   tự khớp vẫn bị chặn bởi source tin cậy trước khi chạy code giải nén. JUnit ở
   `local/hybrid-rc-safety-and-p0-anchored-final-20261114.xml`.
2. Giải nén ngoài checkout, kiểm toàn manifest; mọi module SCAR-X được import
   từ payload, không import Torch. Closure build receipt là file
   `.verification.json` cạnh ZIP.
3. **74 regression fixtures qua** trên code thực trong payload: HTTP/tool
   transport, video binding, SQLite retrieval, checkpoint/load/rollback, RNG/
   replay, học local và lifecycle. Native CPU parity test được loại rõ; không
   gọi native VLM/ASR mới hoặc đo chất lượng chat/video trong lượt này.
   ALIF có forward/cập nhật thật trong fixture CPU cô lập; không dùng hoặc sửa
   synapse của phiên Hermes đang phục vụ. Không đánh đồng 0 native calls với
   0 phép tính neural.
4. `manage.py --check` gọi đúng registration và **từ chối vì worker 8 giờ còn
   chạy**, trước stop/write. Monitor sau đó xác nhận actual PID 46940 và 36 file
   source còn khớp. Lượt kiểm lại có 37 video cycles, 7 checkpoint, chưa final PASS.
5. Kiểm read-only thực tế **64 asset pins, 1.157 file Hermes và dependency** đều
   khớp. Plan overlay đúng 2 file; không start/stop model hoặc đổi database.

Chạy lại phép kiểm trước terminal bằng:

```powershell
& 'local/hybrid-v1/runtime-env/Scripts/python.exe' scripts/verify_hybrid_rc_candidate.py 'artifacts/releases/SCAR-X-Hybrid-1.0.0-rc.1+2f1fc57c0f59.zip'
```

Verifier lưu log/JUnit/result trong thư mục mới ở `local/release-verification/`;
không cài, stop hoặc restart. Bản kiểm đầu là `20261114T125311649576Z`, kiểm lại
sau khi thêm đối chiếu source tin cậy là `20261114T125649675911Z`. Cả hai giữ
nguyên receipt. Kiểm asset thêm lưu cạnh receipt thứ hai.
Sau terminal, không dùng phép kiểm pre-terminal này thay cho audit/acceptance.

## Việc còn lại của release

1. Worker thực sự kết thúc; đọc receipt và chạy độc lập
   `scripts/audit_small_mixed_endurance.py` trước mọi activation.
2. Chỉ nếu đạt cửa engineering đã đăng ký, áp dụng hai patch trong một lượt
   maintenance có backup đầy đủ/CAS; kiểm startup, đúng CUDA và state còn nguyên.
3. Nghiệm thu chat Hermes thật: gửi/ACK, workspace/projects, tool giới hạn và
   dừng được, phiên dài không lộ compaction, search, ảnh/video, một native GPU
   worker và restart/save/load/rollback. Ghi mọi lỗi chưa đạt.

Đây là prepared RC, không phải production hoặc bằng chứng học kỹ năng lâu dài.
Giữ nguyên các báo cáo SNN chưa vượt RAG và forgetting chưa đạt.
