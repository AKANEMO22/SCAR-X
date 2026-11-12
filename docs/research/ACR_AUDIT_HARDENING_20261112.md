# Siết postprocess theo protocol, không đổi learner/gate

Reviewer độc lập phát hiện các điều kiện runner có kiểm nhưng auditor chưa
chấm lại: main ref, resource cap/measurement errors, số row/job đầy đủ và bộ
36stress/144diagnostic artifacts. Bổ sung readonly assertions trước PASS.
Missing cells chỉ hợp lệ khi đúng hai trajectory của preserved invalidnormpair;
không cho một control lỗi che các cell khác bị thiếu. Parent220filehashes/55
reuse được kiểm cả nguồn và bản sao.

Sửa mapping nhãn không thành công để đúng văn bản đã đăng ký: specialized
chỉ khi đúng một family rawgain; nếu nhiều rawgains nhưng broad/control gates
không đạt thì không tự gán specialized. Nhãn ACR_NO_GAIN trong trường hợp đó
có nghĩa không có gain đạt bộ điều kiện đã khóa, không phủ nhận rawtradeoffs
được ghi đầy đủ. Không đổi threshold, nomination, seed, data, profile hoặc
criteria numeric. Không thể dùng thay đổi nhãn này mở ADVANCE/P1/P2.

Các sửa chỉ nằm trong audit/report scripts, không nằm trong learner/source
lock của worker. Scientificprotocol cũ nguyên vẹn; allnegative/invalid/missing
evidence được giữ, không dùng quyết định sau nghiệm để chọn params.
