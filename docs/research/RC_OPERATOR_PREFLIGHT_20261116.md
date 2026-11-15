# RC operator D0 preflight — chuẩn bị 16/11/2026

**SOURCE_READY_FOR_ROOT_COMMIT; NOT_RUN.** Root chỉ mở D0 data-only, CPU1,
20s, peak RSS≤300MB. Chưa neural/profile/fit. Không tạo dataset hoặc run trước
source/protocol commit.

- Protocol: [rc-operator-pilot-v1](../../experiments/rc-operator-pilot-v1.md),
  status ROOT_REVIEWED_D0_ONLY.
- Pure world/oracle/codec/data/control source:
  [rc_operator_world.py](../../scripts/research/rc_operator_world.py).
- Runner/provenance/resources/independent tuple audit:
  [rc_operator_preflight.py](../../scripts/research/rc_operator_preflight.py).

Sau root commit, runner yêu cầu exact `--expected-source-sha` và các reviewed
files phải tracked, không khác HEAD. Không tunable seed/split/quota CLI. Mỗi
run create-exclusive, source snapshot, manifest trước data và raw partial
evidence ngay cả STOP_DATA/STOP_ADEQUACY/STOP_RESOURCE. Không retry cứu lỗi
khoa học. D0 PASS cũng chưa mở neural execution.

Nguồn được kiểm tĩnh khi soạn, chưa execute/compile/import. Audit/run receipts
và kết quả hiện **NOT_MEASURED**; 0 model calls, 0 fits, 0 updates.
