# Literature-first — reference ALIF H48, 12/11/2026

**REFERENCE_MIXED**, integrity **PASS**. Nominee toàn cục `eprop_matched`; chưa P1/P2 hay production.
Không đổi forward ALIF, topology, input D8, readout, optimizer hoặc main/Hermes. Đây là adapter chuyển phương trình về SCAR-X, không tái lập accuracy paper gốc.

## Lượt thực nghiệm và kiểm toán

Run `20261112T080533Z-literature-reference-dev-v1-839e07`, source commit `a3b79ea66cb6676f8b03a22b0cb907d158e5b2a2`. Protocol [đăng ký](../../experiments/literature-reference-dev-v1.md).
Ba seed DEV đã dùng 140011/140017/140023; 1.024 update mỗi primary cell. TRAIN1.024/refit256/ID512/OOD512; calibration32TRAIN. Không sử dụng confirmation bank để tune.
Đủ 132 jobs:72primary,12nominee-shuffle,48equalCPU. Wall 766.97s. Audit 8.80s, không có cập nhật học.
Chấm lại 675,840 prediction entries, 132 checkpoint round-trip, 2,112 direct replay và 1,536 probe bằng eigensolver độc lập.
Peak owned-tree RSS 133.71MB; commit 175.26MB; trần2.000MB. Windows cap receipt: `{'enforced': True, 'limit_bytes': 2000000000, 'kind': 'Windows per-process committed memory; shared pages monitored separately'}`. Observer process được tính theo PID; không có iGPU/GPU.
ResourceMonitor flag hard_cap_enforced không được truyền từ cap receipt; dùng receipt thực và measured RSS/commit, không đồng nhất committed memory với lượng swap.

## Direct output và tài nguyên

| Method | Binding | XOR | Distractor | Order | OOD direct TB3primary | CPU train TB/job | Mảng model |
|---|---:|---:|---:|---:|---:|---:|---:|
| own | 85.42% | 91.86% | 56.97% | 100.00% | 74.41% | 2.742s | 29,472B |
| stop | 82.62% | 99.35% | 55.66% | 99.61% | 75.28% | 3.171s | 29,472B |
| eprop_matched | 78.58% | 98.24% | 60.03% | 100.00% | 75.33% | 3.005s | 29,472B |
| eprop_aligned | 69.92% | 99.54% | 55.01% | 99.93% | 70.94% | 2.785s | 29,472B |
| braintrace | 68.62% | 99.22% | 53.32% | 99.93% | 70.03% | 3.341s | 29,472B |
| readout_only | 62.96% | 82.75% | 52.67% | 91.73% | 63.48% | 2.803s | 29,472B |

CPU là actual single-stream cho own/stop/reference; không phải dual-stream ACR endpoint. Mảng model không phải process RAM. Bảng OOD tách từng nhóm bên dưới để không che regression.

| Method | Binding ID probe | Binding OOD direct/probe | XOR ID probe | XOR OOD direct/probe | Distractor ID probe | Distractor OOD direct/probe |
|---|---:|---:|---:|---:|---:|---:|
| own | 86.98% | 83.59%/85.03% | 96.68% | 85.55%/89.19% | 56.97% | 54.10%/56.18% |
| stop | 85.29% | 82.88%/85.03% | 99.54% | 86.65%/90.04% | 63.41% | 56.32%/62.76% |
| eprop_matched | 82.75% | 79.30%/82.62% | 99.61% | 87.17%/92.19% | 61.98% | 59.51%/58.66% |
| eprop_aligned | 70.77% | 68.42%/66.02% | 99.80% | 91.21%/90.62% | 58.07% | 53.19%/55.86% |
| braintrace | 74.22% | 67.97%/71.61% | 98.96% | 89.13%/88.48% | 55.79% | 52.99%/54.69% |
| readout_only | 62.30% | 67.25%/67.38% | 83.27% | 71.55%/68.62% | 53.84% | 51.63%/55.60% |

### Equal-CPU secondary — 0,5s học, không thay primary

| Method | Binding direct | XOR direct | Distractor direct | Order direct | Số episode min–max |
|---|---:|---:|---:|---:|---:|
| own | 58.33% | 67.97% | 51.95% | 96.94% | 159–212 |
| stop | 56.25% | 86.20% | 52.80% | 95.31% | 136–171 |
| eprop_matched | 54.43% | 86.13% | 51.17% | 98.18% | 156–215 |
| readout_only | 57.10% | 73.05% | 52.02% | 87.30% | 161–214 |

Các điểm thấp được giữ nguyên. CPU cutoff kiểm ở cuối episode, có overshoot
được ghi raw; số update khác nhau nên không thay thế đối chứng1.024update.
Không có lợi thế CPU đa năng được chứng minh.

Bảng per-seed v2 tra readout cùng budget. Bản đầu tra readout1.024update cho
cả equalCPU; trong thí nghiệm này probe của frozen hidden exact ở hai budget
nên không có giá trị gain nào bị đổi. Hai bản được giữ để ghi rõ sửa cách
lookup; primary metrics và quyết định audit v1/v2 không đổi.

## Hidden learning và các gate không được miễn

| Task | Nominee direct | Probe gain vs readout [95% mô tả] | Direct causal drop revert-both | Δdirect vs better endpoint | OOD preserved |
|---|---:|---:|---:|---:|---|
| strict_binding | 78.58% | +20.44đ [+7.23đ;+30.66đ] | +25.00đ | -6.84đ | False |
| xor | 98.24% | +16.34đ [+11.72đ;+19.73đ] | +44.60đ | -1.11đ | True |
| distractor_binding | 60.03% | +8.14đ [-0.20đ;+12.70đ] | +9.44đ | +3.06đ | True |

Có 0/3 family đạt cải thiện direct kèm probe/causality theo ruleDEV; 3/3 đạt probe margin so shuffle. Protected order: `True`.
Ba seed chỉ cho thống kê khám phá: exact sign-test hai phía có p nhỏ nhất0,25. Không thể gọi significance/confirmation. Probe gain không thay direct skill; full-revert recovery được kiểm exact với readout-only.
Chưa có actual-update-norm matched control cho reference ở vòng này: loss, physical weight-change norms và optimizer giữ được ghi lại nhưng không chứng minh lợi ích độc lập với amplitude.

## Những gì đã kế thừa và điều vẫn thiếu

- `ALREADY_SOLVED_IN_LITERATURE`: eligibility/Jacobian theo intrinsic multi-state dynamics, phân tích e-prop factorization và SI path importance.
- `REPRODUCED_IN_SCARX`: NumPy adapters, upstream trace AST parity, local/full surrogate oracle và strict checkpoint/cold resume trong phạm vi technical tests.
- `PARTIALLY_SOLVED`: hướng credit trên frozen fixtures; local block vẫn bỏ cross-neuron recurrent sensitivities, learning signal và normalizer còn xấp xỉ.
- `UNRESOLVED`: một checkpoint tích lũy nhiều kỹ năng, readout overwrite, task-agnostic retention, distractor/OOD bền vững, chat/vision đa năng.
- `POTENTIAL_NEW_CONTRIBUTION`: chưa đủ bằng chứng gán nhãn này cho ACR; không tính việc chuyển phương trình có sẵn thành tính mới.

Chi tiết: [e-prop](literature/eprop.md), [BrainTrace](literature/braintrace.md), [reuse matrix/SI/EWC/DECOLLE/DGN/R9b](literature/continual-and-local-signals.md).

[Diagnostic đóng băng sau học](literature/trained-credit-oracle.md) có180row,
45checkpoint,0learningupdates. Trong trạng thái đã học, BrainTrace có trường
hợp XOR alignment thấp hơn nhiều so fullsurrogate oracle; fixture ban đầu gần1
không bảo đảm tương tự ở mọi trạng thái. Hai phép đo dùng input/signal khác
nhau nên không quy cosine giảm chỉ cho việc học. Các method có finalweights
khác nhau nên không dùng bảng cosine như causalranking về chất lượng.
SI helper qua10test toán nhưng chưa có retention/P2. Explicit consolidation dùng task boundaries là boundary-assisted; dùng local credit thay loss gradient là biến đổi cần kiểm riêng. EWC/Fisher và replay chưa được chứng minh ở protocol mới. DGN đổi neuron dynamics nên ngoài primary forward đang khóa; DECOLLE pin hiện có điều khoản phi thương mại, không copy code.

## ACR và bước tiếp theo

ACR cũ bị gián đoạn sau55cell hợp lệ; còn110AB cell và chưa C/norm/shuffle/equalCPU. Không gọi partial study là ACR thắng/thua. [Audit gián đoạn](literature/acr-interruption-audit.md) xác nhận12own-reference cells khớp exact với alpha0 cũ. Không vứt bỏ raw hoặc chạy lại confirmation.
Quyết định mở rộng ACR phải dựa cả reference DEV này, direct/representation riêng và matched controls. Không bảo vệ router bằng cách nới gate. P1 vẫn cần2/3primary đủ direct65%, representation CI/matched shuffle, OODNI/protected; P2 vẫn acquisition65%/final65%/mean forgetting3đ/worst5đ theo seed. Chưa mở fresh confirmation/P2.

## Tái lập và bàn giao

Branch `codex/scarx-literature-first-20261112`, base `950718bd410371f8ef15dab2d78d722b04ab506d`. Main trước/sau `e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541`.
```powershell
python scripts/prepare_literature_references.py
python scripts/build_native.py
python -m pytest -q -p no:cacheprovider tests/test_literature_eprop.py tests/test_literature_braintrace.py tests/test_literature_si.py tests/test_literature_snapshots.py tests/test_literature_oracle.py tests/test_native.py
python scripts/run_literature_reference_dev.py
# Runner reserves once; reproductions use a new isolated checkout, never delete old reservation to overwrite evidence.
python scripts/audit_literature_reference_dev.py --run runs/20261112T080533Z-literature-reference-dev-v1-839e07
```
Raw `runs/20261112T080533Z-literature-reference-dev-v1-839e07/`; per-seed table `per-seed-results-v2.csv`, audit `audit-v2.json`, checkpoint/initial/history ở từng job. Source hashes, upstream commit/license và data hashes đi cùng run. Observer http://127.0.0.1:8772 chỉ đọc outputs thật, không mô phỏng lời nghĩ.
Chưa merge; final delivery commit/push được xác minh riêng. Không đổi production core/checkpoint/SQLite/Hermes.
