# RC operator D0 preflight — 16/11/2026

**STOP_DATA — v1 CLOSED trước neural execution.** Chạy D0 đúng một lần,
không thay seed/partition/quota hoặc giảm threshold để cứu family.

Source SHA `6cadbbe2113f94c88b44ecc87678adb5770bb552`; [run](../../runs/20261115T193825Z-rc-operator-pilot-v1-a76d1c/result.json).
D0 data-only: no neural/profile/fit, 0 model calls, 0 updates. No main/live/Hermes changes.

Reason: DEV valid-command shortage: IID/(0, 0, 0, 2, 3)/SWAP

## Kết quả có bằng chứng

- Codec/oracle contract PASS: 1,024 state roundtrips, 30,720 unique state-command
  inputs, 30,720 independent tuple-oracle comparisons. Đây không là neural PASS.
- TRAIN pool9,520; chọn512 examples, MOVE256/SWAP256, mỗi loại224changed+32noop;
  351 unique prestates/332 unique targets, max2 examples/prestate. Mọi min
  command coverage đạt; mỗi location/object ở input và target trong[20%,30%].
  Input frequencies22.07–27.73%, target21.875–27.539%.
- D0 dừng khi dựng primitive DEV ở initialstate `00023`: không đủ hai changed
  SWAP có target ngoài TRAIN_STATESET. Sequence generation, split audit hoàn
  chỉnh, control scoring, kNN quality/cost timing đều **NOT_RUN**.
- Backend CPU_DATA_ONLY, affinity một logical CPU. D0 through-report wall
  1.000374s, CPU0.59375s, process peakRSS42,872,832B; conservative tree peak
  bound52,948,992B gồm git provenance child. Đều dưới20s/300MB. Không GPU.
- [Manifest nguồn trước data](../../runs/20261115T193825Z-rc-operator-pilot-v1-a76d1c/manifest.json),
  [raw partitions/TRAIN](../../runs/20261115T193825Z-rc-operator-pilot-v1-a76d1c/raw-data.json),
  [kết quả/receipts/histograms](../../runs/20261115T193825Z-rc-operator-pilot-v1-a76d1c/result.json).
  Raw SHA256 `723a62407e5135d9b590a491164aad3e455db97ec850bbdda737e541f0881b20`.

## Postmortem riêng từ saved data, không rerun D0

[Audit stdlib](../../runs/20261115T193825Z-rc-operator-pilot-v1-a76d1c/audit-shortage.py)
đọc lại hash raw đã lưu và tính đúng mười SWAP targets bằng tuple comprehension
độc lập. [Receipt](../../runs/20261115T193825Z-rc-operator-pilot-v1-a76d1c/shortage-audit.json)
ghi2.478ms (<1s), 0 model/fit/update, `D0_rerun=false`. Đây là derived diagnostic
sau STOP, không gọi những targets tính lại này là predictions/raw bị thiếu.

| SWAP | Target từ `00023` | Changed? | Target trong TRAIN partition? | Hợp lệ cho changed DEV? |
|---|---|---|---|---|
| (0,1) | 00023 | Không | Không | Không, noop |
| (0,2) | 00023 | Không | Không | Không, noop |
| (0,3) | 20003 | Có | Có | Không |
| (0,4) | 30020 | Có | Có | Không |
| (1,2) | 00023 | Không | Không | Không, noop |
| (1,3) | 02003 | Có | Có | Không |
| (1,4) | 03020 | Có | Không | Có |
| (2,3) | 00203 | Có | Có | Không |
| (2,4) | 00320 | Có | Có | Không |
| (3,4) | 00032 | Có | Có | Không |

Trước target filter có7 changed SWAP;6 targets nằm trong TRAIN partition,
còn **1 valid changed SWAP, cần2**. Điều này đủ giải thích STOP_DATA tại state
được ghi, không cần chạy lại generator hoặc mở neural learner.

Giới hạn artifact: generator ném exception trước khi primitive DEV trả về,
nên raw chỉ có partitions và TRAIN; partial primitive records/missing-count
receipt chưa được giữ trong original D0 raw. `control-predictions.json` là
object rỗng. Postmortem không dựng lại whole DEV rồi gọi đó là original raw.
Các input/output histograms hiện có chỉ thuộc TRAIN. Compact episodic payload
512×14uint8=7,168B là arithmetic storage estimate, chưa đo kNN index/runtime,
latency hoặc accuracy.

## Diễn giải và STOP

Đã phát hiện một recipe/split không đáp ứng quota DEV đã khóa: partition theo
cyclic location orbit không bảo đảm mỗi evaluationstate có đủ admissible
changed SWAP edges. **Không bác bỏ ALIF, GRU hoặc neural composition** vì chưa
có neural calls. ALIF H128 representability, hidden gain, composition quality,
H5 và continual retention vẫn NOT_MEASURED. Không tăng H/lr/epochs hoặc thử
seed/partition khác dưới v1; main/Hermes/checkpoint cũ nguyên vẹn.

Bước tiếp theo được đề xuất, **chưa chạy/chưa protocol mới**: data-only
coverage-graph/admissible-evaluation-state builder để định nghĩa upfront tập
states có đủ operator edges và neutral trajectory coverage. Nó phải giữ
unseen-state rule, matched information và scientific thresholds; không sửa
v1 hậu nghiệm hoặc coi D0 PASS là neural GO.

Audit level: same-team independent tuple implementation và saved-data recount,
không external peer review. Original automated D0 receipt excerpt giữ dưới đây.

```json
{
  "stage": "primitive_dev",
  "train_receipt": {
    "pool_records": 9520,
    "selected_records": 512,
    "distinct_prestates": 351,
    "max_records_per_prestate": 2,
    "command_counts": {
      "MOVE:0:-1:0": 13,
      "MOVE:0:-1:1": 10,
      "MOVE:0:-1:2": 8,
      "MOVE:0:-1:3": 16,
      "MOVE:1:-1:0": 17,
      "MOVE:1:-1:1": 16,
      "MOVE:1:-1:2": 21,
      "MOVE:1:-1:3": 11,
      "MOVE:2:-1:0": 11,
      "MOVE:2:-1:1": 12,
      "MOVE:2:-1:2": 8,
      "MOVE:2:-1:3": 9,
      "MOVE:3:-1:0": 15,
      "MOVE:3:-1:1": 12,
      "MOVE:3:-1:2": 13,
      "MOVE:3:-1:3": 16,
      "MOVE:4:-1:0": 8,
      "MOVE:4:-1:1": 18,
      "MOVE:4:-1:2": 12,
      "MOVE:4:-1:3": 10,
      "SWAP:0:1:-1": 29,
      "SWAP:0:2:-1": 22,
      "SWAP:0:3:-1": 35,
      "SWAP:0:4:-1": 26,
      "SWAP:1:2:-1": 30,
      "SWAP:1:3:-1": 26,
      "SWAP:1:4:-1": 16,
      "SWAP:2:3:-1": 26,
      "SWAP:2:4:-1": 18,
      "SWAP:3:4:-1": 28
    },
    "quota_counts": {
      "MOVE:changed": 224,
      "MOVE:noop": 32,
      "SWAP:changed": 224,
      "SWAP:noop": 32
    }
  },
  "balance": {
    "state": {
      "counts": [
        [
          120,
          136,
          126,
          130
        ],
        [
          125,
          136,
          113,
          138
        ],
        [
          130,
          116,
          134,
          132
        ],
        [
          123,
          117,
          130,
          142
        ],
        [
          129,
          125,
          131,
          127
        ]
      ],
      "fractions": [
        [
          0.234375,
          0.265625,
          0.24609375,
          0.25390625
        ],
        [
          0.244140625,
          0.265625,
          0.220703125,
          0.26953125
        ],
        [
          0.25390625,
          0.2265625,
          0.26171875,
          0.2578125
        ],
        [
          0.240234375,
          0.228515625,
          0.25390625,
          0.27734375
        ],
        [
          0.251953125,
          0.244140625,
          0.255859375,
          0.248046875
        ]
      ],
      "within_registered_20_to_30_percent": true
    },
    "target": {
      "counts": [
        [
          125,
          134,
          112,
          141
        ],
        [
          124,
          137,
          130,
          121
        ],
        [
          122,
          127,
          134,
          129
        ],
        [
          123,
          117,
          140,
          132
        ],
        [
          130,
          130,
          119,
          133
        ]
      ],
      "fractions": [
        [
          0.244140625,
          0.26171875,
          0.21875,
          0.275390625
        ],
        [
          0.2421875,
          0.267578125,
          0.25390625,
          0.236328125
        ],
        [
          0.23828125,
          0.248046875,
          0.26171875,
          0.251953125
        ],
        [
          0.240234375,
          0.228515625,
          0.2734375,
          0.2578125
        ],
        [
          0.25390625,
          0.25390625,
          0.232421875,
          0.259765625
        ]
      ],
      "within_registered_20_to_30_percent": true
    }
  },
  "split_audit": null,
  "adequacy": null,
  "resources": {
    "elapsed_active_wall_seconds": 0.9986282000027131,
    "cpu_seconds": 0.59375,
    "peak_rss_bytes": 42872832,
    "peak_tree_rss_conservative_bound_bytes": 52948992,
    "provenance_child_peak_rss_bytes": 10076160,
    "provenance_child_memory_samples": 144,
    "current_rss_bytes": 42872832,
    "sampled_peak_private_commit_bytes": 31780864,
    "sampled_peak_tree_private_commit_bound_bytes": 33906688,
    "peak_pagefile_usage_bytes": 31780864,
    "sample_count": 227,
    "logical_cpu_count_allowed": 1,
    "affinity_mask_before": 1048575,
    "affinity_mask_used": 1,
    "caps": {
      "wall_seconds": 20,
      "peak_rss_bytes": 300000000
    }
  }
}
```

Strict split/quotas/seed were not changed. D0 FAIL closes this family without rescue sweeps.
D0 PASS requires another root source/dependency/backend review before any neural profiling/training.
Audit level: same-team independent tuple implementation and saved-data recomputation, not external peer review.
H128 representability, hidden contribution, H3 generalization, H5 and continual retention remain NOT_MEASURED.
