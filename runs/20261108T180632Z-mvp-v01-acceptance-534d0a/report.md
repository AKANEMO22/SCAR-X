# MVP V0.1 measured demo

Product regression/demo after iterative development; 4 known signal types with unseen amplitude/noise variants. No claim of learned language, novel concepts, hidden representation advantage or solved continual learning.

| Stage | A accuracy | B accuracy | Updates in batch |
|---|---:|---:|---:|
| Before | 0.5000 | 0.5000 | 0 |
| After A #1 | 0.6406 | 0.4766 | 256 |
| After A #2 | 0.9922 | 0.4766 | 288 |
| After B #3 | 0.9531 | 0.6797 | 288 |
| After B #4 | 0.8984 | 0.7422 | 288 |

Restart and restore: exact. RSS=44183552 bytes; commit=61763584 bytes.

One A batch gained14.0625pp, below20pp target. Build iteration2 uses two A batches then two B batches; no neural rule/encoder changes.

Product acceptance passed; research forgetting/representation claims remain unproven.
