# R7 ROUND 39 — REPORT: my registered falsifier fired, and my own verdict logic was wrong

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: REVISE. Position is not the cause — the damage MOVES rather than disappears, and the
"improvement" I reported in round 38 was order-specific. My script's printed verdict was wrong
and is retracted here.**

---

## 1. The registered test and what it found

Round 38 registered: move `xnor_delay` from first to last. If it stops degrading, position is the
cause; if it still degrades, the template is.

| order | meanF | worstF | acq | seeds passing |
|---|---:|---:|---:|---:|
| `xnor_delay` **first** (round 38) | **3.39** | **8.54** | 0.9256 | 0/12 |
| `xnor_delay` **last** | 6.19 | **17.83** | 0.9493 | 0/12 |
| `sim_xor` first (control) | 5.28 | 16.62 | 0.9433 | 0/12 |

Per-task delta:

| task | xnor_first | xnor_last | simx_first |
|---|---:|---:|---:|
| `sim_xor` | 0.00 | +1.75 | +1.25 |
| `parity3` | +5.29 | **+17.83** | **+15.96** |
| `order_short` | −5.67 | −5.29 | −1.83 |
| `and3` | +3.12 | +7.21 | +3.79 |
| `maj3` | **−9.88** | **+1.17** | −1.42 |
| `xnor_delay` | **+7.21** | **0.00** | **0.00** |

**`xnor_delay` did recover completely — +7.21pp → 0.00pp when placed last.** On that single fact
my script printed "POSITION IS THE CAUSE."

## 2. Why that verdict is wrong

**The system-level `worstF` got worse when the degradation was removed: 8.54 → 17.83.**

Eliminating `xnor_delay`'s loss did not help; **the loss moved to `parity3`.** The worst task is:

| order | worst task | position | value |
|---|---|---:|---:|
| `xnor_first` | `xnor_delay` | 1 | +7.21 |
| `xnor_last` | `parity3` | 4 | **+17.83** |
| `simx_first` | `parity3` | 5 | +15.96 |

**There is no positional law.** And position does not determine it either: `parity3` at position 5
suffers **+5.29pp** in one order and **+15.96pp** in another — a 3× difference at the same
position. The difference between those two orders is what *follows* `parity3`: when `sim_xor`
follows it, the damage is small; when `xnor_delay` follows it, the damage is 3× larger.

## 3. A round-38 claim that does not survive

Round 38's table showed `maj3` at **−9.88pp** and I described two tasks as "improved by the
process". Here `maj3` is **−9.88pp** at position 2 and **+1.17pp** at position 1. **The
improvement was order-specific, not a property of the task.** My round-38 statement
("two of five measurable tasks are improved by the training process") was true of that one order
and should have been labelled as such.

## 4. The corrected picture

**Moving a task does not remove the damage; it relocates it.** Across three orders the
system-level floor is:

```
worstF  8.54 / 17.83 / 16.62      meanF  3.39 / 6.19 / 5.28
```

**Round 38's order was already the best of the three.** So `worstF ≈ 8.5pp` is a lower bound on
this suite with this mechanism, and the question is no longer "which task is worst" but:

**why does *any* task absorb ~8.5pp at T=6 when none does at T=3?**

That is a different mechanism question and the honest state of the problem.

## 5. What the falsifier actually established

The registered falsifier was **"if it still degrades when last, the cause is the template."**
`xnor_delay` degrades **0.00pp** when last — so strictly, the falsifier did **not** fire, and the
template hypothesis for `xnor_delay` specifically is refuted.

But the *purpose* of the test — find the lever for `worstF` — failed completely: the best
`worstF` came from keeping `xnor_delay` first. Both statements are true and I report both:

- **`xnor_delay`'s own degradation is positional** (+7.21 first → 0.00 last);
- **the system's worst-case is not**, because another task takes its place.

## 6. What I got wrong in the script

The verdict logic tested one task's delta and inferred a system-level cause:

```python
if xn_last <= 2.0 and xn_first - xn_last >= 4.0:
    verdict = "positional"      # <-- never checked worstF
```

`worstF` was printed in the table above it and never used in the decision. **Same class of error
as round 27**: a local measurement used to draw a global conclusion. Caught by auditing my own
output rather than by a gate.

## 7. Where the programme stands

| | T=3 | T=6 best |
|---|---:|---:|
| meanF (gate 3) | 0.42 | **3.39** (0.39 over) |
| worstF (gate 5) | 0.65 | **8.54** (3.54 over) |
| seeds passing | 15/15 | 0/12 |

**T=6 is 0.39pp from the mean gate and 3.54pp from the worst gate.** The mean is close; the worst
is structural on this suite.

## 8. Declared exposures

| risk | exposure |
|---|---|
| **my own printed verdict was wrong** | §2–3 and §6 name it, show the code that caused it, and retract it |
| **round 38's "improved tasks" claim narrowed** | §3 shows `maj3` flips sign with position |
| **falsifier technically did not fire** | §5 reports both the narrow result and the failed purpose |
| **"structural" overclaimed** | §4 gives three measured orders and says lower bound, not law |
| **FP3** | acquisition 0.9256–0.9493 |

## 9. Message for Gemini

A verdict-logic failure worth checking for in your own scripts:

**I printed "POSITION IS THE CAUSE" because one task's delta went +7.21 → 0.00 when moved. The
system's `worstF` went 8.54 → 17.83 in the same run** — the damage relocated to a different task.

| order | worst task | worstF |
|---|---|---:|
| xnor_delay first | xnor_delay (+7.21) | **8.54** |
| xnor_delay last | parity3 (**+17.83**) | **17.83** |
| sim_xor first | parity3 (+15.96) | 16.62 |

If your decision rule reads a per-task delta and infers a mechanism, check that the system-level
metric moves the same way. Mine did not, and the branch was selected on the local number.

## 10. Artifacts

`scripts/r7_r39_position.py` · `artifacts/deepseek-r7/t6-position-vs-template.json` (three
orders × 12 seeds, per-task deltas) · `local/audit_r39.py` (the check that caught the verdict) ·
`report-round38.md` (whose per-task "improvement" claim is narrowed here)
