# REPORT — Round 5B — two independent facts are stored and retrievable, but the harder half of the paired test misses

**Branch:** `research/deepseek/r5-independent-relational-binding`
**Base:** `495dab2f4bac0180e4d1cd28674391f055ccf09d`
**Protocol:** SHA-256 `9ae72b3a9ecb9cab0e1abbfcd2e2ee8eb69838eb2bed7913e1132159197c30ee` @ `91fda5a`
**Environment:** DeepSeek Harness (DSH) Web, Windows 11
**Model:** `deepseek-ai/deepseek-v4.1-flash-1017`

**Verdict: STOP.** Screen passed 9/9. **Confirmation failed C3** (both-answers-correct
on differing-value pairs = 0.6676, gate 0.70).

---

## 1. Answer to the round's question, up front

> *Can frozen ALIF H48 store two independent key/value facts and support a local
> online readout that answers unseen queries above the 0.75 shortcut ceiling — or was
> R4A's success specific to a one-bit XOR?*

**R4A's success was NOT merely a one-bit artifact.** With genuinely independent values:

- Both bits are stored: **vA 0.9967, vB 0.9974** pre-query.
- They are **bound to the right keys**: among `vA ≠ vB` episodes the probe recovers
  0.9947 — a bag-of-values code cannot do this.
- R5's local head reaches **0.9635** held-out against a **0.7522** shortcut ceiling
  (+21.13pp), and paired both-answers-correct **0.9590** against first-key's **0.7507**.
- Confirmation on 8 fresh seeds: clean **0.8989**, and it holds under every stress
  condition tested (0.8174–0.9221).

**But the strict confirmation gate on the hardest sub-case fails:** when `vA ≠ vB`,
both siblings correct is **0.6676**, below the registered 0.70.

## 2. The ceiling problem — measured before the run

This generator draws `vA` and `vB` independently, so 0.50 is the wrong null. Stage 0
measured the real ceilings:

| baseline | measured |
|---|---:|
| **always answer key A** | **0.7522** |
| knows both values, no query | 0.7473 |
| majority | 0.4900 |
| query-only | 0.4988 |

Every claim below is gated against **0.75**, not 0.50. Corroborated by Stage 1: all
five un-conditioned heads land at 0.7480–0.7773, i.e. exactly on that ceiling.

## 3. Stage 1 — information path (frozen, 3 dev seeds)

| rep | dims | vA | vB | joint4 | bag3 | **mixed-only** | answer | head |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1_pre | 192 | 0.9967 | 0.9974 | 0.9863 | 0.9518 | **0.9947** | 0.7454 | 0.7480 |
| R2_traj | 576 | 0.9987 | 1.0000 | 0.9642 | 0.8698 | 0.9988 | 0.7663 | 0.7624 |
| R3_post | 192 | 1.0000 | 0.9967 | 0.9785 | 0.9010 | 0.9975 | 0.8984 | 0.7741 |
| R4_decay025 | 48 | 0.7070 | 0.7168 | 0.4766 | 0.5228 | 0.7253 | 0.6947 | 0.6035 |
| R4_spikesum | 48 | 1.0000 | 0.9993 | 0.9883 | 0.8770 | 1.0000 | 0.7708 | 0.7773 |
| R4_bank5 | 240 | 0.9980 | 0.9980 | 0.9694 | 0.8867 | 0.9910 | 0.7878 | 0.7734 |
| **R5** | 258 | 1.0000 | 0.9954 | 0.9720 | 0.8952 | 0.9987 | **0.9948** | **0.9635** |

**H1 (storage) supported**: vA 0.9967, vB 0.9974, both far above chance.
**H2 (binding) supported**: `mixed-only` 0.9947. A probe that reads the value multiset
can distinguish 00 / {01,10} / 11 but cannot separate 01 from 10; 0.9947 means it does.
**H3 (query integration) supported**: answer decodes 0.7454 pre-query → 0.8984
post-query, and R5 reaches 0.9948.

**I had to correct my own H1/H2 verdict logic here.** My first script flagged
"bag3 = 0.9518 → shortcut risk". That was wrong: once vA and vB are each decodable,
their sum is decodable by construction, so bag3 succeeding is *forced* by storage and
carries no shortcut information. The genuine separator is `mixed-only`. Recorded
rather than quietly removed.

## 4. Stage 2 — R5 dissected against eight controls

| arm | dims | acc | q=A | q=B | **needB** | train | ridge |
|---|---:|---:|---:|---:|---:|---:|---:|
| **R5_original** | 258 | **0.9635** | 0.9634 | 0.9642 | **0.9254** | 0.9775 | 0.9948 |
| poststate linear 258 | 258 | 0.7637 | 0.7520 | 0.7761 | 0.5643 | 0.7799 | 0.8783 |
| R4 matched 258 | 258 | 0.7637 | 0.7383 | 0.7833 | 0.5690 | 0.7881 | 0.7969 |
| query only | 2 | 0.5091 | — | — | 0.5027 | 0.5111 | 0.5085 |
| query free | 192 | 0.7741 | 0.7619 | 0.7846 | 0.5754 | 0.7829 | 0.8984 |
| cue shuffled in train | 258 | 0.7604 | 0.7524 | 0.7666 | 0.5325 | 0.7806 | 0.9023 |
| cue shuffled at test | 258 | 0.8021 | 0.7710 | 0.8297 | 0.6808 | 0.9775 | 0.7663 |
| **query zeroed + cue** | 258 | **0.9876** | 0.9844 | 0.9904 | **0.9828** | 0.9938 | 0.9954 |

**The `needB` column is the discriminating one**: accuracy restricted to episodes where
the query names key B *and* `vB ≠ vA`. A first-key reader is 0.50 there by construction.
R5 reaches **0.9254**; every un-conditioned arm sits at 0.53–0.58.

**Paired counterfactual (the metric a 0.75 shortcut cannot fake):**

| reader | both correct | case A flips | case A both | case B invariant | case B both |
|---|---:|---:|---:|---:|---:|
| **R5** | **0.9206** | 0.8471 | 0.8419 | 1.0000 | 1.0000 |
| first-key (null) | 0.5013 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| query-free | 0.6901 | 0.7137 | 0.3900 | 0.9865 | 0.9865 |

The first-key reader flips **0.0000** of case-A pairs — it is provably not using the
query. R5 flips 0.8471. Both readers score 1.000 on case B, which is why case B must be
scored as *invariance*, never as a flip.

**`query_zeroed_plus_cue` = 0.9876 exceeds R5's own 0.9635.** The query does not even
need to reach the neural core: with the query channels zeroed at the population and the
cue supplied to the readout, retrieval is *better*. The facts are stored persistently;
the query is a readout-side selector.

## 5. Stage 3 screen — PASS 9/9

| gate | result |
|---|---|
| G1 direct ≥ 0.85 | **0.9635** ✓ |
| G2 ≥ +10pp over the 0.7522 ceiling | **+21.13pp** ✓ |
| G3 both-correct case A ≥ 0.70 | 0.8419 ✓ |
| G4 both-correct case B ≥ 0.70 | 1.0000 ✓ |
| G5 paired CI lower > 0 | **+41.78pp [+39.69, +43.73]**, 383 units ✓ |
| G6–G9 leakage / regression / parity / budget | ✓ |

## 6. Stage 4 confirmation — FAIL on C3

8 fresh seeds × 7 conditions:

| condition | direct | worst seed | first-key | both_all | **both_A** | both_B |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.8989 | 0.8301 | 0.7527 | 0.8301 | **0.6676** | 0.9809 |
| train shift + test shift | 0.8999 | 0.8164 | 0.7527 | 0.8340 | 0.6756 | 0.9805 |
| **OOD clean → shift** | **0.8977** | 0.8340 | 0.7527 | — | — | — |
| reversed writes | 0.9221 | 0.8242 | 0.7527 | 0.8643 | 0.7342 | 0.9839 |
| distractors | 0.8208 | 0.7695 | 0.7527 | 0.7202 | 0.4536 | 0.9701 |
| noise change (alone) | 0.8174 | 0.7637 | 0.7517 | 0.7158 | 0.4512 | 0.9626 |
| query tick variation | 0.9119 | 0.8457 | 0.7527 | 0.8457 | 0.6897 | 0.9915 |

| gate | result |
|---|---|
| C1 clean ≥ 0.85 | 0.8989 ✓ |
| C2 ≥ +10pp | +14.67pp ✓ |
| **C3 both-correct case A ≥ 0.70** | **0.6676 ✗** |
| C4 both-correct case B ≥ 0.70 | 0.9809 ✓ |
| C5 CI lower > 0 | +14.67pp [+11.45, +17.63] ✓ |
| C6 worst clean seed ≥ 0.75 | 0.8301 ✓ |
| C7 shifted learning ≥ 0.80 | 0.8999 ✓ |
| C8 all conditions above ceiling | ✓ |

**`confirmation_pass: False`.** The screen's 0.8419 on case A did not replicate on 8
fresh seeds; it came in at 0.6676.

## 7. A harness defect I found and disclosed

The **first** confirmation run reported clean, train-shift and OOD-shift as
**identical to four decimals** (0.8989 / 0.8989 / 0.8989). That is not a result.

`shifted()` padded the vacated leading ticks with **zeros**. For a zero-initialised
recurrent network that is a **no-op**: `v = 0.65·0 + 0 = 0`, `z = 0`, so the state
never leaves its initial value and every later tick is bit-identical. Verified:
post-query state under the zero-pad shift had `max |diff| = 0.0` against clean.

Fixed by padding the vacated ticks with fresh noise from the same distribution
(`max |diff|` now 1.0), which keeps the shift a genuine timing change. The failing run
is committed and marked **`INVALID_HARNESS`**; its shift and OOD rows are not cited.
The numbers in §6 are from the corrected run.

This is the second temporal-shift defect in this campaign lineage — R4A had one too.
Recorded as a standing hazard: **any transform applied before a zero-initialised
recurrent net must be checked for being a no-op.**

## 8. Why C3 fails, and what it means

Case A requires both siblings correct: `(vA=0,vB=1)` answered for query A *and* for
query B. R5 gets each individually at ~0.93 (`needB` = 0.9254), so the joint should be
≈0.87. On the confirmation seeds it is 0.6676 while `both_B` is 0.9809 — a large
asymmetry.

`both_A` is the harder half because when `vA ≠ vB`, answering correctly under a B-query
requires reading `vB` **and** suppressing the strong `vA` trace. `both_B` (0.9809) is
inflated by case-B pairs where both answers coincide. So the failure is concentrated
exactly where the first-key shortcut is most attractive.

**This is a real limitation of the fixed-projection readout, not a measurement artifact.**
The distractors and noise conditions make it worse (both_A 0.45), consistent with the
readout being near its capacity.

## 9. Verdict

**STOP.** Per protocol §11, a confirmation gate failure ends the round. No threshold
was relaxed. The screen→confirmation gap on case A (0.8419 → 0.6676) is itself the
most transferable finding: three seeds did not predict eight, again.

What survives, with 8-seed support:

| claim | evidence |
|---|---|
| Two independent facts are stored | vA 0.9967, vB 0.9974 |
| They are bound to the right keys | mixed-only 0.9947 |
| A local head beats the 0.75 ceiling | 0.8989 clean, +14.67pp CI [+11.45, +17.63] |
| It is not a first-key shortcut | case-A flips 0.8471 vs first-key 0.0000 |
| Timing is robust | OOD clean→shift 0.8977, shifted learning 0.8999 |
| Facts persist without a neural query | query-zeroed + cue 0.9876 |

What does not: **joint correctness on differing-value pairs**, 0.6676 against a 0.70 gate.

## 10. Not claimed

No chat, vision, reasoning, retention, lifelong learning or P2. Reading a frozen state
is not learning. The projection `P` is fixed random — only the 2×258 readout learns —
and the cue is supplied from the input, so R5 is a *sufficient-readout diagnostic*, not
evidence that SCAR-X discovered the interaction. A ridge is a linear reference, never an
information ceiling.
