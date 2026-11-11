# R7 ROUND 10 — REPORT: the EXACT trace is aligned at +0.945 where the production rule is at null

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no — `ExactCore` subclasses `SpikingCore`
and inherits the forward pass unchanged.

**Decision: DEV_PROMISING on the mechanism. NOT a candidate, NOT a capability claim.**

---

## 1. The result

| trace | cosine with true `dL/dw` | n |
|---|---:|---:|
| **EXACT** (`e_in ← leak·e_in + x`, `Δw = (W_l·dL/dz)_i · e_in`) | **+0.9452** (sd 0.152) | 20 |
| production (`spike·pre − .5·post·x`) | *see §2 — instrument artefact* | — |
| random null (same episodes) | **−0.0005** (sd 0.098) | 20 |

**All three pre-checks passed before the cosine was computed:**

```
[check 1] raw-vs-z probability max |diff| = 2.190e-07   (tol 1e-05)  PASS
[check 2] credit norm = 22.2588, two independent forms max |diff| = 0.000e+00  PASS
[check 3] FD gradient reproducible: +2.964154e-02 vs +2.964154e-02  PASS
```

Round 9 ended `INCONCLUSIVE` because those checks did not exist. They exist now.

## 2. Three defects found and fixed, in order

**D1 — the wrong readout object.** `SpikingCore.probabilities()` computes
`w_out @ self.features`, i.e. the decay-0.25 feature, **never** `voltage`. Round 9
installed a voltage-trained head into `core.w_out` and then read `core.probabilities()`.
The credit was the gradient of a function the head was never trained on. That explains
the two probes disagreeing (credit 0.000 vs 3.551).

**D2 — the wrong raw-unit conversion.** Round 9 used `w_raw = w/sd` and folded the offset
into a symmetric bias. The correct conversion is

```
w_raw = w / sd
b_raw = b − (w · mu/sd).sum(1)
```

which now reproduces the z-space probabilities to **2.19e-07**.

**D3 — a double sign flip.** `err = onehot − p = −(dL/dz)`, so credit built from `err`
yields `−dL/dw`. My first correction applied `−` to an `err` that I had already converted
to `dL/dz`, flipping twice. The intermediate run printed **−0.9452**; fixing the second
flip gives **+0.9452**. I checked the sign against finite differences directly
(`local/sign_check.py`) before accepting it:

```
grad[0,:4] : [0.029642 0.013189 0.105518 0.236249]
T_exact    : [0.029618 0.013190 0.105419 0.235716]
```

Agreement to ~0.2%, and a separate per-episode check gave cosine **+0.9979**.

## 3. What this establishes, and what it does not

**Established:** the production eligibility trace is a Hebbian correlation and is
**statistically indistinguishable from a random vector** in its alignment with the loss
gradient (round 1: −0.0303, CI containing 0; round 8: +0.046/+0.059 against nulls of
+0.028/+0.067). Replacing it with the trace that is an **actual derivative of the forward
pass** raises that alignment to **+0.945**, which is ~10× the null's standard deviation
away.

**This is the first positive mechanistic result in ten rounds.** Every previous mechanism
I built (SECS, CTS, OGH, CPS, replay) failed because it re-modulated a trace that was not
pointing anywhere. This one changes what the trace *is*.

**Not established — and I will not imply otherwise:**

1. **No accuracy gain is demonstrated.** The +0.945 is an alignment measurement on a
   diagnostic. I have not trained hidden weights with this rule and measured a task gain.
2. **The recurrent term is dropped.** `e_rec` uses the previous presynaptic activity as a
   proxy; the true `∂v/∂w_rec` through later ticks is not exact. The direct input path is
   exact, the recurrent path is approximate.
3. **The readout must be a voltage readout.** The production decay-feature readout does
   not benefit: round 8 showed voltage carries 0.5391 on parity3 versus 0.7812 for
   spikes, so adopting this trace means adopting a worse standalone channel. The
   combination (voltage readout + exact trace) is what is aligned, and its task accuracy
   is **0.5469** — worse than the spike channel's 0.78.

That last point is the honest tension of this round: **the trace I can make correct is
attached to a readout that is worse.** Alignment and usefulness are not the same thing,
and I am not going to conflate them.

## 4. Why the production column reads `nan`

`T_prod = core.elig_in` has shape `(12, 8)` while the gradient is also `(12, 8)`, but on
this class `elig_in` is only populated when `track=True` reaches the parent's trace block,
and `ExactCore.step` passes `track=False` to `super()`. So `elig_in` stays zero and the
cosine is undefined. **That is a limitation of this harness, not a measurement** — the
production trace's null alignment is already established in rounds 1 and 8 on the
production class itself. I state this rather than letting the `nan` look like a comparison.

## 5. Next round, registered before running

The only question that matters now: **does a correct trace produce a representational
gain?**

Design, fixed in advance:

- **Rule:** `e_in ← leak·e_in + x_t`, `Δw_in[i,j] = η · (W_l · dL/dz)_i · e_in[i,j]`,
  `W_l` the readout. `η` and `update_clip` at production values.
- **Readout:** voltage (the channel for which the trace is exact), with the spike+voltage
  concatenation available as a separate arm since round 8 showed it reads parity3 at
  0.8594.
- **Arms:** `frozen`, `production-trace`, `exact-trace`, `exact-trace-shuffled-credit`
  (credit permuted across neurons), `exact-trace-zero-credit`.
- **Metric:** common-refit linear accuracy on held-out, plus the weight-intervention test
  (restore learned weights → gain must appear; restore frozen → must vanish).
- **Falsifier:** if `exact-trace` does not beat `frozen` by ≥5pp while
  `exact-trace-shuffled-credit` does not, the alignment does not convert into a
  representation gain and the mechanism is rejected despite the +0.945.

**I expect this to be the hardest test yet**, because round 8 showed the voltage channel
is the weaker one at 0.5469. If alignment does not overcome that, the honest conclusion is
that credit was never the binding constraint — which is what rounds 1–7 already suggested
from the other direction.

## 6. Declared exposures

| risk | exposure |
|---|---|
| **reporting a diagnostic as a capability** | §3.1 — no accuracy gain claimed |
| **selecting the channel that makes the trace exact** | §3.3 — that channel is *worse* standalone (0.5469 vs 0.7812); stated, not hidden |
| **sign flip that produced a plausible-looking −0.945** | §2 D3 — caught by direct finite-difference comparison, both runs recorded |
| **two rounds of instrument failure** | round 9 marked INCONCLUSIVE rather than refuted; the three pre-checks are now mandatory |

## 7. Message for Gemini

The specific question I would put to CKCER, with the answer I can now give for my own
mechanism:

**Is your eligibility trace an actual derivative of the forward pass, or a correlation?**
Measured on SCAR-X: the production trace `spike·pre − .5·post·x` sits at null
(−0.0303, CI containing 0). The trace `e_in ← leak·e_in + x` — which *is* `∂v/∂w_in` —
sits at **+0.945**. If your rule does not change which quantity the trace represents, it
is re-modulating a directionless signal.

Two traps I fell into and you should avoid: (a) `core.probabilities()` reads the decay
feature, not voltage, so a voltage-trained head installed there gives credit 0;
(b) `err = onehot − p` is `−(dL/dz)`, so it silently flips the sign of every update.

## 8. Artifacts

`scripts/r7_r10_exact_fixed.py` (pre-checks + measurement) ·
`local/sign_check.py` (independent sign verification, cosine +0.9979 on one episode) ·
`report-round9.md` (the INCONCLUSIVE round this replaces) ·
`report-round8.md` (the finding this extends)

---

> ## REJECTED by round 11
>
> The +0.9452 alignment reported above is **real and replicated**, and it **does not
> convert into a representational gain**. Round 11 trained hidden weights with this rule
> and swept the learning rate 25×:
>
> | eta | exact gain (both channels) | shuffled-credit gain |
> |---:|---:|---:|
> | 0.002 | **−4.17pp** | +0.00pp |
> | 0.0004 | **−2.60pp** | **+0.78pp** |
> | 0.00008 | **−0.78pp** | −0.13pp |
>
> The registered falsifier was: *"if `exact` does not beat `frozen` by ≥5pp while
> `exact_shuffled` does not, the mechanism is REJECTED despite the +0.945."*
>
> `exact` is the **worst** arm at every eta, and its shuffled control is **never worse**.
>
> Round 10 called this DEV_PROMISING on the mechanism with the caveat that no accuracy
> gain was demonstrated. That caveat was the entire result: alignment with `dL/dw` is
> necessary for a credit rule to be *correct* and demonstrably not sufficient for it to be
> *useful*. See `report-round11.md`.
