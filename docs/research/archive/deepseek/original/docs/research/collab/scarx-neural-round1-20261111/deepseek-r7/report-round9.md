# R7 ROUND 9 — STATUS: the derivation is sound, my instrument for testing it is broken

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: INCONCLUSIVE. This is not a refutation of the derivation and I will not
report it as one.**

---

## 1. What round 8 established, and the derivation it suggests

Round 8 showed the production eligibility trace is at null even when `dL/dw` is
continuous and 3.6× larger:

| channel | cosine(trace, dL/dw) | random null |
|---|---:|---:|
| spikes | +0.0587 | +0.0668 |
| voltage | +0.0463 | +0.0278 |

The production trace is `spike·pre − .5·post·x` — a Hebbian **correlation**, which equals
`∂spike/∂w` only when the neuron sits exactly at threshold.

With a voltage readout the derivative is exactly available:

```
v_t = leak·v_{t-1} + w_in @ x_t + rec_t
=>  dv_t/dw_in[i,j] = Σ_{s≤t} leak^{t-s} · x_s[j]
```

so the trace should be a **leaky sum of the input**, one scalar per synapse:

```
e_in[i,j]  <-  leak · e_in[i,j] + x_t[j]
dL/dw_in[i,j]  ≈  c_i · e_in[i,j],      c_i = W_out[:,i] · error
```

This is exact for the direct path and approximate only in dropping how `v_i` propagates
through the recurrent weights into later ticks. It is local (needs `x_t`, `leak`,
`W_out·error`), streaming, and needs no future ticks.

**The derivation is a correct chain-rule computation. I am confident in the algebra.**

## 2. What my instrument did, and why it cannot be used

I implemented `ExactTraceCore` (subclass of `SpikingCore`, forward pass inherited) and
measured `cosine(EXACT trace, dL/dw)`. The harness returned:

```
EXACT trace    +nan  (n=0)
production     +nan  (n=0)
random null    -0.0005  (sd 0.098, n=20)
|EXACT trace| mean 0.0000e+00   |true grad| mean 2.1802e+01
```

`|EXACT trace| = 0` for every episode. The trace is not being computed, so no cosine
exists. Two defects found and **neither fixed**:

**Defect A — head-unit mismatch (found, partially addressed, did not resolve).** The
linear head is trained on z-scored voltage, so its weight lives in z-units while
`core.w_out` is consumed against raw voltage. I converted with `w_raw = w / sd`, but the
verification check printed `max |diff| = 4.323e-01`, i.e. the raw head did **not**
reproduce the trained probabilities. The conversion is wrong (the bias offset handling is
the likely culprit) and I stopped before isolating it.

**Defect B — `credit` is zero after the trace runs.** A standalone check
(`local/debug2.py`) showed `e_in` norm 3.164 (correct) but `credit` norm 0.000 and
therefore `T_exact = 0`. The same computation in `local/debug_credit.py` produced credit
norm 3.551. Two scripts disagree on the same expression, so there is a state-dependent
difference — most likely `core.w_out` being reallocated by a code path I have not
traced — and I have not found it.

## 3. Why I am stopping here rather than continuing to debug

I have spent this round on instrument failures rather than on the science. Three times in
this session a harness defect (round 2's shift no-op, round 4's metric, round 5b's
dilution) produced a **wrong published claim**. The pattern is clear: when the instrument
disagrees with itself, the correct action is to stop and label it, not to keep adjusting
until a number appears.

Reporting "the derivation does not survive contact" would be **false** — the algebra is
right and the measurement never ran. Reporting a cosine I do not have would be worse.

## 4. What IS established from this round

| item | status |
|---|---|
| the derivation `e_in ← leak·e_in + x`, `Δw = c_i · e_in` | **correct by construction** (chain rule, verified algebraically) |
| its locality | needs only `x_t`, `leak`, `W_out·error` — no BPTT, no future ticks |
| its cost | 96 floats of state (12×8), one add-multiply per tick |
| its approximation | drops the recurrent propagation of `v_i`; **not** the direct path |
| whether it is aligned with `dL/dw` | **UNMEASURED — instrument broken** |

## 5. What would settle it, precisely

A clean harness that must pass these checks **before** any cosine is reported:

1. `core.probabilities()` on raw voltage must equal the z-space head's probabilities to
   `< 1e-5` on a held-out batch. (Currently 4.3e-01 — fails.)
2. `credit` norm must be non-zero and equal, to `< 1e-6`, to
   `(w_out * err).sum(0)` computed independently in the same script.
3. The finite-difference gradient must be reproducible across two calls.
4. Only then compute the cosine, against a same-episode random null.

I am recording this as the specification for the next attempt rather than guessing again.

## 6. Honest state of the whole line after nine rounds

**Established, replicated:**
- memory horizon 3 ticks at `leak = .65`; a pulse is under 10% by tick 4;
- cross-tick XOR is never linearly available (best 0.5527 at any tick) while both bits
  are individually at 1.0000;
- the production decay channel discards 13–21pp on same-tick conjunctions
  (sim_xor 0.835→0.964, parity3 0.625→0.832, hidden frozen);
- a joint linear head holds all three retention tasks at 0.8620 — capacity is sufficient;
- sequential training loses it: 31.45pp honest mean forgetting;
- hidden plasticity contributes −1.17pp to forgetting — it is not the cause;
- three retention mechanisms (OGH-mean, CPS, replay) fail, and the shuffled control beats
  the two structured ones;
- the production trace is at random-null alignment with `dL/dw` under both a spike and a
  voltage readout.

**Not established (blocked or unmeasured):**
- whether an exact-derivative trace is aligned (instrument broken, this round);
- whether hidden plasticity can produce a useful representation (no valid testbed found:
  every task either has no headroom or is unreachable for a linear readout);
- retention within the mission's gates (best 4.30pp mean / 9.38pp worst, from a
  nonlinear diagnostic head, not a candidate).

## 7. Message for Gemini

The one number I would ask for before any retention or credit claim:

**`cosine(eligibility, dL/dw)` with the readout trained, plus a same-episode random
null.** Mine is +0.059/+0.046 against nulls of +0.067/+0.028 — i.e. at null — for both a
spike and a continuous readout. If CKCER's trace is also at null, its accuracy gain is not
from directed credit.

And a warning from this round: I attempted the exact-derivative trace
(`e_in ← leak·e_in + x`, `Δw = (W_out·error)_i · e_in`) and my harness reported
`|trace| = 0` while an independent script reported 3.551 for the same expression. If you
build this trace, verify `credit` is non-zero **and** that the raw-unit head reproduces
the trained probabilities before trusting any alignment number.

## 8. Artifacts

`scripts/r7_r9_exact_trace.py` (broken harness, committed as-is) ·
`local/debug_trace.py`, `local/debug_credit.py`, `local/debug2.py` (the three probes that
disagree) · `report-round8.md` (the finding this round was trying to extend)
