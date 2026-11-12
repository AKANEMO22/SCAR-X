# R5B / Grok R6B historical reproduction

Source: `c39e9db7771582d9933af402a209917328b71119`, with R5B original
`d6c609df6f90fc8df96fb3de424dd54388f8dc59`. Effective Python bindings are used;
the original source/history is unchanged.

Eight historical seeds98001/98007/98013/98019/98023/98029/98031/98037 and the
three published screen seeds120011/120017/120023 were replayed, not fresh.
All published scientific fields reproduce exactly; runtime/RSS fields are new
measurements, not expected to be identical. Raw pair predictions are preserved.

| Metric | Independently reproduced |
|---|---:|
| Head both_A | .6676171928664432 |
| Ridge both_A on same tensor | .9858055982735111 |
| NoClip gain, mean | +.00982010306580842 |
| NoClip gain CI95% | [−.07079646,.06349206] |
| Protected order drop | .1875 |

**Candidate FAIL/STOP stands.** A high ridge is evidence that the selected fixed
features permit a fitted linear diagnostic; it is not a universal information
ceiling, an online learner, or newly learned hidden representation.

## Source defects and completed audit corrections

`grok_r6_causal.py` defines `pair_stats`, `fit_predict`, `screen_seed` twice.
The effective last bindings start at lines223/241/204. The toy test imports real
NoClip and then redefines a class with the same name. Two original toy tests
pass, but that binding is not the actual candidate. Codex exercised the real
class and verified its unclipped one-step arithmetic and finite weights.

`scripts/forensic_normalized_grok_r6_causal.py` removes only the three overwritten
earlier definitions. The effective AST bodies are unchanged and source line
numbers preserved; GROK_NORMALIZATION.json records the transformation/hashes.
The scored reproduction used the original last bindings, not a rewritten rule.

## Protocol vs execution

| Registered condition | Executed in original candidate screen |
|---|---|
| Clean direct/caseA/caseB | Yes |
| Second training order | Yes |
| QueryA/queryB paired answers | Yes, same sibling evaluation |
| Protected simultaneous XOR/order | Yes, separate fitted protected heads |
| Reversed writes | Missing |
| Shift | Missing |
| Distractor | Missing |
| Noise | Missing |
| Fresh confirmation | Correctly not opened after screen failure |

Missing stress cells are NOT_TESTED; they are not filled by new experiments.
The existing negative screen is sufficient to keep STOP. Independent counter-
factual metrics validate grouping/labels, rather than trusting a supplied flip
mask or a summary. The actual published pair arrays are saved in the unique audit
run and can be regraded without invoking the agent's scorer.
