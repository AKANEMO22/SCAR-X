# Orchestrator rereview — Grok d509878

Reviewed `d50987862ff80d2bd8556eecfb6af1b259cc591e`, PR#2.
**APPROVE_RESEARCH_ONLY_WITH_ORCHESTRATOR_HARDENING**. This is not approval to
activate a serving learner, reopen the hypothesis, or claim useful hiddenlearning.

## Scientific correction verified

Six author tests passed. Independent replay used the exact registered data,
seeds, arms, two-pass budget and thresholds. Only the output directory changed;
no source/protocol/result in the author checkout was overwritten. Every saved
metric and prediction matches d509878's corrected result exactly, including the
71005data hash and all eight confirmation initializations.

The lower95% bound of slow-correct minus slow-frozen is
-0.0028908192607432423, below the unchanged0.05 gate. Slow-zero equals frozen on
all8seeds. **NO-GO remains** for the registered usefulhiddenlearning claim.
The readable slow state is not newly learned representation.

The original corrected file recorded parent commit14bf4b2 while correctioncode
was uncommitted. Our separate replay pins d509878 and preserves the old files;
use its source/log/metric evidence for reproducibility instead of pretending
that14bf4b2alone produced the corrected experiment.

## Engineering review

The four original findings are addressed: correct confirmation split71005;
outerZIP/type/range validation;3264extra memory bytes counted; historicaloutput
refused and CI/thread/provenance reported. A remaining nestedNPZ issue was
reproduced with a tiny duplicate inner member, not a large memory allocation.

The integration branch adds only malformed-extra validation: exactly2uniqueNPY
members, declared uncompressed bounds, expected dtype/shape/header/body size
before np.load can allocate. Two negative regressions reject an innerduplicate
and a huge declaredshape without allocating it. Valid checkpoints still load;
no equation, forward, weights, seeds, budget or measured outcome changed.

Full integration regression:300passed, no skippedtests; native kernel built in
this temporary checkout. A duplicate-member fixture emits an expected warning.
Production app/neural/learning code and publicrelease descriptor have no diff
from main. No G1liveSQLite, weights, process, Hermes or provider configuration
was written/restarted. This is a research integration candidate only. Main remains unchanged; serving
adoption is NO-GO. Source/evidence is ready for optional archival integration.

## Reproduction from a checkout with recorded results

The default runner intentionally refuses its already-recorded output. Use a
fresh output directory; do not delete/overwrite history. The verified replay
uses this same import-and-output-only override:

```python
import importlib.util
from pathlib import Path
from uuid import uuid4
script = Path('scripts/grok_slow_trace_screen.py').resolve()
spec = importlib.util.spec_from_file_location('locked_grok_runner', script)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = script.parents[1] / 'runs' / ('grok-review-' + uuid4().hex[:8])
runner.main()
```

Run with PYTHONPATH=src and the four thread variables recorded in the result.
The original protocol and both author results remain immutable. The amendment's
wording “sample standard deviation with one degree of freedom” means ddof=1;
the actual two-sided t interval uses n=8 and df=7, as verified in code.
