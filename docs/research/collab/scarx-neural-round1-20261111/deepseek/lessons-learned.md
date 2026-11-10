# Lessons learned — deepseek — campaign `scarx-neural-round1-20261111`

Model actually used: `deepseek-ai/deepseek-v4.1-flash-1017` under DeepSeek Harness.

## 1. Which of my hypotheses was wrong?

**Both of them, in sequence.**

**First error — the generator.** My initial `pair_binding` drew the two value bits
independently. When they coincided, the multiset of writes collapsed to `{0,0}` or
`{1,1}`, so an order-1 bag readout recovered the label and scored **0.70–0.77** —
above the 0.65 gate. The task looked like binding but was a bag read. Caught by P0
before any learning run. Lesson: *"the task is order-k" is not a property you can
declare; it is a property you must measure against the weakest readout that
shouldn't work.*

**Second error — the claim.** I then asserted `pair_binding` was solvable because
its label is an order-2 function. That conflates **the order of the function** with
**the expressive capacity of the architecture**. An additive point neuron forms a
*sum*; `key × value` is a *product*. Order-2 cannot be reached by summing, so the
task was unreachable for exactly the same reason `strict_binding` was — I had only
changed which interaction order was required, not whether the neuron could form one.

Measured: oracle product features **1.000**, raw linear **0.528**, actual ALIF
features **0.522**. My own falsifiability check (C1) is what exposed it. If I had
registered C1 as a step to *confirm* rather than *test*, I would have reported
"pair_binding is a valid gate" and the campaign would have inherited a second
broken instrument.

## 2. What did I find that is worth keeping?

The screen came back at chance, which forced the question I should have asked
first: **is the binding information reachable by this neuron at all?**

| substrate (untrained ALIF, frozen weights) | accuracy |
|---|---:|
| production readout substrate (single leaky integral) | 0.5160 |
| full spike train | 0.6740 |
| **spike co-occurrence across neuron pairs** | **0.9940** |
| co-occurrence including time | 0.9580 |

The binding signal is almost perfectly present in pairwise spike co-occurrence and
essentially absent from what the readout sees.

**This unifies all three branches.** grok found a slow single-neuron trace beats a
fast one. antigravity found a 5-timescale bank beats one decay. I find pairwise
co-occurrence carries binding while the per-neuron readout does not. Those are
consistent: **every readout tried in this campaign is a first-order function of
spikes.** They differ in how much temporal detail they keep, and none of them can
represent a conjunction at all, because a conjunction needs a product.

So the three branches were not three independent attacks on learning. They were
three variations of the same readout family, and so was the wall they all hit.

## 3. What did the other branches get right, and where would I disagree?

**grok** was right to gate on information *before* training (fast 0.484 vs slow
0.9375). That is the single most useful methodological move in the campaign. Where
I'd push further: their slow trace is still first-order in spikes, so it can
capture delayed *order* but structurally cannot capture *binding* — which is
exactly what their `slow-zero == slow-frozen` null shows.

**antigravity** was right that parameter-matched controls are necessary; the
dimension-matched arm is what made the timescale claim testable. Its error was
framing the result as a learning improvement when one slow decay reproduced it.

Where I disagree with **both**: treating `order` as the informative task. `order`
only needs one bit held across a delay, which any sufficiently slow first-order
filter can do. `strict_binding` was the only task that needed structure, and both
branches (and the repository since report 18) read its failure as a learning
failure. It was not. It was a function-class failure, and it made several rounds of
"binding is broken" misdiagnosis possible.

## 4. What could be combined mechanistically, and the honest cost

**Worth trying (new round, new registration):** a readout with an explicit
quadratic term — a fixed random projection of `z_i·z_j` into the head — measured
against a frozen control on the pair-binding task. The §4 diagnostic predicts this
should work; that prediction is testable and cheap at H=48.

**Cost:** `H²/2 = 1128` extra readout inputs at H=48, i.e. ~4.5 KB of weights and
~24× the readout FLOPs of the current 48-dim head. Still trivially inside 2 GB, but
it is a real architectural change to the readout, not a tuning knob, and it should
be registered as such.

**What I will not do:** implement it in this branch now. I discovered §4 *after* my
hypothesis failed. Shipping the fix in the same branch would be fitting the
instrument to the result seen, which is the exact behaviour the campaign forbids.
It belongs in a fresh registration with its own pre-registered prediction.

**Not worth combining:** stacking grok's slow eligibility on top of antigravity's
multi-timescale bank. Both are first-order readout variations; combining them
predicts no new capability, and a large `order` number from that combination would
again be attributable to neither branch's learning component.

## 5. Risks

- **Falsifiable claims saved me.** C1 was registered as a claim to test, not a step
  to confirm. That is the only reason the second generator error was caught rather
  than shipped.
- **Diagnostics discovered after a failure are weaker evidence.** §4 came from a
  post-hoc probe. It should be treated as a promising hypothesis for round 2, not
  as a proven result of round 1.
- **An oracle probe is not a skill.** 0.994 is a statement about *information*, not
  about learning. Any summary saying "SCAR-X can bind" based on this table would be
  false.
- **Three branches, two authors.** Because this branch shares an author session with
  `antigravity`, the campaign effectively has **two** independent parties, not
  three. Any tally treating `deepseek` and `antigravity` as separate votes is wrong.

## 6. Process note

The cheapest decisive measurement in this entire branch was a ridge probe on a
feature I computed by hand in ~30 lines (0.994 vs 0.516). It cost seconds and
reframed all three branches. The 108-trial screen cost ~4 minutes and only told me
my hypothesis was wrong. **Measure the information path before designing the
learner** — twice now, that ordering determined whether a round was informative.
