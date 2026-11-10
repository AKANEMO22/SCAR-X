# Generation 1 — bounded product acceptance, registered 2026-11-09

Authorized: one shared recurrent core, idle processing/trace, one Hermes ID,
replace retired product choices. Historical research protocols/results unchanged.
No growth/rewiring, pretrained model, BPTT, automatic idle weight learning or
general language/vision claim. H=48, K=8, R=12. New modality input/output sizes
are interfaces, not evidence that adding capacity solves hidden learning.

## Build and training

- One ALIF SpikingCore, one recurrent weight matrix and state shared by text and
  8x8 digits; modality slices of one local readout. Start from scratch, do not
  merge incompatible old pilot neurons/checkpoints.
- Authored small dialogue families; fixed train/development/acceptance prompts.
  UCI sklearn digits, stratified 60/20/20 split seeds 2701/2703. Prior digits
  diagnostics were seen; report the new split as diagnostic, not pristine data.
- Two independent initialization/order seeds 17,29. At most 100 joint epochs,
  600 seconds/seed, one CPU numerical thread, below-normal priority, 2 GB cap.
- Alternate dialogue lessons with digits; output-local delta, reward-modulated
  hidden eligibility. No backward. Select only on equal-weight mean of dev
  text exact match and digit accuracy; ties use lower dev text loss. Final
  acceptance prompts are opened once after selecting each seed's checkpoint.
- Product candidate targets: >=80% on six greeting acceptance paraphrases,
  >=75% diagnostic digits; report other dialogue errors and seed variation.
  Failure does not justify relabeling this as general chat or a research GO.

## Engineering acceptance

- Exactly one model ID/core loaded; image/text training change the same recurrent
  arrays, and checkpoint contains both output slices and the full shared state.
- Idle ticks advance neuron state and bounded trace without any weight updates;
  foreground waiting preempts the next background quantum. No fake prose thoughts.
- No reset_state during runtime chat or background ticks; use a bounded neural
  settling interval at modality/episode boundaries. Training uses same boundary.
- Checkpoint/restore includes neuron/eligibility/weights/stable/RNG/controller,
  replay cursor, pending probe state. Quiescent exact continuation, checksum
  rejection, DB failure rollback, lock ownership and no silent old-format import.
- Pause/resume and idle speed controls, resource guard, <=2 GB process cap where
  Windows supports it. Trace ring <=256 records; no unbounded idle disk logging.
- SDK nonstream/SSE, image route, invalid request rejection and UI at desktop/mobile.
- Final package contains public initial weights only, no local/ or Hermes secrets.

## Research status

PIVOT_NO_SCALE. Previous forgetting mean<=3 pp / worst<=5 pp gates remain failed.
Training loss, greetings and visible spikes cannot establish useful hidden
representation, reasoning, continual learning or experience-to-skill transfer.
Adaptive capacity: activation first, fixed-budget rewiring second, growth only
after independent approval and causal diagnostics; none enabled in this release.
