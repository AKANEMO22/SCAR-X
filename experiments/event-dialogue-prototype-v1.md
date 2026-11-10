# Event-to-word shared-core dialogue prototype v1

2026-11-09, before end-to-end training. Within product MVP/P1/P2, NOT a new
framework/model router, bigger hidden population, language pretraining or
permission to replace serving weights. Input encoder alone did not demonstrate
language; this trial explicitly measures the actual autoregressive word path.

Data is versioned curated Antigravity40dialogues,194targettokens overall,
train32(20vi12en)/development8(4vi4en). Use ONLY train target tokens to define
vocabulary; report development OOV and never expand from dev labels. Preserve
all prefix roles/turns; target is the next assistant utterance, never in its
input prefix. Labels synthetic/editorially reviewed, not factual proof.
Development is already available to reviewers, not a new final confirmation.

One fresh UnifiedCore48ALIF,K8,R12 per independent seed5101,5113,5119 and condition:
bag_raw,ordered_raw,ordered_normalized,ordered_normalized_full. Channel layout
V+128+64+6,word+digit slices, no pretrained weights. Word-decoder output masks
BOS/other-modality classes and firstEOS; no punctuation ban (old serving negative
answer bug is not silently migrated). Full condition uses the existing scalar
advantage/STDP hidden rule; all other conditions freeze hidden.

Chronological events include prompt tokens/speaker/boundaries. Bag control
replaces text hash channels with their per-prefix mean at EVERY event, retaining
the same event count/control markers. Ordered and bag conditions receive same
past dialogue prefix, unlike the live latest-message-only transport.
Use a normalized pooled text summary as the existing decoder context channels
after encoding; ordered state is additional information. All word steps use the
current3ticks, same ALIF dynamics. RGBpatch event path can feed the SAME core
and word readout; no separate vision model, but natural-photo skill is untrained
and must remain FALSE. No photo labels used in this language trial.

Calibrate per-neuron readout mean/std using one unlabeled/teacher-forced walk of
TRAIN utterances only; observed previous target words are teacher forcing,
never current/future target features. Freeze stats before updates. Both raw and
normalized heads share the same output-local lr0.08,clip0.025,weightclip3;
hidden_lr0.0001, no new random-feedback family. Train8passes through the same
96response examples, no neural-state restart across passes/conversations.
Seeded dialogue-order permutation each pass, preserve turn order in each dialogue.
These bootstrap passes are an experiment, not retraining during each user chat.

Evaluate initial/after4/after8 on diagnostic clones only, identical data budget.
Metrics: train/dev exact utterance, teacher-forced NLL/token accuracy, dev OOV,
generated text, repeated-token rate. Existing research gates unchanged; do not
equate low NLL or memorizing phrases with normal conversation or useful hidden
representation. If both4and8pass checkpoints have <10ppdev exact gain vs their
initial model, stop this prototype, no longer training/larger data/model.
No best-checkpoint deployment/selection from dev; freeze artifacts after8.
Report normalized vs raw and full vs readout paired effects descriptively;
3seeds insufficient for broad claims. Per-condition90s, whole900s,<=2GB sampled
tree RAM, CPU1thread, max2workers. Session persistence must recover model/head
statistics/conditioning/scheduler counter, with cold-process inference equality.
No retention claim: known20context gate still failed. Any further confirmation,
interference/oldskills/digits/vision tests must be separately preregistered.
