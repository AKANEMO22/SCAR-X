# SCAR-X working rules

- Load ponytail rules/tools/context first when available. Record unavailability honestly.
- Keep all SCAR-X changes inside this project; the parent contains unrelated font work.
- Start at README.md, then docs/PROJECT_MAP.md and docs/STATUS.md.
- Production neural code in src/scarx/neural must not import torch/autograd or use backward gradients.
- Non-negotiable user constraint: SCAR-X remains one Spiking Recurrent Neural Network with LIF/ALIF neurons and local synaptic learning. Never replace its core with a Transformer, pretrained LLM, or LLM proxy to improve chat/vision scores. Qwen and other architectures are isolated evaluation controls only, never serving/training components of the SCAR-X core. Preserve the CPU-first 2 GB target. Research changes must preserve this identity; do not silently change the architecture.
- Backprop controls live only in src/scarx/evaluation/controls.py and run as separate experiments.
- User approved proposal30 research on2026-11-10: a separate ALIF prototype may combine neuron-local forward eligibility with one-step learned readout-error feedback (W_out.T @ error). This explicitly relaxes signal locality; it is not permission for Transformer, whole-network/time backward, candidate autograd/RTRL, scaling or deployment. Production core/weights stay unchanged. Older pending notices describe their historical time.
- Register protocol and seeds before formal research benchmarks; never select a winner on final test data. Product-first MVP development and normal interactive teaching are authorized without research gates; record acceptance scenarios and limits honestly.
- Results go under runs/<unique-run-id>; do not rewrite previous run results.
- Trained checkpoints go in artifacts/, source fixtures in data/fixtures, user data in local/.
- Update docs/STATUS.md after a milestone; report failures and unmeasured claims.
- Do not expand to language pretraining or bigger models when the preceding gate has failed.
- Product entrypoint: run_app.py and README.md. Master plan: docs/RESEARCH_PLAN.md. Preserve historical research protocols/results; research gates do not block building a weak but real local MVP.
- Run pytest and CPU/native parity tests for neural/runtime changes.
- The localhost observer is passive: never require the user to select a model/source or click start to see agent work. Publish actual edits/checks through runtime.work_trace.activity with timestamps/evidence, and publish actual answers/learning measurements through the registered runner. Keep technical work separate from neural activity; never invent measurements or launch training merely to fill the page.

