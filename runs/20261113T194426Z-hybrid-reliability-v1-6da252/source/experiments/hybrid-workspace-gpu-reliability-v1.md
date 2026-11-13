# Engineering registration14/11 — workspace, bounded tools, compression, full GPU

User reported local-project requests becoming repeated web searches; requested
real tests, long-running checks and cold-start explanation. Additional amendment:
GPUmode moves VLM/vision, ASR and ALIF numerical processing/learning to CUDA;
CPUmode uses CPU (iGPU optional, no discrete GPU). OS/I/O/tokenization/SQLite
remain host work. Original precision, ALIF equations/topology/local eligibility,
scientific gates, main/R13 and private production learning state stay unchanged.

P0: reproduce existing tool-loop config; enable existing Hermes hard stops rather
than warnings, preserve other tools. Expose filesystem/terminal schemas for SCAR-X
alongside web through existing disclosure; verify actual project cwd/prompt and
read-only execution. Fix scoped measured16K compression feasibility (no fake64K),
preserve generic minimum on other routes, verify real compression and message shape.

P0GPU: pin official whisper.cpp b5454 CUDA12.4 assets + publishedSHA, nativeFP32
weights; CuPy14.2 + minimalCUDA12.9 libraries in private runtime-env. No autograd,
backward/BPTT/precision reduction. CPU/GPU initial arrays exact, 3seed260113/260127/
260131, 20 deterministic local updates each, p/trace/weights tolerances
atol3e-5/rtol3e-5 (spike equality except documented threshold crossings). Cold
same-device checkpoint next update must match; cross-device continuation is
tolerance-checked, never promised bit-exact float reduction ordering.

Real acceptance: isolated Hermes home/session and immutable run receipts;
file-reading fixtures then actual app z fold read-only if same path configured.
Fail if uses web instead of reading local source or claims unseen contents.
Run known repeated empty-search tool sequence through actual Hermes guardrails;
hard stop by configured threshold and webcap. Model must not execute unbounded
tool calls on held-out workspace prompts; retain all mistakes/interrupts.

Measure GPU offload logs + Nvidia/PDH during actual inference, VLM cold load vs
prompt processing vs generation, CPU/GPU ASR on same audio, native tensor parity.
Report component placement honestly: a CUDA flag alone is not proof all kernels
left CPU. CPU mode must allocate no discrete-GPU model tensors.

Endurance engineering: aim30minutes mixed real read-only agent tasks, compression,
idle review, sleep/wake, ALIF updates and restart/rollback in isolated datastore.
Bound individual request90s and tools per turn; save receipts incrementally and
expose actual events automatically at localhost8775. No user-history training,
fake model outputs or silent retuning after results. An interrupted/short run is
reported at actual duration. This does not substitute for the8hour release soak.

Keep lifelong gates unchanged (20stages forgetting mean<=3pp/no group>5pp).
Device/loop/persistence PASS is not learned-skill/hidden-advantage/LoRA-policy PASS.
GPU auto-rest keeps the frozen VLM warm by default now that GPU RAM is uncapped;
explicit sleep still unloads it. CPU automatic release remains unchanged. Measure
one genuine cold reload separately; do not label cached or loaded state cold.
Unqualified automatic LoRA creation/loading remains blocked; no closed-test reuse
as training labels or fabricated success. Report unresolved semantic/capacity limits.

DEV showed the native0.8B can ignore a named tool_choice and echo prompt markers
even with correct cwd/eager tools. Engineering amendment before endurance:
use one explicitly attributed runtime read-only workspace prelude (bounded
directory listing + existing top-level docs), executed by Hermes' normal terminal
tool/approval path. Do not replace model answers or claim this as learned skill.
Preserve original failed DEV outputs; all subsequent raw model answers stay raw.

The first corrected endurance preflight exposed a second compression bug: the
gateway silently defaulted the uncapped summary request to512outputtokens while
Hermes targeted2000. Keep this FAIL/95stimeout. Before the new run, let identified
Hermes summary calls use up to4096tokens, capped by native-tokenizer measured
free room in the actual16384window. Ordinary outputs stay1024max. Use bounded
repeat_penalty1.05 at inference to discourage degenerate output; storage and native
numerical parity calls remain unchanged. Last runtime cwd marker wins over project
prose. This is registered engineering repair, not a retuned scientific benchmark.

The output-room DEV still truncated on the0.8B summarizer. Enable the existing
Hermes redacted deterministic fallback immediately on summary failure, scoped to
the same local16K model. This stays within the original compression transaction,
not a manual history deletion; record extractive fallback versus generated summary.
It can lose detail and is not a semantic-memory or quality PASS. Other models'
recovery and64K feasibility floor stay unchanged.
