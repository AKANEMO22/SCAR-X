# Passive technical work trace — acceptance v1

Product acceptance, not a representation/retention experiment. No new learning
rule, model size, dataset, training schedule, or quality threshold is introduced.

Before execution: publish this protocol and runner source in a registered run.
The user only opens localhost; no model/source selection or start action.

1. Publish completed source edits with SHA256 evidence, explicitly as technical
   work rather than neural thoughts or measurements.
2. Run the existing full pytest suite in an isolated subprocess. Keep its actual
   exit status and output; a failure fails acceptance, never fabricate success.
3. Call the existing observation endpoint for one reviewed public prompt. It
   executes the live neural core; no target enters the input or lesson queue.
4. Repeated GETs must expose that exact generated answer, reference and archived
   Qwen output, while the live token-update counter remains unchanged and the
   learning queue stays empty. Existing real learning records must remain readable.
5. Verify rendered localhost separately through Computer Use, without clicking
   any model/start control. Logs retain their timestamps after completion.

No new training just to populate the UI. No inferred/mocked loss, spikes,
answers, Qwen outputs or guard verdicts. Unit tests may use their isolated
fixtures; their output is identified as software testing, never live model quality.
Resource cap remains 2,000,000,000 bytes for the AI runtime. Browser and pytest
checks are reported separately from deployed runtime. Existing research gates
and failed/stopped families remain unchanged.
