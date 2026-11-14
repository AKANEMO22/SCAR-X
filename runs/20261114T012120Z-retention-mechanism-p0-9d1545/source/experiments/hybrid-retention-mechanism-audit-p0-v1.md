# Retention mechanism audit P0 —14/11/2026 before numerical intervention

Information-path and actualtool audits changed the next action: do not train a
new error policy until observations, recovery outcomes and retention mechanisms
are understood. Current Hybrid LocalALIF contains importance/stable arrays whose
causal use in learn/consolidate must be established, not inferred from names.

Read actual source. On GPU create three independent seed clones260113/260137/
260171 with identical trainable weights/state/RNG.4bounded deterministic owned
input sequences/8ticks, externally defined action indices0..3, repeated8passes
(32existing local updates/seed). No imported livecheckpoint/VLMforward/training,
new learning formula, extra replay, neuron scaling or hypothesis retuning.

Before updates alter ONLY unused-candidate protection state in paired clone:
importance filled1000, stable_in/rec/out/bias shifted+0.25. Run identical inputs/
targets/evidence and compare trainable w_in/w_rec/w_out/bias at every step.
If exactly identical, these stored buffers have no demonstrated protection
effect in this path. It says nothing about metaplasticity algorithms elsewhere.
Check original importance remains zero and stable buffers unchanged frominit;
inspect actual consolidate for whether it calls stable-EMA or only replays learn.

Separate forward diagnostic: keep all synaptic weights fixed on checkpointclones,
alter only f_mean/f_var between initial and after32update statistics, evaluate
the same4inputs. Report logits/probability/rank change. Normalization is an online
state mechanism and must not be mislabeled learned hiddenweight benefit or
proven forgetting cause. These4ownedinput probes are not an accuracy benchmark.
No broad oldskill/3-5ppgate conclusion from this P0.

Record sourcehashes, fullarrays/RNG/replay checkpoints, per-step maxdifference,
stats shifts, input hashes and readout outputs; CPUhost only serialization/RNG,
all neuralforward/localupdateCUDA.≤2minutes/32updates×3seeds×2clones.
Observe actualGPUheadroom; do not modify frozen8hsourcefiles or liveweights.
Publish measured results at8775 as orchestrator, separate from longrunlab counts.
Future importance/anchor formula remains a distinct proposal needing scientific
registration; this audit does not implement one or promote any policy.
