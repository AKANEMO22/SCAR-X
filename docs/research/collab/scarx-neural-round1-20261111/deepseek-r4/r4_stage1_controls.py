"""R4A Stage 1 EXTENSION — the three controls my central claim depends on.

Post-hoc, disclosed as such. These are diagnostics, not a new candidate and not a
gate change. Without them the claim "the interaction, not the width, recovers the
information" is untestable.

C1 matched capacity: project the R4 substrates to 258 dims (the width of R5) so
   that width and interaction are separated.
C2 query ablation: on R5, replace the query cue with a shuffled one. If accuracy
   survives, the interaction is not being used.
C3 tick control: capture post-state at t=10 with the query tick zeroed, so that
   "the query did it" is separated from "one more tick did it".
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r4_probe as P  # noqa: E402
from scarx.evaluation import r4_relational as R  # noqa: E402

DEV = (95001, 95007, 95013)
N_TRAIN, N_TEST = 1024, 512


def project_to(x_tr, x_te, out_dim, seed):
    rng = np.random.default_rng(seed + 31_000)
    W = rng.normal(0, 1 / np.sqrt(x_tr.shape[1]), (x_tr.shape[1], out_dim)).astype(np.float32)
    return x_tr @ W, x_te @ W


def main():
    out = []
    print("=" * 84)
    print("STAGE 1 EXTENSION — controls the central claim rests on (post-hoc)")
    print("=" * 84)
    for seed in DEV:
        trx, try_, _ = R.query_set(seed + 200_000, N_TRAIN)
        tex, tey, _ = R.query_set(seed + 300_000, N_TEST)
        ctr = P.capture(seed, trx)
        cte = P.capture(seed, tex)
        post_tr = P.state_vector(ctr, ("spikes", "voltage", "adaptation", "features"), "post")
        post_te = P.state_vector(cte, ("spikes", "voltage", "adaptation", "features"), "post")
        q_tr = P.query_cue(trx); q_te = P.query_cue(tex)

        # ---- C1 matched capacity: same 258 dims, NO interaction
        rec = {"seed": seed}
        for name in ("decay025", "spikesum", "bank5"):
            a = ctr["final"][name]; b = cte["final"][name]
            pa, pb = project_to(a, b, 258, seed)
            rec[f"C1_{name}_proj258_ridge"] = P.ridge(pa, try_, pb, tey)
            h = P.train_head(pa, try_)
            rec[f"C1_{name}_proj258_head"] = h.acc(pb, tey)
        # and post-state itself at 258 dims with no interaction
        ppa, ppb = project_to(post_tr, post_te, 258, seed)
        rec["C1_poststate_proj258_ridge"] = P.ridge(ppa, try_, ppb, tey)
        rec["C1_poststate_proj258_head"] = P.train_head(ppa, try_).acc(ppb, tey)

        # ---- full R5 for reference on the same seeds
        bil_tr = P.bilinear_query_map(post_tr, q_tr)
        bil_te = P.bilinear_query_map(post_te, q_te)
        rec["R5_ridge"] = P.ridge(bil_tr, try_, bil_te, tey)
        rec["R5_head"] = P.train_head(bil_tr, try_).acc(bil_te, tey)

        # ---- C2 query ablation: shuffle the cue
        rng = np.random.default_rng(seed + 555)
        qs_tr = q_tr[rng.permutation(len(q_tr))]
        qs_te = q_te[rng.permutation(len(q_te))]
        rec["C2_shuffled_cue_ridge"] = P.ridge(
            P.bilinear_query_map(post_tr, qs_tr), try_,
            P.bilinear_query_map(post_te, qs_te), tey)
        # shuffled cue at evaluation only, correct cue at training
        rec["C2_cue_shuffled_at_test_ridge"] = P.ridge(
            bil_tr, try_, P.bilinear_query_map(post_te, qs_te), tey)

        # ---- C3 tick control: zero the query tick, then capture at t=10
        zx = tex.copy()
        zx[:, 10, :] = 0.0
        zc = P.capture(seed, zx)
        zpost = P.state_vector(zc, ("spikes", "voltage", "adaptation", "features"), "post")
        ztr = trx.copy(); ztr[:, 10, :] = 0.0
        zctr = P.capture(seed, ztr)
        zpost_tr = P.state_vector(zctr, ("spikes", "voltage", "adaptation", "features"), "post")
        rec["C3_query_zeroed_poststate_ridge"] = P.ridge(zpost_tr, try_, zpost, tey)
        # with the CORRECT cue added on top of a query-zeroed state
        rec["C3b_query_zeroed_plus_cue_ridge"] = P.ridge(
            P.bilinear_query_map(zpost_tr, q_tr), try_,
            P.bilinear_query_map(zpost, q_te), tey)

        out.append(rec)
        print(f"  seed {seed} done", flush=True)

    def M(k):
        return float(np.mean([r[k] for r in out]))

    print("\n" + "=" * 84)
    print("C1 — MATCHED CAPACITY (258 dims, NO interaction)")
    print("=" * 84)
    for name in ("decay025", "spikesum", "bank5"):
        print(f"  {name:11s} proj258 ridge {M(f'C1_{name}_proj258_ridge'):.4f}  "
              f"head {M(f'C1_{name}_proj258_head'):.4f}")
    print(f"  {'post-state':11s} proj258 ridge {M('C1_poststate_proj258_ridge'):.4f}  "
          f"head {M('C1_poststate_proj258_head'):.4f}")
    print(f"\n  R5 WITH interaction  ridge {M('R5_ridge'):.4f}  head {M('R5_head'):.4f}")

    print("\n" + "=" * 84)
    print("C2 / C3 — does the QUERY do the work?")
    print("=" * 84)
    print(f"  C2 shuffled cue (train+test)      {M('C2_shuffled_cue_ridge'):.4f}")
    print(f"  C2 correct train, shuffled test   {M('C2_cue_shuffled_at_test_ridge'):.4f}")
    print(f"  C3 query tick zeroed, post-state  {M('C3_query_zeroed_poststate_ridge'):.4f}")
    print(f"  C3b zeroed state + real cue       {M('C3b_query_zeroed_plus_cue_ridge'):.4f}")

    verdict = {
        "interaction_beats_width":
            M("R5_ridge") - max(M("C1_poststate_proj258_ridge"),
                                M("C1_spikesum_proj258_ridge")) > .10,
        "cue_is_necessary": M("R5_ridge") - M("C2_cue_shuffled_at_test_ridge") > .10,
        "query_content_required":
            M("C3b_query_zeroed_plus_cue_ridge") - M("C3_query_zeroed_poststate_ridge") > .10,
    }
    print("\n  VERDICT INPUTS:", json.dumps(verdict, indent=1))

    p = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r4"
    p.mkdir(parents=True, exist_ok=True)
    (p / "stage1-controls.json").write_text(
        json.dumps({"per_seed": out, "verdict_inputs": verdict}, indent=2,
                   default=float) + "\n", encoding="utf-8")
    print(f"\nwritten: {p / 'stage1-controls.json'}")


if __name__ == "__main__":
    main()
