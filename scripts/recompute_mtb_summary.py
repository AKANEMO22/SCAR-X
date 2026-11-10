"""Recompute the screen summary from an already-completed run's raw trials.

The 297 trials are the expensive part and are already on disk. The summary step
had two bugs (empty lr lock in screen phase; vacuous-True screen_pass). This
script reuses the exact raw rows and rewrites result.json/report.md with the
corrected aggregation. It does not re-train and does not touch trial numbers.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_collab_mtb as R  # noqa: E402


def main(run_dir: Path):
    # Prefer result.json (written at completion, so it holds every trial);
    # progress.json is flushed every 10 trials and can lag the final batch.
    src = run_dir / "result.json"
    if src.exists():
        trials = json.loads(src.read_text(encoding="utf-8"))["trials"]
        source_name = "result.json"
    else:
        trials = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        source_name = "progress.json"
    print(f"raw trials loaded from {source_name}: {len(trials)}")

    # integrity: every registered cell must be present exactly once
    expected = {(t, s, c, lr) for t in R.TASKS for s in R.DEV_SEEDS
                for c in R.CONDITIONS for lr in R.READOUT_LR_GRID}
    seen = {(r["task"], r["seed"], r["condition"], r["readout_lr"]) for r in trials}
    missing = expected - seen
    dupes = len(trials) - len(seen)
    print(f"expected cells: {len(expected)} | present: {len(seen)} | "
          f"missing: {len(missing)} | duplicates: {dupes}")
    if missing:
        print("MISSING:", sorted(missing)[:10])
        raise SystemExit("incomplete trial grid; refusing to summarise")

    locked, lr_detail = R.select_lr_per_arm(trials)
    print("locked lr per arm:", json.dumps(locked))
    summary = R.summarise(trials, "screen", locked)

    result = {
        "protocol": "collab-mtb-screen-v1",
        "phase": "screen",
        "config": json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config"],
        "trials": trials,
        "summary": summary,
        "lr_selection": lr_detail,
        "resources": json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
        "scope": "Registered synthetic tasks at H=48. No chat/vision/reasoning claim.",
        "aggregation_note": (
            "result.json regenerated from the same raw trials after fixing two "
            "aggregation bugs: the per-arm lr lock was empty during the screen "
            "phase, and screen_pass was vacuously True because all gates were "
            "None. Trial numbers are unchanged; integrity of the 297-cell grid "
            "is asserted above."),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    (run_dir / "report.md").write_text(R.render_report(result), encoding="utf-8")

    lock_path = Path(__file__).resolve().parents[1] / "artifacts" / "antigravity-r1" / "locked-readout-lr.json"
    lock_path.write_text(json.dumps({
        "per_arm": locked, "selected_on": "development seeds only",
        "selection_metric": "mean dev_direct across 3 tasks x 3 dev seeds",
        "screen_run": run_dir.name, "screen_pass": summary["screen_pass"],
        "note": "Written before any confirmation trial. Never re-tuned after.",
    }, indent=2) + "\n", encoding="utf-8")

    print("\nSCREEN GATES")
    for k, v in summary["screen"].items():
        print(f"   {k:34s} {v}")
    print(f"   screen_gates_unknown: {summary['screen_gates_unknown']}")
    print(f"\nscreen_pass: {summary['screen_pass']}")
    print("\nCONTRASTS (direct, pp)")
    for k, c in summary["contrasts"].items():
        x = c["direct"]
        if x:
            print(f"   {k:34s} {x['mean']*100:+7.2f} [{x['lower95']*100:+7.2f}, {x['upper95']*100:+7.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
