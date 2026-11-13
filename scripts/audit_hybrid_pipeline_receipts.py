"""Read only fixed-fixture public receipts; explicit pass vs admission/completion."""

import argparse
import json
from pathlib import Path


def answer(value):
    if "answer" in value:
        return value["answer"]
    return value.get("choices", [{}])[0].get("message", {}).get("content", "")


def inspect(run):
    before = json.loads(
        (run / "before_restart.public.json").read_text(encoding="utf-8")
    )
    after = json.loads((run / "after_restart.public.json").read_text(encoding="utf-8"))
    traffic = json.loads(
        (run / "traffic-and-source-audit.json").read_text(encoding="utf-8")
    )
    result = {
        "run": run.name,
        "cases": [],
        "limitations": [
            "One public speech sample plus synthetic shapes; not P1/P2/P3 or hidden-learning proof."
        ],
    }
    for row in before["cases"] + after["cases"]:
        name = row["case"]
        value = row.get("result") or {}
        item = {
            "case": name,
            "transport_completed": row["completed"],
            "seconds": row["seconds"],
        }
        if "expected" in value and "response" in value:
            item.update(
                expected=value["expected"],
                actual=answer(value["response"]),
                correct=answer(value["response"]).strip().casefold()
                == value["expected"].casefold(),
            )
            if value["response"].get("ttft_seconds") is not None:
                item["ttft_seconds"] = value["response"]["ttft_seconds"]
        elif name == "function_round_trip":
            first = (
                value.get("first_response", {})
                .get("choices", [{}])[0]
                .get("message", {})
            )
            calls = first.get("tool_calls") or []
            function = calls[0].get("function", {}) if calls else {}
            try:
                args = json.loads(function.get("arguments", "null"))
            except ValueError:
                args = None
            item["correct"] = (
                function.get("name") == value["expected_function"]
                and args == value["expected_arguments"]
                and answer(value.get("second_response", {})).strip() == "17"
            )
        elif name == "uploaded_video":
            item.update(
                source_seconds=value.get("duration"),
                capture=value.get("capture_frontier"),
                ASR=value.get("asr_frontier"),
                vision=value.get("vision_frontier"),
                degraded=value.get("degraded", False),
                skipped=value.get("skipped_intervals", []),
                status=value.get("status"),
            )
        elif name == "rollback":
            item["correct"] = all(
                value.get(k) is True
                for k in ("full_snapshot_exact", "evidence_retained", "events_retained")
            )
        elif name == "screen_capture_own_fixture":
            s = value.get("stream", {})
            item.update(
                capture=s.get("capture_frontier"),
                vision=s.get("vision_frontier"),
                audio_requested=value.get("system_audio_requested"),
                microphone_used=value.get("microphone_used"),
            )
        elif name == "automatic_idle_review":
            item.update(
                completed_reviews=value.get("new_review_count"),
                synaptic_updates=value.get("synapse_updates"),
                hidden_updates=value.get("hidden_updates"),
                replay_items=value.get("verified_replay_items"),
            )
            if traffic["before_restart"]["foreign_streamed_requests_during_idle"]:
                item["classification"] = "INCONCLUSIVE_CONTAMINATED_BY_LIVE_TRAFFIC"
            else:
                item["classification"] = (
                    "LIFECYCLE_OBSERVED"
                    if value.get("new_review_count", 0) > 0
                    else "NOT_OBSERVED"
                )
        result["cases"].append(item)
    result.update(
        ASR=before.get("ASR"),
        cold_persistence=after.get("cold_persistence"),
        peak_ram_bytes=before["resources"]["peak_conservative_ram_bytes"],
        limit_bytes=before["resources"]["limit_bytes"],
        traffic=traffic,
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    a = p.parse_args()
    result = inspect(a.run.resolve())
    with (a.run / "fixture-score-audit.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=True, indent=2))
