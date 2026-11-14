"""Require actual active, available peak samples; final shutdown status is not proof."""


def qualify_active_samples(samples, baseline_shared):
    active = [r for r in samples if r.get("owned_ASR_active") is True]
    valid = [r for r in active if r.get("shared_available") is True
             and r.get("dedicated_available") is True and "free_MiB" in r]
    fresh = {r["PDH_collection_monotonic"] for r in valid}
    peak_shared = max([r["shared_bytes"] for r in valid] or [baseline_shared])
    minimum = min([r["free_MiB"] for r in valid] or [0])
    sufficient = len(fresh) >= 2 and len(valid) == len(active) and bool(active)
    return {
        "active_samples": len(active), "available_active_samples": len(valid),
        "fresh_PDH_collections": len(fresh), "sufficient_active_coverage": sufficient,
        "peak_shared_bytes": peak_shared,
        "shared_growth_bytes": peak_shared - baseline_shared,
        "sampled_min_free_MiB": minimum,
        "resource_gate_pass": bool(sufficient and minimum >= 512
                                   and peak_shared - baseline_shared <= 256 * 1024 * 1024),
        "scope": "Sampled observed allocations; PDH2s/globalNvidia0.25s may miss shorter transients",
    }
