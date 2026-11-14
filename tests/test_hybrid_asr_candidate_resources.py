from scarx.evaluation.asr_candidate_resources import qualify_active_samples


def sample(epoch=1, **kwargs):
    return dict(owned_ASR_active=True, shared_available=True, dedicated_available=True,
                free_MiB=1000, shared_bytes=100, PDH_collection_monotonic=epoch, **kwargs)


def test_postexit_or_closed_monitor_cannot_pass_peak_resource_gate():
    row = sample()
    row.update(owned_ASR_active=False, shared_available=False)
    assert not qualify_active_samples([row], 100)["resource_gate_pass"]
    row.update(owned_ASR_active=True)
    assert not qualify_active_samples([row], 100)["resource_gate_pass"]


def test_cached_single_PDH_reading_does_not_count_as_independent_active_coverage():
    assert not qualify_active_samples([sample()] * 10, 100)["resource_gate_pass"]
    assert not qualify_active_samples([sample(), sample(3)], 100)["resource_gate_pass"]
    assert qualify_active_samples([sample(), sample(3), sample(5)], 100)["resource_gate_pass"]


def test_any_recorded_peak_growth_or_reserve_failure_is_not_hidden_by_final_value():
    rows = [sample(), sample(3), sample(5)]
    rows[0]["shared_bytes"] = 100 + 257 * 1024 * 1024
    assert not qualify_active_samples(rows, 100)["resource_gate_pass"]
    rows[0]["shared_bytes"] = 100
    rows[0]["free_MiB"] = 511
    assert not qualify_active_samples(rows, 100)["resource_gate_pass"]
