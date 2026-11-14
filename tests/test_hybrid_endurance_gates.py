from scarx.evaluation.endurance_gates import qualifies


def full():
    return dict(requested=28800, elapsed=28800, cycles=96, checkpoints=16,
                updates=50, replay_seen=20, device="gpu")


def test_short_receipt_cannot_be_reported_as_eight_hours():
    row = full()
    row["elapsed"] = 240
    assert not qualifies(**row)
    row["elapsed"] = 28800
    assert qualifies(**row)


def test_updates_without_replay_or_mutable_learning_do_not_satisfy_long_profile():
    row = full()
    row["replay_seen"] = row["updates"]
    assert not qualifies(**row)
    row["updates"] = 0
    row["replay_seen"] = 0
    assert not qualifies(**row)


def test_missing_video_checkpoint_or_failure_cannot_hide_behind_elapsed_time():
    for change in (
        {"cycles": 79}, {"checkpoints": 7}, {"device": "cpu"},
        {"critical_failures": 1}, {"active_capture": True},
    ):
        assert not qualifies(**(full() | change))
