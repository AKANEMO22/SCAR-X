"""Registered engineering coverage; never a semantic/lifelong quality verdict."""


def qualifies(*, requested, elapsed, cycles, checkpoints, updates, replay_seen,
              device, critical_failures=0, active_capture=False):
    if requested not in (90, 240, 28800):
        raise ValueError("Unregistered endurance horizon")
    long = requested == 28800
    return bool(
        elapsed >= requested
        and cycles >= (80 if long else 1)
        and checkpoints >= (8 if long else 1)
        and updates > 0
        and (not long or updates > replay_seen)
        and device == "gpu" and not critical_failures and not active_capture
    )
