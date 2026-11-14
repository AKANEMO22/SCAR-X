import io
import json
import zipfile
import numpy as np
import pytest
from scarx.evaluation.recovery_policy import RecoveryPolicy, ACTION_ORDER
from scarx.hybrid.learner import LocalALIF


def observation():
    return {"tool": "read_file", "reported_status": "unknown", "error_type": "",
            "empty_result": False, "spillover": False, "streak": 3, "returned_epoch": 100,
            "probe": {"kind": "requested_path_binding", "read_ok": True,
                      "reported_not_found": False, "original_path_differs": True}}


def rewrite(raw, change=None, extra=False, duplicate=False):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    if change:
        meta = json.loads(parts["metadata.json"])
        change(meta)
        parts["metadata.json"] = json.dumps(meta).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as z:
        for name, data in parts.items():
            z.writestr(name, data)
        if extra:
            z.writestr("extra", b"extra")
        if duplicate:
            z.writestr("metadata.json", parts["metadata.json"])
    return output.getvalue()


def test_gpu_policy_cold_checkpoint_preserves_exact_next_proposal_and_state():
    policy = RecoveryPolicy()
    before = policy.propose("Inspect actual source", [observation()], decision_epoch=101)
    snapshot = policy.dumps()
    restored = RecoveryPolicy.loads(snapshot)
    assert restored.dumps() == snapshot
    a = policy.propose("Inspect actual source", [observation()], decision_epoch=101)
    b = restored.propose("Inspect actual source", [observation()], decision_epoch=101)
    assert a == b == before
    assert policy.dumps() == restored.dumps()
    assert policy.model.updates == policy.model.hidden_updates == 0
    assert policy.model.device == "gpu"


def test_masks_are_explicit_and_never_claim_a_forced_choice_is_learned():
    policy = RecoveryPolicy()
    forced = policy.propose("Inspect actual source", [observation()],
                            decision_epoch=101, allowed=["defer"])
    assert forced["action"] == "defer" and forced["single_action_forced"]
    assert not forced["executed"] and not forced["policy_qualified"]
    for invalid in ([], ["other"], ["defer", "defer"]):
        with pytest.raises(ValueError):
            policy.propose("Inspect actual source", [observation()],
                           decision_epoch=101, allowed=invalid)


@pytest.mark.parametrize("mutator", [
    lambda m: m.update(encoder="old-encoder"),
    lambda m: m.update(action_order=list(reversed(ACTION_ORDER))),
    lambda m: m.update(seed=m["seed"] + 1),
    lambda m: m.update(payload_sha256="0" * 64),
    lambda m: m.update(mode="readout_only"),
])
def test_wrong_encoder_actions_checksum_or_inner_identity_are_rejected(mutator):
    with pytest.raises(ValueError):
        RecoveryPolicy.loads(rewrite(RecoveryPolicy(device="cpu").dumps(), mutator), device="cpu")


def test_separate_envelope_cannot_be_loaded_as_a_live_hybrid_checkpoint():
    raw = RecoveryPolicy(device="cpu").dumps()
    with pytest.raises((ValueError, KeyError)):
        LocalALIF.loads(raw)
    with pytest.raises(ValueError):
        RecoveryPolicy.loads(LocalALIF().dumps(), device="cpu")
    for extra, duplicate in ((True, False), (False, True)):
        with pytest.raises(ValueError):
            RecoveryPolicy.loads(rewrite(raw, extra=extra, duplicate=duplicate), device="cpu")
