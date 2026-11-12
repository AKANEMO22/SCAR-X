"""P0 continuation fixture shared by parent/cold worker, never a quality test."""
import hashlib
from .output_codec_p0 import CodecFixtureSession


def continuation_fingerprint(session):
    losses=[]
    while session.controller['phase']=='teaching':losses.append(session.advance_fixture())
    replay=session.next_replay_fixture()
    replay_draw=int(session.rng.integers(1_000_000))
    capacity_draw=int(session.model.capacity_rng.integers(1_000_000))
    # No expected answer/source is provided to this bounded neural generation.
    generated=session.model.generate('P0 continuation fixture only',max_words=4)
    return {'losses':losses,'replay_fixture':replay,'rng_draw':replay_draw,'capacity_rng_draw':capacity_draw,
        'generated_tokens_for_continuation_identity_only':generated['tokens'],
        'state_sha256':hashlib.sha256(session.dumps()).hexdigest(),'fixture_updates':session.fixture_updates,
        'scope':'Engineering state identity, not answer correctness or acquisition'}
