"""Head-side finite difference and canonical generator digests."""
import hashlib

import numpy as np

from scarx.evaluation.readout_mechanism import (
    delay_set,
    direction_error,
    order_pairs,
    order_with_distractors,
    query_set,
    sim_xor,
)


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def test_delta_matches_cross_entropy_finite_difference():
    # Float32 truncation at epsilon 1e-4. A wrong sign or a missing factor is O(1).
    assert direction_error() < 1e-3


def test_query_digest_matches_r3b_generator():
    sequences, labels, first = query_set(94104, 32)
    assert digest(sequences) == "924b83fc1338754859aa15139973e152526c00eb6317973e0363fb410cb4cbd9"
    assert digest(labels) == "6dc2f936f16e7cf064c404bfafef7d4a1b93d3b816f82ea0f2c9103345f14f56"
    assert digest(first) == "08f0dd54fb55c767b6872baf0d4ba1a5ac51bb91bded2aa65c0813472ee79dcf"


def test_order_and_xor_digests_match_r3b():
    early, early_y, late, late_y = order_pairs(94104, 16)
    assert digest(early) == "4166e0dc24012f1c4f660466fd32ba98ce726c86d3573c0ef9649d1a74e1523b"
    sequences, labels = sim_xor(81001, 16)
    assert digest(sequences) == "3b382f14d822652e746aeb28f1089a2b774b3a204753cbfaf128bfb194d69e5e"
    assert digest(labels) == "7baa314db2c716976528f09ea60f05b2f7e496b5b1454bc18e9586a13c915422"
    sequences, labels = delay_set(94104, 16, 6)
    assert digest(sequences) == "366314bf90415a534358f160bf94e2bf68b1796472fb59213e0bb98438ba6aea"
    sequences, labels = order_with_distractors(94104, 16, 1)
    assert digest(sequences) == "d6458f1cc6b11ef4c1f09bf9873c923c124b45a8975e5bfb75f125f8557c77db"


def test_shuffled_labels_move_a_separable_ridge():
    from scarx.evaluation.tasks21 import ridge_probe
    rng = np.random.default_rng(3)
    train = rng.normal(size=(40, 3)).astype(np.float32)
    labels = np.arange(40) % 2
    train[:, 0] += 3 * labels
    intact = ridge_probe(train, labels, train, labels)
    broken = ridge_probe(train, rng.permutation(labels), train, labels)
    assert intact > broken + 0.25
