import numpy as np
from scarx.evaluation.common_probe_diagnostic import fit_probe, predict_probe


def test_common_ridge_probe_matches_closed_form_and_does_not_use_eval_labels():
    rng = np.random.default_rng(26011499)
    features = rng.normal(size=(50, 12)).astype(np.float32)
    targets = (features[:, 0] - 2 * features[:, 1] > 0).astype(int)
    original = features.copy()
    probe = fit_probe(np, features, targets)
    expected = np.linalg.solve(probe["gram"], probe["right"])
    np.testing.assert_allclose(probe["weights"], expected, rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(features, original)
    assert probe["solver"]["relative_residual"] <= 1e-9
    scores, predicted = predict_probe(np, features, probe)
    assert (predicted == targets).mean() > .8
    assert scores.shape == (50,)
