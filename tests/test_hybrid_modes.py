"""Mode configuration validation/persistence; real GPU parity is measured separately."""

import json
import pytest
from scarx.hybrid.config import HybridConfig, load_runtime_config


def test_default_cpu_and_explicit_gpu_keep_precision_independent_settings(tmp_path):
    assert load_runtime_config(tmp_path).execution_mode == "cpu"
    (tmp_path / "config.json").write_text(
        json.dumps({"execution_mode": "gpu", "gpu_device": 0, "searxng_enabled": True})
    )
    cfg = load_runtime_config(tmp_path)
    assert cfg.execution_mode == "gpu" and cfg.gpu_device == 0 and cfg.searxng_enabled
    assert cfg.effective_ram_limit is None and cfg.effective_soft_ram_limit is None
    assert (
        cfg.ram_limit == 4_500_000_000
        and cfg.hidden == 128
        and cfg.context_tokens == 16384
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"execution_mode": "cpu", "searxng_enabled": True})
    )
    assert load_runtime_config(tmp_path).execution_mode == "cpu"
    assert load_runtime_config(tmp_path).effective_ram_limit == 4_500_000_000


@pytest.mark.parametrize(
    "settings",
    [
        {"execution_mode": "auto"},
        {"execution_mode": "GPU"},
        {"gpu_device": True},
        {"gpu_device": -1},
        {"gpu_device": 8},
        {"searxng_enabled": 1},
    ],
)
def test_invalid_placement_refused(settings):
    with pytest.raises(ValueError):
        HybridConfig(**settings)


def test_config_cannot_silently_raise_budget_or_change_neural_architecture(tmp_path):
    for key in ("ram_limit", "hidden", "quantization", "unknown"):
        (tmp_path / "config.json").write_text(json.dumps({key: 99}))
        with pytest.raises(ValueError):
            load_runtime_config(tmp_path)
