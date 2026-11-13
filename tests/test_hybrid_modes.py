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


@pytest.mark.parametrize(
    "settings",
    [
        {"batch_size": True},
        {"batch_size": 2048},
        {"microbatch_size": 256},
        {"microbatch_size": 8},
        {"flash_attention": "quantized"},
        {"cuda_graphs": 0},
    ],
)
def test_invalid_buffer_configuration_rejected(settings):
    with pytest.raises(ValueError):
        HybridConfig(**settings)


def test_execution_buffers_do_not_change_architecture_precision_or_context(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "execution_mode": "gpu",
                "batch_size": 64,
                "microbatch_size": 32,
                "flash_attention": "on",
                "cuda_graphs": False,
            }
        )
    )
    cfg = load_runtime_config(tmp_path)
    assert (
        cfg.hidden == 128
        and cfg.context_tokens == 16384
        and cfg.effective_ram_limit is None
    )
    assert cfg.microbatch_size == 32 and cfg.cuda_graphs is False


def test_candidate_identity_must_be_a_pinned_known_pair():
    with pytest.raises(ValueError):
        HybridConfig(model_name="LiquidAI/LFM2.5-VL-1.6B")
    candidate = HybridConfig(
        model_name="LiquidAI/LFM2.5-VL-1.6B",
        model_revision="919fde3d022e3f90a4716006f993938ee8c2eb97",
    )
    assert candidate.hidden == 128 and candidate.context_tokens == 16384
