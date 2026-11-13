"""Component scheduling uses physical GPU headroom, never a CPU weight fallback."""

from types import SimpleNamespace
import pytest
from scarx.hybrid.device_budget import gpu_headroom, audio_requires_release
from scarx.hybrid.backends import NativeBackends
from scarx.hybrid.config import HybridConfig


def test_no_release_when_audio_and_reserve_fit_exactly():
    assert audio_requires_release({"free_MiB": 1049})
    assert not audio_requires_release({"free_MiB": 1050})


def test_gpu_sampler_validates_one_device_without_shell(monkeypatch):
    received = []

    def run(cmd, **kwargs):
        received.append((cmd, kwargs))
        return SimpleNamespace(stdout="1100, 6141\n")

    monkeypatch.setattr("scarx.hybrid.device_budget.subprocess.run", run)
    result = gpu_headroom(0)
    assert result["free_MiB"] == 1100 and result["reserve_MiB"] == 512
    assert "--id=0" in received[0][0] and not received[0][1].get("shell", False)


@pytest.mark.parametrize(
    "output", ["-1,6141", "7000,6141", "not available,6141", "400,6141\n500,6141"]
)
def test_bad_gpu_budget_never_silently_falls_back_to_cpu(monkeypatch, output):
    monkeypatch.setattr(
        "scarx.hybrid.device_budget.subprocess.run",
        lambda *a, **kw: SimpleNamespace(stdout=output),
    )
    with pytest.raises((ValueError, RuntimeError)):
        gpu_headroom()


def test_gpu_pressure_releases_whole_owned_vlm_and_retains_command(
    tmp_path, monkeypatch
):
    events = []
    terminated = []
    backend = NativeBackends(
        tmp_path,
        SimpleNamespace(event=lambda *x: events.append(x)),
        SimpleNamespace(limit=None, sample=lambda: {"rss_bytes": 1_000_000}),
        HybridConfig(execution_mode="gpu"),
    )
    process = SimpleNamespace(
        poll=lambda: None,
        terminate=lambda: terminated.append(True),
        wait=lambda timeout: None,
    )
    backend.vlm = process
    backend.vlm_command = ["owned-native-model"]
    monkeypatch.setattr(
        "scarx.hybrid.backends.gpu_headroom",
        lambda _: {"free_MiB": 700, "reserve_MiB": 512},
    )
    backend.pause_vlm_for_audio()
    assert (
        terminated
        and backend.vlm is None
        and backend.vlm_command == ["owned-native-model"]
    )
    assert events[-1][1]["reason"] == "GPU workspace reserve"


def test_cpu_scheduler_does_not_query_discrete_gpu(tmp_path, monkeypatch):
    backend = NativeBackends(
        tmp_path,
        SimpleNamespace(),
        SimpleNamespace(limit=4_500_000_000, sample=lambda: {"rss_bytes": 10_000_000}),
        HybridConfig(execution_mode="cpu"),
    )

    def forbidden(_):
        raise AssertionError("CPU mode probed GPU scheduling")

    monkeypatch.setattr("scarx.hybrid.backends.gpu_headroom", forbidden)
    backend.pause_vlm_for_audio()
