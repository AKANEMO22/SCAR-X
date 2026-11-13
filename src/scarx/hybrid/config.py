from dataclasses import asdict, dataclass
from pathlib import Path
import json

MAX_RAM_BYTES = 4_500_000_000
DEFAULT_SOFT_RAM_BYTES = 4_250_000_000
NATIVE_VLMS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "LiquidAI/LFM2.5-VL-1.6B": "919fde3d022e3f90a4716006f993938ee8c2eb97",
}


@dataclass(frozen=True)
class HybridConfig:
    ram_limit: int = MAX_RAM_BYTES
    soft_ram_limit: int = DEFAULT_SOFT_RAM_BYTES
    execution_mode: str = "cpu"
    gpu_device: int = 0
    searxng_enabled: bool = False
    hidden: int = 128
    seed: int = 260113
    search_per_hour: int = 12
    conversations_per_hour: int = 3
    max_attempts: int = 2
    idle_cpu_fraction: float = 0.10
    presence_seconds: int = 600
    review_quiet_seconds: int = 30
    review_interval_seconds: int = 60
    consolidation_after_seconds: int = 60
    consolidation_interval_seconds: int = 60
    sleep_replay_budget: int = 4
    rest_after_seconds: int = 180
    release_vlm_on_rest: bool = True
    gpu_keep_warm: bool = True
    batch_size: int = 128
    microbatch_size: int = 64
    flash_attention: str = "auto"
    cuda_graphs: bool = True
    archive_bytes: int = 10_000_000_000
    archive_seconds: int = 7 * 86400
    context_tokens: int = 16384
    frame_interval: float = 8.0
    segment_seconds: int = 10
    model_revision: str = "2fc06364715b967f1860aea9cf38778875588b17"
    model_name: str = "Qwen/Qwen3.5-0.8B"
    asr_revision: str = "e37978b90ca9030d5170a5c07aadb050351a65bb"

    def __post_init__(self):
        if NATIVE_VLMS.get(self.model_name) != self.model_revision:
            raise ValueError("Unregistered native VLM identity/revision")
        if (
            self.execution_mode not in ("cpu", "gpu")
            or type(self.gpu_device) is not int
            or not 0 <= self.gpu_device <= 7
        ):
            raise ValueError("Expected cpu/gpu mode and a valid CUDA device index")
        if type(self.searxng_enabled) is not bool:
            raise ValueError("Expected boolean searxng_enabled")
        if (
            type(self.batch_size) is not int
            or type(self.microbatch_size) is not int
            or not 16 <= self.microbatch_size <= self.batch_size <= 1024
            or self.flash_attention not in ("auto", "on", "off")
            or type(self.cuda_graphs) is not bool
        ):
            raise ValueError("Invalid bounded native execution buffers")
        for field in (
            "review_quiet_seconds",
            "review_interval_seconds",
            "consolidation_after_seconds",
            "consolidation_interval_seconds",
            "rest_after_seconds",
        ):
            if type(getattr(self, field)) is not int:
                raise ValueError("Expected integer scheduler timing")
        if (
            type(self.release_vlm_on_rest) is not bool
            or type(self.gpu_keep_warm) is not bool
            or type(self.sleep_replay_budget) is not int
            or not 1 <= self.sleep_replay_budget <= 4
            or not 1 <= self.consolidation_interval_seconds <= 3600
            or not self.review_quiet_seconds <= self.consolidation_after_seconds <= 3600
            or not self.review_quiet_seconds <= self.rest_after_seconds <= 86400
        ):
            raise ValueError("Invalid bounded wake/sleep policy")
        if not 0 < self.soft_ram_limit < self.ram_limit <= MAX_RAM_BYTES:
            raise ValueError(
                "Whole-system RAM budget must not exceed 4,500,000,000 bytes"
            )
        if self.hidden != 128 or not 128 <= self.context_tokens <= 16384:
            raise ValueError("Registered initial H128 / bounded working buffer")
        if not 0 < self.idle_cpu_fraction <= 0.10 or self.max_attempts != 2:
            raise ValueError("Registered idle-learning limits")
        if (
            not 10 <= self.review_quiet_seconds <= 600
            or not 30 <= self.review_interval_seconds <= 3600
        ):
            raise ValueError("Bounded post-turn review cadence")
        if (
            not 0 < self.search_per_hour <= 12
            or not 0 < self.conversations_per_hour <= 3
        ):
            raise ValueError("Registered communication limits")

    def to_dict(self):
        return asdict(self) | {
            "effective_ram_limit": self.effective_ram_limit,
            "effective_soft_ram_limit": self.effective_soft_ram_limit,
        }

    @property
    def effective_ram_limit(self):
        return None if self.execution_mode == "gpu" else self.ram_limit

    @property
    def effective_soft_ram_limit(self):
        return None if self.execution_mode == "gpu" else self.soft_ram_limit


def load_runtime_config(directory):
    path = Path(directory) / "config.json"
    if not path.exists():
        return HybridConfig()
    if path.stat().st_size > 16384:
        raise ValueError("Runtime config too large")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or set(value) - {
        "execution_mode",
        "gpu_device",
        "searxng_enabled",
        "review_quiet_seconds",
        "review_interval_seconds",
        "consolidation_after_seconds",
        "consolidation_interval_seconds",
        "sleep_replay_budget",
        "rest_after_seconds",
        "release_vlm_on_rest",
        "gpu_keep_warm",
        "batch_size",
        "microbatch_size",
        "flash_attention",
        "cuda_graphs",
        "model_name",
        "model_revision",
    }:
        raise ValueError("Unknown runtime config fields")
    return HybridConfig(**value)


def default_data_dir():
    return Path(__file__).resolve().parents[3] / "local" / "hybrid-v1"
