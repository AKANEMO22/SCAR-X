from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HybridConfig:
    ram_limit: int = 4_000_000_000
    soft_ram_limit: int = 3_850_000_000
    hidden: int = 128
    seed: int = 260113
    search_per_hour: int = 12
    conversations_per_hour: int = 3
    max_attempts: int = 2
    idle_cpu_fraction: float = 0.10
    presence_seconds: int = 600
    review_quiet_seconds: int = 30
    review_interval_seconds: int = 60
    archive_bytes: int = 10_000_000_000
    archive_seconds: int = 7 * 86400
    context_tokens: int = 16384
    frame_interval: float = 8.0
    segment_seconds: int = 10
    model_revision: str = "2fc06364715b967f1860aea9cf38778875588b17"
    asr_revision: str = "e37978b90ca9030d5170a5c07aadb050351a65bb"

    def __post_init__(self):
        if not 0 < self.soft_ram_limit < self.ram_limit <= 4_000_000_000:
            raise ValueError(
                "Whole-system RAM budget must not exceed 4,000,000,000 bytes"
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
        return asdict(self)


def default_data_dir():
    return Path(__file__).resolve().parents[3] / "local" / "hybrid-v1"
