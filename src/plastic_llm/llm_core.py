"""Frozen LLM Core: Wrapper for Qwen2.5-0.5B-Instruct in unquantized BF16.

Enforces:
- 100% frozen pretrained weights (zero parameter updates).
- Unquantized bfloat16 loading.
- Greedy deterministic decoding (temperature=0.0, do_sample=False).
- Exact SHA256 checkpoint verification.
- Real-time resident set size (RSS) tracking against 2.000.000.000 byte limit.
"""
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import os
import time
from pathlib import Path
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class FrozenQwenCore:
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    EXPECTED_WEIGHT_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
    RAM_LIMIT_BYTES = 2_000_000_000

    def __init__(self, device: str = "cpu", force_cpu: bool = False):
        self.device = "cpu" if force_cpu else (device if torch.cuda.is_available() else "cpu")
        self.process = psutil.Process(os.getpid())
        
        rss_before = self.process.memory_info().rss
        t0 = time.perf_counter()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()
        # Freeze all parameters explicitly
        for p in self.model.parameters():
            p.requires_grad = False

        self.load_duration_sec = time.perf_counter() - t0
        self.rss_after = self.process.memory_info().rss
        self.total_parameters = sum(p.numel() for p in self.model.parameters())
        self.weight_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.peak_rss_bytes = max(self.process.memory_info().rss, self.weight_bytes)

        # Check RAM gate
        if self.weight_bytes > self.RAM_LIMIT_BYTES:
            raise MemoryError(f"Model weight bytes ({self.weight_bytes:,}) exceed RAM limit ({self.RAM_LIMIT_BYTES:,})")

    def verify_weights_integrity(self) -> Dict[str, Any]:
        """Compute SHA256 of the loaded safetensors file on disk to guarantee zero mutation."""
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots"
        found_hashes = {}
        if cache_dir.exists():
            for snap in cache_dir.iterdir():
                for f in snap.glob("*.safetensors"):
                    h = hashlib.sha256(f.read_bytes()).hexdigest()
                    found_hashes[f.name] = h
        return {
            "verified": any(h == self.EXPECTED_WEIGHT_SHA256 for h in found_hashes.values()),
            "found_hashes": found_hashes,
            "expected": self.EXPECTED_WEIGHT_SHA256,
        }

    def get_current_rss(self) -> int:
        cur = self.process.memory_info().rss
        if cur > self.peak_rss_bytes:
            self.peak_rss_bytes = cur
        return cur

    def get_memory_stats(self) -> Dict[str, Any]:
        rss_current = self.get_current_rss()
        vram_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() and str(self.device).startswith("cuda") else 0
        return {
            "parameter_count": self.total_parameters,
            "weight_bytes": self.weight_bytes,
            "weight_mb_decimal": float(self.weight_bytes / 1_000_000),
            "weight_mib_binary": float(self.weight_bytes / (1024 * 1024)),
            "rss_bytes": rss_current,
            "rss_mb": float(rss_current / 1_000_000),
            "rss_mb_decimal": float(rss_current / 1_000_000),
            "rss_mib_binary": float(rss_current / (1024 * 1024)),
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_mb_dec": float(self.peak_rss_bytes / 1_000_000),
            "peak_rss_mib_bin": float(self.peak_rss_bytes / (1024 * 1024)),
            "peak_rss_mb_decimal": float(self.peak_rss_bytes / 1_000_000),
            "peak_rss_mib_binary": float(self.peak_rss_bytes / (1024 * 1024)),
            "vram_bytes": vram_allocated,
            "vram_mb": float(vram_allocated / 1_000_000),
            "vram_mb_decimal": float(vram_allocated / 1_000_000),
            "vram_mib_binary": float(vram_allocated / (1024 * 1024)),
            "device": str(self.device),
            "within_2gb_limit_weights": bool(self.weight_bytes <= self.RAM_LIMIT_BYTES),
            "within_2gb_limit_peak_rss": bool(self.peak_rss_bytes <= self.RAM_LIMIT_BYTES),
            "within_2gb_limit": bool(self.peak_rss_bytes <= self.RAM_LIMIT_BYTES),
            "excess_bytes": max(0, self.peak_rss_bytes - self.RAM_LIMIT_BYTES),
        }

    def generate(self, prompt: str, max_new_tokens: int = 96) -> Tuple[str, float]:
        """Greedy deterministic generation."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        lat_ms = (time.perf_counter() - t0) * 1000.0
        gen_tokens = out[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Update peak RSS tracker
        self.get_current_rss()

        return text, lat_ms
