"""SCAR-X Round 13.1: LLM Comparison Baselines.

Implements:
1. LLM-1: ScratchTransformer20M (~19.3M parameters) trained from scratch on symbolic associations.
2. LLM-2: SmolLM2Evaluator (HuggingFaceTB/SmolLM2-135M-Instruct, 134.5M parameters) in-context reasoning benchmark.
"""
from typing import List, Dict, Tuple, Optional
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(b, t, d)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class ScratchTransformer20M(nn.Module):
    """Causal Transformer with ~19.3M parameters trained from scratch on symbolic associations."""

    def __init__(self, vocab_size: int = 300, d_model: int = 512, n_layers: int = 6, n_heads: int = 8, d_ff: int = 2048, max_len: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        mask = torch.tril(torch.ones((t, t), device=idx.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x)


class SmolLM2Evaluator:
    """Evaluates HuggingFaceTB/SmolLM2-135M-Instruct on symbolic transitive reasoning."""

    MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_ID, torch_dtype=torch.float32).to(device)
        self.model.eval()

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def evaluate_in_context_transitive(
        self,
        map_a: Dict[int, int],
        map_b: Dict[int, int],
        test_queries: List[Tuple[int, int]],
        max_symbols_in_prompt: int = 8,
    ) -> Tuple[float, List[float]]:
        """Run in-context reasoning with Mapping A and Mapping B in prompt."""
        # Format prompt
        lines_a = [f"X{k} -> Y{v}" for k, v in list(map_a.items())[:max_symbols_in_prompt]]
        lines_b = [f"Y{k} -> Z{v}" for k, v in list(map_b.items())[:max_symbols_in_prompt]]

        prompt_prefix = (
            "You are an exact transitive reasoning system.\n"
            "Given two mappings:\n"
            "Mapping A:\n" + "\n".join(lines_a) + "\n\n"
            "Mapping B:\n" + "\n".join(lines_b) + "\n\n"
            "Rule: If X_a -> Y_b and Y_b -> Z_c, then X_a maps to Z_c.\n"
        )

        latencies = []
        correct = 0

        for x_sym, target_z in test_queries:
            prompt = prompt_prefix + f"Query: What is Z for X{x_sym}?\nAnswer (only output the symbol like Z{target_z}): Z"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

            gen_text = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            # Check if target digit is in generated text
            expected_str = str(target_z)
            if expected_str in gen_text:
                correct += 1

        accuracy = correct / len(test_queries) if test_queries else 0.0
        return accuracy, latencies
