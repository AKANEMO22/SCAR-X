"""Native-precision repacking. Whisper never takes an intermediate FP16 path."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.store import Store, encode

DATA = ROOT / "local/hybrid-v1"
MAPPING = {
    "self_attn.k_proj": "attn.key",
    "self_attn.q_proj": "attn.query",
    "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out",
    "self_attn_layer_norm": "attn_ln",
    "encoder_attn.q_proj": "cross_attn.query",
    "encoder_attn.k_proj": "cross_attn.key",
    "encoder_attn.v_proj": "cross_attn.value",
    "encoder_attn.out_proj": "cross_attn.out",
    "encoder_attn_layer_norm": "cross_attn_ln",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    "encoder.layer_norm.bias": "encoder.ln_post.bias",
    "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.layer_norm.bias": "decoder.ln.bias",
    "decoder.layer_norm.weight": "decoder.ln.weight",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
}


def whisper_name(name):
    if name == "proj_out.weight":
        return "decoder.proj.weight"
    parts = name.split(".")[1:]
    if len(parts) > 1 and parts[1] == "layers":
        return ".".join(
            parts[:1]
            + ["blocks"]
            + parts[2:3]
            + [MAPPING[".".join(parts[3:-1])]]
            + parts[-1:]
        )
    key = ".".join(parts)
    return MAPPING.get(key, key)


def convert_whisper(source, target, filters_path):
    config = json.loads((source / "config.json").read_text())
    vocab = json.loads((source / "vocab.json").read_text(encoding="utf-8"))
    symbols = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = symbols.copy()
    extra = 0
    for byte in range(256):
        if byte not in symbols:
            symbols.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    byte_decoder = {chr(point): byte for byte, point in zip(symbols, codepoints)}
    with np.load(filters_path, allow_pickle=False) as filters:
        mel = np.asarray(filters["mel_" + str(config["num_mel_bins"])], dtype="<f4")
    manifest = []
    with (
        target.open("wb") as output,
        safe_open(source / "model.safetensors", framework="np") as checkpoint,
    ):
        ints = [
            0x67676D6C,
            config["vocab_size"],
            config["max_source_positions"],
            config["d_model"],
            config["encoder_attention_heads"],
            config["encoder_layers"],
            config["max_target_positions"],
            config["d_model"],
            config["decoder_attention_heads"],
            config["decoder_layers"],
            config["num_mel_bins"],
            0,
        ]
        output.write(struct.pack("<12i", *ints))
        output.write(struct.pack("<2i", *mel.shape))
        output.write(mel.tobytes())
        output.write(struct.pack("<i", len(vocab)))
        for word, _ in sorted(vocab.items(), key=lambda kv: kv[1]):
            raw = bytes(byte_decoder[ch] for ch in word)
            output.write(struct.pack("<i", len(raw)))
            output.write(raw)
        for source_name in checkpoint.keys():
            data = checkpoint.get_tensor(source_name)
            if data.dtype != np.float32:
                raise ValueError("Whisper source must be native FP32")
            name = whisper_name(source_name)
            data = data.squeeze()
            if name in ("encoder.conv1.bias", "encoder.conv2.bias"):
                data = data.reshape(-1, 1)
            raw_name = name.encode()
            output.write(struct.pack("<3i", data.ndim, len(raw_name), 0))
            output.write(struct.pack("<" + "i" * data.ndim, *reversed(data.shape)))
            output.write(raw_name)
            raw = np.ascontiguousarray(data, dtype="<f4").tobytes()
            output.write(raw)
            manifest.append(
                {
                    "source": source_name,
                    "target": name,
                    "shape": list(data.shape),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "dtype": "F32",
                }
            )
    # Independent readback of every serialized tensor, not only dtype labels.
    with target.open("rb") as handle:
        handle.seek(48)
        rows, cols = struct.unpack("<2i", handle.read(8))
        handle.seek(rows * cols * 4, 1)
        tokens = struct.unpack("<i", handle.read(4))[0]
        for _ in range(tokens):
            length = struct.unpack("<i", handle.read(4))[0]
            handle.seek(length, 1)
        for expected in manifest:
            dims, length, ftype = struct.unpack("<3i", handle.read(12))
            shape = struct.unpack("<" + "i" * dims, handle.read(4 * dims))
            name = handle.read(length).decode()
            size = int(np.prod(shape)) * 4
            raw = handle.read(size)
            if (
                ftype != 0
                or name != expected["target"]
                or hashlib.sha256(raw).hexdigest() != expected["sha256"]
            ):
                raise ValueError("Whisper exact FP32 tensor readback failed")
        if handle.read(1):
            raise ValueError("Unexpected bytes in Whisper export")
    return manifest


def qwen_inventory(source):
    report = []
    for path in sorted(source.glob("*.safetensors")):
        with safe_open(path, framework="np") as checkpoint:
            for name in checkpoint.keys():
                tensor = checkpoint.get_slice(name)
                report.append(
                    {
                        "name": name,
                        "dtype": tensor.get_dtype(),
                        "shape": list(tensor.get_shape()),
                    }
                )
    if any(row["dtype"] not in ("BF16", "F32") for row in report):
        raise ValueError("Unexpected native dtype")
    return report


def main():
    store = Store(DATA)
    source = DATA / "models/Qwen3.5-0.8B"
    output = DATA / "models/native"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "native-conversion-v1",
        "quantization": False,
        "whisper": {},
        "qwen": {},
    }
    try:
        store.event(
            "conversion_started",
            {"precision": "Whisper direct FP32; Qwen BF16/F32; no quantization"},
            "orchestrator",
        )
        report["whisper"]["tensors"] = convert_whisper(
            DATA / "models/whisper-base",
            output / "whisper-base-f32.bin",
            DATA / "sources/whisper/assets/mel_filters.npz",
        )
        store.event(
            "whisper_native_readback_pass",
            {
                "tensors": len(report["whisper"]["tensors"]),
                "all_tensor_bytes_equal": True,
            },
            "orchestrator",
        )
        report["qwen"]["source_inventory"] = qwen_inventory(source)
        converter = DATA / "sources/llama.cpp-b12360/convert_hf_to_gguf.py"
        for mode, name in (
            ([], "qwen-native-bf16.gguf"),
            (["--mmproj"], "qwen-vision-native-bf16.gguf"),
        ):
            command = [
                sys.executable,
                str(converter),
                str(source),
                "--outtype",
                "bf16",
                "--outfile",
                str(output / name),
            ] + mode
            store.event(
                "converter_running",
                {"command": command, "no_mtp_omission_requested": True},
                "orchestrator",
            )
            with (DATA / (name + ".conversion.log")).open("w", encoding="utf-8") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        # Use matching GGUF parser without installing into the shared environment.
        sys.path.insert(0, str(converter.parent / "gguf-py"))
        import gguf

        targets = []
        for path in output.glob("qwen-*.gguf"):
            reader = gguf.GGUFReader(str(path))
            inventory = [
                dict(
                    name=t.name, type=t.tensor_type.name, shape=list(map(int, t.shape))
                )
                for t in reader.tensors
            ]
            if any(t["type"] not in ("BF16", "F32", "F64") for t in inventory):
                raise ValueError("Reduced-precision/quantized GGUF tensor")
            targets.append(
                {"file": path.name, "bytes": path.stat().st_size, "tensors": inventory}
            )
            del reader
        report["qwen"]["exports"] = targets
        report["qwen"][
            "status"
        ] = "TYPES_CHECKED; full numerical/coverage parity still required by P0"
        (DATA / "native-conversion.json").write_text(encode(report), encoding="utf-8")
        store.event(
            "native_exports_created",
            {"whisper_exact_readback": True, "qwen_status": report["qwen"]["status"]},
            "orchestrator",
        )
    except Exception as exc:
        store.event("conversion_failed", {"error": str(exc)}, "orchestrator")
        raise
    finally:
        store.close()


if __name__ == "__main__":
    main()
