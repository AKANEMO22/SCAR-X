"""Pinned optional original-precision VLM candidate, never changes live config."""

import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REV = "15852e8c16360a2fea060d615a32b45270f8a8fc"
MODEL = "Qwen/Qwen3.5-2B"
WEIGHTS = "model.safetensors-00001-of-00001.safetensors"
EXPECTED = "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"


def main():
    target = ROOT / "local/hybrid-v1/candidates/qwen3.5-2b/source"
    target.mkdir(parents=True, exist_ok=True)
    api = json.load(
        urllib.request.urlopen(
            "https://huggingface.co/api/models/"
            + MODEL
            + "/revision/"
            + REV
            + "?blobs=true",
            timeout=30,
        )
    )
    files = [
        r
        for r in api["siblings"]
        if r["rfilename"] == WEIGHTS
        or r["rfilename"]
        in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "chat_template.jinja",
            "video_preprocessor_config.json",
                "model.safetensors.index.json",
            "README.md",
            "LICENSE",
        )
    ]
    entries = []
    started = time.monotonic()
    for entry in files:
        name = entry["rfilename"]
        path = target / name
        expected = entry.get("lfs", {}).get("sha256")
        if name == WEIGHTS and (expected != EXPECTED or entry["size"] != 4548221488):
            raise ValueError("Pinned original weight metadata mismatch")
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            offset = partial.stat().st_size if partial.exists() else 0
            req = urllib.request.Request(
                "https://huggingface.co/" + MODEL + "/resolve/" + REV + "/" + name,
                headers={"Range": "bytes=" + str(offset) + "-"} if offset else {},
            )
            print(json.dumps({"download": name, "resume_bytes": offset}), flush=True)
            with urllib.request.urlopen(req, timeout=60) as response:
                if offset and response.status != 206:
                    raise ValueError("Server ignored download resume range")
                with partial.open("ab" if offset else "wb") as stream:
                    previous = time.monotonic()
                    while True:
                        chunk = response.read(4 << 20)
                        if not chunk:
                            break
                        stream.write(chunk)
                        if time.monotonic() - previous > 25:
                            print(
                                json.dumps(
                                    {
                                        "download": name,
                                        "bytes": stream.tell(),
                                        "seconds": time.monotonic() - started,
                                    }
                                ),
                                flush=True,
                            )
                            previous = time.monotonic()
            partial.replace(path)
        actual = None
        if expected:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("Candidate source checksum mismatch: " + name)
        entries.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": actual or hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    receipt = {
        "model": MODEL,
        "revision": REV,
        "source": api["id"],
        "files": entries,
        "seconds": time.monotonic() - started,
        "live_config_changed": False,
        "quantization": False,
    }
    (target.parent / "source-receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"download_complete": True, "seconds": receipt["seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
