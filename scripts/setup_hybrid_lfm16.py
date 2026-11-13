"""Pinned native-precision multimodal candidate; never changes live model/config."""

import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "LiquidAI/LFM2.5-VL-1.6B"
REV = "919fde3d022e3f90a4716006f993938ee8c2eb97"
WEIGHT_SHA = "7fc7458e4382fc6e558cfdda45857fbf9ab5b40a8bf199c9cd073003b14ac26d"


def main():
    data = ROOT / "local/hybrid-v1/candidates/lfm2.5-vl-1.6b"
    target = data / "source"
    target.mkdir(parents=True, exist_ok=True)
    meta = json.load(
        urllib.request.urlopen(
            "https://huggingface.co/api/models/"
            + REPO
            + "/revision/"
            + REV
            + "?blobs=true",
            timeout=30,
        )
    )
    allowed = {
        "LICENSE",
        "NOTICE",
        "README.md",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    }
    files = [
        r
        for r in meta["siblings"]
        if r["rfilename"] in allowed or r["rfilename"] == "model.safetensors"
    ]
    report = []
    start = time.monotonic()
    for row in files:
        name = row["rfilename"]
        dest = target / name
        expected = row.get("lfs", {}).get("sha256")
        if name == "model.safetensors" and (
            expected != WEIGHT_SHA or row["size"] != 3193334216
        ):
            raise ValueError("Pinned LFMsource changed")
        if not dest.exists():
            partial = dest.with_suffix(dest.suffix + ".partial")
            offset = partial.stat().st_size if partial.exists() else 0
            last = time.monotonic()
            print(json.dumps({"download": name, "offset": offset}), flush=True)
            req = urllib.request.Request(
                "https://huggingface.co/" + REPO + "/resolve/" + REV + "/" + name,
                headers={"Range": "bytes=" + str(offset) + "-"} if offset else {},
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                if offset and response.status != 206:
                    raise ValueError("Resume range ignored")
                with partial.open("ab" if offset else "wb") as stream:
                    while True:
                        block = response.read(4 << 20)
                        if not block:
                            break
                        stream.write(block)
                        if time.monotonic() - last >= 25:
                            print(
                                json.dumps({"download": name, "bytes": stream.tell()}),
                                flush=True,
                            )
                            last = time.monotonic()
            partial.replace(dest)
        with dest.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected and actual != expected:
            raise ValueError("Original checkpoint integrity mismatch")
        report.append({"name": name, "bytes": dest.stat().st_size, "sha256": actual})
    (data / "source-receipt.json").write_text(
        json.dumps(
            {
                "model": REPO,
                "revision": REV,
                "files": report,
                "seconds": time.monotonic() - start,
                "live_unchanged": True,
                "dtype": "original BF16",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps({"download_complete": True, "seconds": time.monotonic() - start}),
        flush=True,
    )


if __name__ == "__main__":
    main()
