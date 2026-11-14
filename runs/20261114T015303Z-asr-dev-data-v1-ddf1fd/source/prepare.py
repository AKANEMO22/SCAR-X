"""Bounded FLEURS dev stream: independent audio/transcripts, no model scores."""

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import subprocess
import tarfile
import time
import traceback
import unicodedata
import uuid
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.store import Store, encode


class LimitedStream:
    def __init__(self, source):
        self.source = source
        self.count = 0

    def read(self, size=-1):
        if size < 0:
            size = 1 << 20
        data = self.source.read(size)
        self.count += len(data)
        if self.count > 500_000_000:
            raise ValueError("Language archive stream exceeds download budget")
        return data


def main():
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-asr-dev-data-v1-" + uuid.uuid4().hex[:6]
    run, data = ROOT / "runs" / name, ROOT / "local" / name
    run.mkdir()
    data.mkdir()
    (run / "source").mkdir()
    (run / "source/prepare.py").write_bytes(Path(__file__).read_bytes())
    protocol = ROOT / "experiments/hybrid-whisper-small-native-v1.md"
    (run / "source/protocol.md").write_bytes(protocol.read_bytes())
    trace = Store(ROOT / "local/hybrid-v1")
    report = {"run_id": name, "samples": [], "languages": [], "failures": [],
              "model_inference": False, "model_score_selection": False,
              "license": "CC-BY4.0", "attribution": "FLEURS: Google, Conneau et al., 2022"}
    started = time.monotonic()
    try:
        for language, code in (("vi", "vi_vn"), ("en", "en_us")):
            url = f"https://storage.googleapis.com/xtreme_translations/FLEURS102/{code}.tar.gz"
            selected, seen, records, tsv_bytes = [], set(), None, None
            language_start = time.monotonic()
            with urlopen(Request(url, headers={"User-Agent": "SCAR-X-ASR-dev-data"}), timeout=40) as response:
                metadata = {"ETag": response.headers.get("ETag"),
                            "Last-Modified": response.headers.get("Last-Modified"),
                            "Content-Length": response.headers.get("Content-Length")}
                stream = LimitedStream(response)
                with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                    for member in archive:
                        if time.monotonic() - language_start > 600:
                            raise TimeoutError("Data stream exceeds language budget")
                        path = PurePosixPath(member.name)
                        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                            raise ValueError("Unsafe archive path")
                        if not member.isfile():
                            continue
                        if member.name.endswith("/dev.tsv"):
                            if member.size > 2_000_000:
                                raise ValueError("TSV annotation exceeds budget")
                            tsv_bytes = archive.extractfile(member).read()
                            records = {}
                            for row in csv.reader(io.StringIO(tsv_bytes.decode("utf-8")), delimiter="\t"):
                                if len(row) >= 3:
                                    records.setdefault(Path(row[1]).name, []).append(row)
                        elif "/audio/dev/" in member.name and member.name.endswith(".wav") and member.size <= 2_000_000:
                            if path.name == "14135486895615497173.wav":
                                continue
                            if records is None or len(records.get(path.name, [])) != 1:
                                raise ValueError("Missing/ambiguous independent annotation")
                            row = records[path.name][0]
                            reference = row[2]
                            normalized = " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", reference).casefold()))
                            if not normalized or normalized in seen:
                                continue
                            raw = archive.extractfile(member).read()
                            destination = data / f"{language}-{len(selected):02d}.wav"
                            destination.write_bytes(raw)
                            info = json.loads(subprocess.check_output([
                                "ffprobe", "-v", "error", "-show_streams", "-show_format",
                                "-of", "json", str(destination)], timeout=10))
                            audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
                            duration = float(audio.get("duration", info["format"]["duration"]))
                            audio_format = {
                                "sample_rate": int(audio["sample_rate"]), "channels": audio["channels"],
                                "sample_format": audio.get("sample_fmt"),
                                "bits_per_sample": audio.get("bits_per_sample"),
                                "source_audio_reencoded": False}
                            sample = {
                                "id": f"{language}-{len(selected):02d}", "language": language,
                                "source_archive": url, "archive_member": member.name,
                                "audio": str(destination), "audio_sha256": hashlib.sha256(raw).hexdigest(),
                                "audio_bytes": len(raw), "duration_seconds": duration,
                                "reference": reference, "annotation_fields": row,
                                "audio_format": audio_format, "known_engineering_sample": False,
                            }
                            selected.append(sample)
                            seen.add(normalized)
                            trace.event("ASR_candidate.data_selected", {
                                k:v for k,v in sample.items() if k != "annotation_fields"}, "orchestrator")
                            if len(selected) == 16:
                                break
                metadata["stream_bytes_read"] = stream.count
            assert len(selected) == 16
            (run / (language + "-dev.tsv")).write_bytes(tsv_bytes)
            report["languages"].append({
                "language": language, "source": url, **metadata,
                "annotation_sha256": hashlib.sha256(tsv_bytes).hexdigest(),
                "sample_count": len(selected), "seconds": time.monotonic() - language_start,
            })
            report["samples"].extend(selected)
        assert len(report["samples"]) == 32
        assert sum(r["audio_bytes"] for r in report["samples"]) <= 150_000_000
        report["status"] = "ASR_DEV_INPUTS_LOCKED_BEFORE_INFERENCE"
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["status"] = "ASR_DATA_PREPARATION_INCOMPLETE"
    finally:
        report["seconds"] = time.monotonic() - started
        with (run / "manifest.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        if not report["failures"]:
            (ROOT / "local/current-asr-dev-data-run.txt").write_text(str(run))
        trace.event("ASR_candidate.data_preparation_result", {
            k:v for k,v in report.items() if k != "samples"}, "orchestrator")
        trace.close()
        print(encode({k:report[k] for k in ("run_id", "status", "failures", "seconds")}), flush=True)
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
