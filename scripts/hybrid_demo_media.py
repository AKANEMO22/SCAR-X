"""Real inference inputs, NOT a research dataset or a mock model response.

Public human Vietnamese speech from FLEURS102 (CC-BY4.0), independently supplied
transcription, and explicitly synthetic visual shapes for engineering checks.
Reads only the first TSV+one audio from the streamed archive, not the corpus.
"""

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
import tarfile
from urllib.request import urlopen
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.store import Store, encode


def main():
    directory = ROOT / "local/hybrid-v1"
    samples = directory / "samples"
    samples.mkdir(exist_ok=True)
    store = Store(directory)
    url = "https://storage.googleapis.com/xtreme_translations/FLEURS102/vi_vn.tar.gz"
    transcript = None
    member_name = None
    wav = samples / "fleurs-vi.wav"
    tsv = None
    store.event(
        "engineering_media_preparation",
        {
            "source": url,
            "license": "CC-BY4.0",
            "scope": "one public speech sample + synthetic shapes; not P1",
        },
        "orchestrator",
    )
    with (
        urlopen(url, timeout=30) as stream,
        tarfile.open(fileobj=stream, mode="r|gz") as archive,
    ):
        for member in archive:
            if member.name.endswith("/dev.tsv"):
                tsv = archive.extractfile(member).read().decode("utf-8")
            elif member.name.endswith(".wav"):
                if member.size > 2_000_000:
                    continue
                member_name = member.name
                wav.write_bytes(archive.extractfile(member).read())
                break
    if not tsv or not member_name:
        raise ValueError("Missing independent speech annotation")
    matching = [
        row
        for row in csv.reader(io.StringIO(tsv), delimiter="\t")
        if Path(member_name).name in row
    ]
    if len(matching) != 1:
        raise ValueError("Could not match source audio to independent transcript")
    row = matching[0]
    transcript = row[2]
    first = Image.new("RGB", (384, 256), "white")
    draw = ImageDraw.Draw(first)
    draw.rectangle((30, 60, 150, 200), fill="red")
    draw.ellipse((230, 70, 340, 180), fill="blue")
    first.save(samples / "scene-first.png")
    second = Image.new("RGB", (384, 256), "white")
    draw = ImageDraw.Draw(second)
    draw.rectangle((30, 60, 150, 200), fill="green")
    draw.ellipse((230, 70, 340, 180), fill="yellow")
    second.save(samples / "scene-second.png")
    duration = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(wav)]
        )
    )["format"]["duration"]
    midpoint = float(duration) / 2
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-loop",
        "1",
        "-i",
        str(samples / "scene-first.png"),
        "-loop",
        "1",
        "-i",
        str(samples / "scene-second.png"),
        "-i",
        str(wav),
        "-filter_complex",
        f"[0:v][1:v]blend=all_expr=if(lt(T\,{midpoint})\,A\,B)[v]",
        "-map",
        "[v]",
        "-map",
        "2:a",
        "-t",
        str(duration),
        "-r",
        "8",
        "-c:v",
        "ffv1",
        "-g",
        "1",
        "-c:a",
        "pcm_s16le",
        str(samples / "engineering-vi-video.mkv"),
    ]
    subprocess.run(command, check=True, timeout=60)
    manifest = {
        "kind": "engineering_fixture_not_research",
        "audio_source": url,
        "archive_member": member_name,
        "license": "CC-BY4.0",
        "attribution": "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech, Google",
        "audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
        "independent_transcript": transcript,
        "transcript_source": "FLEURS dev.tsv",
        "duration": float(duration),
        "visual_source": "script-authored synthetic shapes",
        "first_scene": "red rectangle left; blue circle right",
        "second_scene": "green rectangle left; yellow circle right",
        "scene_change_seconds": midpoint,
        "video": str(samples / "engineering-vi-video.mkv"),
    }
    (samples / "engineering-manifest.json").write_text(
        encode(manifest), encoding="utf-8"
    )
    store.event("engineering_media_ready", manifest, "orchestrator")
    store.close()
    print(encode(manifest), flush=True)


if __name__ == "__main__":
    main()
