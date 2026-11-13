"""Ephemeral WGC/loopback worker. Started only for a user-selected HWND.

Native capture is isolated from neural state. No microphone, no network.
"""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
import json
from pathlib import Path
import threading
import time
import wave


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hwnd", type=int, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--system-audio", action="store_true")
    args = p.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    from windows_capture import WindowsCapture
    import cv2

    cv2.setNumThreads(1)

    stop = threading.Event()
    lock = threading.Lock()
    started = time.monotonic()
    last = [-10.0]
    journal = (directory / "capture.jsonl").open("a", encoding="utf-8", buffering=1)

    def emit(kind, **fields):
        with lock:
            journal.write(
                json.dumps(
                    {"kind": kind, "time": time.monotonic() - started, **fields},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def audio():
        import pyaudiowpatch as pa

        api = pa.PyAudio()
        stream = None
        try:
            info = api.get_default_wasapi_loopback()
            if not info.get("isLoopbackDevice"):
                raise RuntimeError(
                    "Refuse non-loopback input; microphone is never a fallback"
                )
            rate = int(info["defaultSampleRate"])
            channels = min(2, int(info["maxInputChannels"]))
            stream = api.open(
                format=pa.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=info["index"],
                frames_per_buffer=1024,
            )
            emit("audio_started", device=info["name"], rate=rate, channels=channels)
            index = 0
            while not stop.is_set():
                start = time.monotonic() - started
                frames = []
                count = 0
                while count < rate * 10 and not stop.is_set():
                    raw = stream.read(
                        min(1024, rate * 10 - count), exception_on_overflow=False
                    )
                    frames.append(raw)
                    count += len(raw) // (channels * 2)
                if not count:
                    break
                path = directory / f"audio-{index:06d}.wav"
                temporary = path.with_suffix(".wav.part")
                with wave.open(str(temporary), "wb") as output:
                    output.setnchannels(channels)
                    output.setsampwidth(2)
                    output.setframerate(rate)
                    output.writeframes(b"".join(frames))
                temporary.replace(path)
                emit("audio", path=path.name, start=start, end=start + count / rate)
                index += 1
        except Exception as exc:
            emit("audio_unavailable", error=str(exc), microphone_used=False)
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            api.terminate()

    capture = WindowsCapture(
        window_hwnd=args.hwnd,
        monitor_index=None,
        cursor_capture=False,
        draw_border=True,
        minimum_update_interval=1000,
    )

    @capture.event
    def on_frame_arrived(frame, control):
        elapsed = time.monotonic() - started
        if stop.is_set():
            control.stop()
            return
        if elapsed - last[0] < 0.9:
            return
        last[0] = elapsed
        array = frame.frame_buffer[:, :, :3]
        height, width = array.shape[:2]
        scale = min(1.0, 1280 / max(width, height))
        image = (
            cv2.resize(array, (max(1, int(width * scale)), max(1, int(height * scale))))
            if scale < 1
            else array
        )
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        path = directory / f"frame-{int(elapsed*1000):010d}.jpg"
        temporary = path.with_suffix(".jpg.part")
        temporary.write_bytes(data.tobytes())
        temporary.replace(path)
        pointer = directory / "latest.json.part"
        pointer.write_text(
            json.dumps({"path": path.name, "time": elapsed, "wall_time": time.time()})
        )
        pointer.replace(directory / "latest.json")
        emit(
            "frame",
            path=path.name,
            start=elapsed,
            end=elapsed,
            width=width,
            height=height,
        )

    @capture.event
    def on_closed():
        emit("window_closed")
        stop.set()

    control = capture.start_free_threaded()
    audio_thread = None
    if args.system_audio:
        audio_thread = threading.Thread(target=audio, daemon=True)
        audio_thread.start()
    emit(
        "started",
        hwnd=args.hwnd,
        system_audio_requested=args.system_audio,
        microphone_used=False,
    )
    try:
        while not stop.wait(0.2):
            if (directory / "stop").exists():
                stop.set()
            heartbeat = directory / "controller-alive"
            if time.monotonic() - started > 20 and (
                not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 20
            ):
                emit("controller_lost")
                stop.set()
    finally:
        control.stop()
        if audio_thread:
            audio_thread.join(timeout=3)
        emit("stopped")
        journal.close()


if __name__ == "__main__":
    main()
