"""ffmpeg/ffprobe helpers: probing, frame-accurate cutting, model audio, waveform peaks."""

import json
import subprocess
from pathlib import Path

import numpy as np

from . import config


class MediaError(RuntimeError):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-8:]
        raise MediaError(f"{Path(cmd[0]).name} failed: " + " | ".join(tail))
    return proc


def probe(path: Path) -> dict:
    """Return {duration_s, width, height, fps, video_codec, audio_codec}."""
    proc = _run(
        [
            config.FFPROBE,
            "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ]
    )
    info = json.loads(proc.stdout)
    out = {
        "duration_s": float(info["format"]["duration"]),
        "width": None, "height": None, "fps": None,
        "video_codec": None, "audio_codec": None,
    }
    for s in info["streams"]:
        if s["codec_type"] == "video" and out["video_codec"] is None:
            out["video_codec"] = s["codec_name"]
            out["width"] = s.get("width")
            out["height"] = s.get("height")
            num, _, den = s.get("avg_frame_rate", "0/1").partition("/")
            if den and float(den) != 0:
                out["fps"] = round(float(num) / float(den), 3)
        elif s["codec_type"] == "audio" and out["audio_codec"] is None:
            out["audio_codec"] = s["codec_name"]
    return out


def transcode_to_h264(src: Path, dst: Path) -> None:
    """One-time transcode for VP9/AV1-only downloads so the browser plays the exact cut source."""
    _run(
        [
            config.FFMPEG, "-y", "-i", str(src),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(dst),
        ]
    )


def cut_clip(src: Path, dst: Path, start_s: float, duration_s: float) -> None:
    """Frame-accurate cut: input-side -ss with re-encode."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            config.FFMPEG, "-y",
            "-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{duration_s:.3f}",
            "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(dst),
        ]
    )


def extract_wav16k(src: Path, dst: Path, start_s: float, duration_s: float) -> None:
    """16 kHz mono wav of the same segment, for the ASR and gender models."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            config.FFMPEG, "-y",
            "-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{duration_s:.3f}",
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(dst),
        ]
    )


def compute_peaks(src: Path, duration_s: float) -> dict:
    """Max-abs amplitude per bucket, normalized to [0,1], for waveform rendering."""
    rate = 8000
    proc = _run(
        [
            config.FFMPEG, "-v", "error",
            "-i", str(src),
            "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-",
        ]
    )
    samples = np.frombuffer(proc.stdout, dtype=np.int16)
    n_buckets = max(1, int(duration_s * config.PEAKS_SAMPLES_PER_SEC))
    if samples.size == 0:
        peaks = [0.0] * n_buckets
    else:
        bounds = np.linspace(0, samples.size, n_buckets + 1, dtype=np.int64)
        abs_samples = np.abs(samples.astype(np.int32))
        peaks = [
            float(abs_samples[bounds[i]:bounds[i + 1]].max()) if bounds[i] < bounds[i + 1] else 0.0
            for i in range(n_buckets)
        ]
        top = max(peaks) or 1.0
        peaks = [round(p / top, 4) for p in peaks]
    return {
        "duration_s": duration_s,
        "samples_per_sec": config.PEAKS_SAMPLES_PER_SEC,
        "peaks": peaks,
    }
