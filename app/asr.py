"""Local Korean ASR: MLX on Apple Silicon; faster-whisper on Linux."""

import threading
from pathlib import Path

from . import config

_lock = threading.Lock()
_loaded = False


def _ensure_loaded() -> None:
    # mlx_whisper caches the model internally per path_or_hf_repo; importing is the
    # slow first step, and the first transcribe() pulls the weights from HuggingFace.
    global _loaded
    with _lock:
        if not _loaded:
            import mlx_whisper  # noqa: F401

            _loaded = True


def transcribe(wav_path: Path) -> dict:
    if config.ASR_BACKEND == "faster-whisper":
        from .asr_faster import transcribe as transcribe_faster

        return transcribe_faster(wav_path)
    _ensure_loaded()
    import mlx_whisper

    with _lock:
        result = mlx_whisper.transcribe(
            str(wav_path),
            path_or_hf_repo=config.ASR_MODEL,
            language=config.ASR_LANGUAGE,
            condition_on_previous_text=False,
        )

    segments = [
        {
            "start": round(float(s["start"]), 3),
            "end": round(float(s["end"]), 3),
            "text": s["text"].strip(),
            "no_speech_prob": round(float(s.get("no_speech_prob", 0.0)), 4),
        }
        for s in result.get("segments", [])
    ]
    kept = [s for s in segments if s["no_speech_prob"] <= config.ASR_NO_SPEECH_THRESHOLD]
    text = " ".join(s["text"] for s in kept).strip()
    return {
        "engine": "mlx-whisper",
        "model": config.ASR_MODEL,
        "language": config.ASR_LANGUAGE,
        "text": text,
        "segments": segments,
    }


if __name__ == "__main__":
    # Pre-warm: download the model before the first annotation session.
    print(f"Downloading/loading {config.ASR_BACKEND}: {config.ASR_MODEL} ...", flush=True)
    if config.ASR_BACKEND == "faster-whisper":
        from .asr_faster import ensure_loaded

        ensure_loaded()
    else:
        _ensure_loaded()
        from huggingface_hub import snapshot_download

        snapshot_download(config.ASR_MODEL)
    print("ASR model ready.")
