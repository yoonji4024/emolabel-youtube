"""Linux ASR adapter; same stored transcript schema as the MLX backend."""
import threading
from pathlib import Path

from . import config

_lock = threading.RLock()
_model = None


def ensure_loaded():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel

            _model = WhisperModel(
                config.ASR_MODEL, device=config.ASR_DEVICE,
                compute_type=config.ASR_COMPUTE_TYPE,
                cpu_threads=config.ASR_CPU_THREADS,
                download_root=str(config.MODELS_DIR / "faster-whisper"),
            )
    return _model


def transcribe(wav_path: Path) -> dict:
    model = ensure_loaded()
    with _lock:
        iterator, _ = model.transcribe(
            str(wav_path), language=config.ASR_LANGUAGE,
            condition_on_previous_text=False, vad_filter=False,
        )
        # Inference is lazy: consume the generator while still holding the lock.
        segments = [
            {"start": round(float(s.start), 3), "end": round(float(s.end), 3),
             "text": s.text.strip(), "no_speech_prob": round(float(s.no_speech_prob), 4)}
            for s in iterator
        ]
    kept = [s for s in segments if s["no_speech_prob"] <= config.ASR_NO_SPEECH_THRESHOLD]
    return {
        "engine": "faster-whisper", "model": config.ASR_MODEL,
        "language": config.ASR_LANGUAGE,
        "text": " ".join(s["text"] for s in kept).strip(), "segments": segments,
    }
