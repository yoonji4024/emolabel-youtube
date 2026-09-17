"""Central configuration. Every value can be overridden via environment variable."""

import os
import platform
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

TOOL_NAME = "emolabel-youtube"
TOOL_VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# .env holds deployment-local settings (HF token etc.); real environment
# variables take precedence over it. See .env.example.
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(os.environ.get("EMOLABEL_DATA_DIR", PROJECT_ROOT / "data"))
DOWNLOADS_DIR = DATA_DIR / "downloads"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = DATA_DIR / "models"
STATIC_DIR = PROJECT_ROOT / "static"

# Keep every downloaded model inside the project so cleanup is just deleting
# data/models. HF_HOME must be set before any huggingface library is imported;
# config is imported first by every app module, so this runs early enough.
os.environ["HF_HOME"] = os.environ.get("EMOLABEL_HF_HOME", str(MODELS_DIR / "huggingface"))

FFMPEG = os.environ.get("EMOLABEL_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = os.environ.get("EMOLABEL_FFPROBE") or shutil.which("ffprobe") or "ffprobe"

_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
ASR_BACKEND = os.environ.get("EMOLABEL_ASR_BACKEND", "auto").strip().lower()
if ASR_BACKEND == "auto":
    ASR_BACKEND = "mlx" if _apple_silicon else "faster-whisper"
if ASR_BACKEND not in {"mlx", "faster-whisper"}:
    raise ValueError("EMOLABEL_ASR_BACKEND must be auto, mlx, or faster-whisper")
if ASR_BACKEND == "mlx" and not _apple_silicon:
    raise ValueError("MLX ASR requires an Apple Silicon Mac; use faster-whisper on Linux")
ASR_MODEL = os.environ.get("EMOLABEL_ASR_MODEL") or (
    "mlx-community/whisper-large-v3-turbo" if ASR_BACKEND == "mlx" else "turbo"
)
ASR_DEVICE = os.environ.get("EMOLABEL_ASR_DEVICE", "cpu")
ASR_COMPUTE_TYPE = os.environ.get("EMOLABEL_ASR_COMPUTE_TYPE", "int8")
ASR_CPU_THREADS = int(os.environ.get("EMOLABEL_ASR_CPU_THREADS", "4"))
ASR_LANGUAGE = os.environ.get("EMOLABEL_ASR_LANGUAGE", "ko")
GENDER_MODEL = os.environ.get(
    "EMOLABEL_GENDER_MODEL", "audeering/wav2vec2-large-robust-24-ft-age-gender"
)

MAX_VIDEO_DURATION_S = float(os.environ.get("EMOLABEL_MAX_VIDEO_DURATION_S", 900))
MIN_CLIP_S = float(os.environ.get("EMOLABEL_MIN_CLIP_S", 0.2))
MAX_CLIP_S = float(os.environ.get("EMOLABEL_MAX_CLIP_S", 60))

PEAKS_SAMPLES_PER_SEC = 25

# Pilot phase: keep the decision space intentionally small and balanced.
EMOTIONS = ["neutral", "angry", "sad", "happy"]
GENDER_OPTIONS = ["male", "female", "unknown"]

# Display texts for Korean annotators; stored values stay English.
EMOTION_LABELS = {
    "neutral": "Neutral(중립)",
    "angry": "Angry(화남)",
    "sad": "Sad(슬픔)",
    "happy": "Happy(행복)",
}
GENDER_LABELS = {
    "male": "Male(남성)",
    "female": "Female(여성)",
    "unknown": "Unknown(불명)",
}
VALENCE_SCALE = {"min": 1, "max": 7, "neutral": 4}
AROUSAL_SCALE = {"min": 1, "max": 7, "neutral": 4}

# HuggingFace token for the gated pyannote diarization models; when unset the
# speaker-count QC falls back to ECAPA embedding clustering.
HF_TOKEN = os.environ.get("EMOLABEL_HF_TOKEN") or os.environ.get("HF_TOKEN")
DIARIZATION_MODEL = os.environ.get(
    "EMOLABEL_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
)

# Segments whose Whisper no_speech_prob exceeds this are treated as non-speech.
ASR_NO_SPEECH_THRESHOLD = 0.6
# Below this gender-model confidence the UI preselects "unknown".
GENDER_CONFIDENCE_THRESHOLD = 0.6


def ensure_dirs() -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
