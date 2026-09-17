"""Advisory clip QC: one face? one speaker? how much of the clip is speech?

Face count: OpenCV YuNet on frames sampled evenly across the clip.
Voice ratio: Silero VAD speech segments over the clip duration.
Speaker count: the pyannote diarization pipeline (config.DIARIZATION_MODEL) when
an HF token is configured (gated model), otherwise ECAPA embeddings on VAD speech
windows clustered by
cosine distance. Approximate on short clips — advisory only, never blocks the
annotation flow.
"""

import threading
from pathlib import Path

import numpy as np

from . import config

YUNET_MODEL = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
FACE_FRAMES = 8
FACE_SCORE_THRESHOLD = 0.7

SPK_WINDOW_S = 1.5
SPK_HOP_S = 0.75
# Cosine-distance cut for "same speaker" with ECAPA embeddings (same speaker
# typically < 0.4, different speakers > 0.7).
SPK_CLUSTER_DISTANCE = 0.6
# pyannote can emit spurious speakers with almost no speech; ignore them.
PYANNOTE_MIN_SPEAKER_S = 0.5

_lock = threading.Lock()
_ecapa = None
_vad = None
_pyannote = None


def check_faces(clip_mp4: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(clip_mp4))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return {"engine": "opencv-yunet", "error": "could not read video frames"}
        detector = cv2.FaceDetectorYN_create(str(YUNET_MODEL), "", (320, 320),
                                             FACE_SCORE_THRESHOLD)
        indices = np.linspace(0, total - 1, min(FACE_FRAMES, total), dtype=int)
        counts = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            counts.append(0 if faces is None else len(faces))
    finally:
        cap.release()

    if not counts:
        return {"engine": "opencv-yunet", "error": "no frames decoded"}
    max_faces = max(counts)
    if max_faces == 0:
        verdict = "none"
    elif max_faces == 1:
        verdict = "single"
    else:
        verdict = "multiple"
    return {
        "engine": "opencv-yunet",
        "frames_sampled": len(counts),
        "faces_per_frame": counts,
        "verdict": verdict,
    }


def _get_vad():
    global _vad
    with _lock:
        if _vad is None:
            from silero_vad import load_silero_vad

            _vad = load_silero_vad()
    return _vad


def _get_ecapa():
    global _ecapa
    with _lock:
        if _ecapa is None:
            from speechbrain.inference.speaker import EncoderClassifier

            _ecapa = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(config.MODELS_DIR / "spkrec-ecapa-voxceleb"),
            )
    return _ecapa


def _speech_segments(audio: np.ndarray, sr: int) -> list[dict]:
    """Silero VAD speech spans as [{start, end}] in samples."""
    import torch
    from silero_vad import get_speech_timestamps

    model = _get_vad()
    with _lock:
        return get_speech_timestamps(torch.from_numpy(audio), model, sampling_rate=sr)


def _speakers_pyannote(wav_path: Path) -> dict:
    global _pyannote
    with _lock:
        if _pyannote is None:
            from pyannote.audio import Pipeline

            _pyannote = Pipeline.from_pretrained(
                config.DIARIZATION_MODEL, token=config.HF_TOKEN
            )
            if _pyannote is None:
                raise RuntimeError(
                    "pyannote pipeline unavailable — accept the model terms on "
                    "HuggingFace and set EMOLABEL_HF_TOKEN"
                )
    with _lock:
        diarization = _pyannote(str(wav_path))
    per_speaker = {}
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        per_speaker[speaker] = per_speaker.get(speaker, 0.0) + segment.duration
    real = {s: round(d, 2) for s, d in per_speaker.items() if d >= PYANNOTE_MIN_SPEAKER_S}
    n = len(real)
    return {
        "engine": config.DIARIZATION_MODEL,
        "estimated_speakers": n,
        "speech_s_per_speaker": dict(sorted(real.items())),
        "verdict": "no_speech" if n == 0 else "single" if n == 1 else "multiple",
    }


def _speakers_ecapa(audio: np.ndarray, sr: int, segments: list[dict]) -> dict:
    import torch

    win, hop = int(SPK_WINDOW_S * sr), int(SPK_HOP_S * sr)
    windows = []
    for seg in segments:
        chunk = audio[seg["start"]:seg["end"]]
        if len(chunk) < int(0.6 * sr):
            continue
        if len(chunk) <= win:
            windows.append(chunk)
        else:
            windows.extend(chunk[i:i + win] for i in range(0, len(chunk) - win + 1, hop))

    result = {"engine": "speechbrain-ecapa", "windows": len(windows)}
    if not windows:
        return {**result, "estimated_speakers": 0, "verdict": "no_speech"}
    if len(windows) == 1:
        return {**result, "estimated_speakers": 1, "verdict": "single",
                "note": "too little speech for a reliable estimate"}

    model = _get_ecapa()
    with _lock, torch.no_grad():
        embs = np.stack([
            model.encode_batch(torch.from_numpy(w)[None]).squeeze().numpy()
            for w in windows
        ])

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    dists = pdist(embs, metric="cosine")
    labels = fcluster(linkage(dists, method="average"),
                      t=SPK_CLUSTER_DISTANCE, criterion="distance")
    n_speakers = len(set(labels))
    return {
        **result,
        "estimated_speakers": n_speakers,
        "verdict": "single" if n_speakers == 1 else "multiple",
    }


def analyze_audio(wav_path: Path) -> tuple[dict, dict]:
    """Returns (vad, speaker) QC dicts."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    duration = len(audio) / sr
    segments = _speech_segments(audio, sr)
    speech_s = sum((s["end"] - s["start"]) / sr for s in segments)
    vad = {
        "engine": "silero-vad",
        "speech_s": round(speech_s, 2),
        "speech_ratio": round(speech_s / duration, 3) if duration else 0.0,
        "n_segments": len(segments),
    }

    if config.HF_TOKEN:
        try:
            return vad, _speakers_pyannote(wav_path)
        except Exception as e:
            fallback = _speakers_ecapa(audio, sr, segments)
            fallback["pyannote_error"] = str(e)[:300]
            return vad, fallback
    speaker = _speakers_ecapa(audio, sr, segments)
    speaker["note_engine"] = "set EMOLABEL_HF_TOKEN to use pyannote diarization"
    return vad, speaker


def run(clip_mp4: Path, wav_path: Path) -> dict:
    qc = {}
    try:
        qc["face"] = check_faces(clip_mp4)
    except Exception as e:
        qc["face"] = {"engine": "opencv-yunet", "error": str(e)[:300]}
    try:
        qc["vad"], qc["speaker"] = analyze_audio(wav_path)
    except Exception as e:
        qc["vad"] = {"engine": "silero-vad", "error": str(e)[:300]}
        qc["speaker"] = {"engine": "unavailable", "error": str(e)[:300]}
    return qc


if __name__ == "__main__":
    print("Downloading/loading QC audio models ...")
    _get_vad()
    _get_ecapa()
    if config.HF_TOKEN:
        print("HF token set — pyannote diarization will load on first use.")
    print("QC models ready.")
