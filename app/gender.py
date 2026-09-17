"""Speaker-gender pre-labeling.

Primary: audeering/wav2vec2-large-robust-24-ft-age-gender (age + {female, male, child}).
The model needs a small custom head, reproduced from its model card.
Fallback: median-F0 heuristic via librosa.pyin.
"""

import threading
from pathlib import Path

from . import config

_lock = threading.Lock()
_model = None
_processor = None


def _load_model():
    global _model, _processor
    with _lock:
        if _model is not None:
            return
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor

        class ModelHead(nn.Module):
            def __init__(self, cfg, num_labels):
                super().__init__()
                self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
                self.dropout = nn.Dropout(cfg.final_dropout)
                self.out_proj = nn.Linear(cfg.hidden_size, num_labels)

            def forward(self, features):
                x = self.dropout(features)
                x = torch.tanh(self.dense(x))
                x = self.dropout(x)
                return self.out_proj(x)

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.wav2vec2 = Wav2Vec2Model(cfg)
                self.age = ModelHead(cfg, 1)
                self.gender = ModelHead(cfg, 3)
                # transformers >= 5 requires post_init() (the model card's
                # init_weights() pattern predates it and breaks weight tying).
                self.post_init()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return self.age(hidden), torch.softmax(self.gender(hidden), dim=1)

        _processor = Wav2Vec2Processor.from_pretrained(config.GENDER_MODEL)
        _model = AgeGenderModel.from_pretrained(config.GENDER_MODEL)
        _model.eval()


def _predict_model(wav_path: Path) -> dict:
    import soundfile as sf
    import torch

    _load_model()
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        raise RuntimeError(f"Expected 16 kHz wav, got {sr}")
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad(), _lock:
        age, gender = _model(inputs.input_values)
    probs = gender[0].tolist()  # order per model card: female, male, child
    labels = ["female", "male", "child"]
    best = max(range(3), key=lambda i: probs[i])
    return {
        "engine": config.GENDER_MODEL,
        "label": "unknown" if labels[best] == "child" else labels[best],
        "probs": {labels[i]: round(probs[i], 4) for i in range(3)},
        "age_estimate": round(float(age[0][0]) * 100, 1),  # model outputs age/100
    }


def _predict_f0(wav_path: Path) -> dict:
    import librosa
    import numpy as np

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=65, fmax=400, sr=sr, frame_length=1024
    )
    voiced = f0[voiced_flag & ~np.isnan(f0)] if f0 is not None else np.array([])
    if voiced.size < 10:
        return {"engine": "f0-heuristic", "label": "unknown", "probs": None, "median_f0_hz": None}
    median_f0 = float(np.median(voiced))
    if median_f0 < 155:
        label = "male"
    elif median_f0 > 175:
        label = "female"
    else:
        label = "unknown"
    return {
        "engine": "f0-heuristic",
        "label": label,
        "probs": None,
        "median_f0_hz": round(median_f0, 1),
    }


def predict(wav_path: Path) -> dict:
    try:
        return _predict_model(wav_path)
    except Exception as e:  # model download/inference failure -> heuristic still gives a pre-label
        result = _predict_f0(wav_path)
        result["fallback_reason"] = str(e)[:300]
        return result


if __name__ == "__main__":
    print(f"Downloading/loading gender model {config.GENDER_MODEL} ...")
    _load_model()
    print("Gender model ready.")
