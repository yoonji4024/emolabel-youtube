"""Clip output layout: data/output/{clip_id}/ with clip.mp4, audio_16k.wav, prelabels.json, annotation.json."""

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import config

CLIP_ID_RE = re.compile(r"^(?P<video_id>[A-Za-z0-9_-]{6,})_(?P<start_ms>\d{7,})_(?P<end_ms>\d{7,})$")


class StoreError(RuntimeError):
    pass


def make_clip_id(video_id: str, start_s: float, end_s: float) -> str:
    return f"{video_id}_{round(start_s * 1000):07d}_{round(end_s * 1000):07d}"


def parse_clip_id(clip_id: str) -> dict:
    m = CLIP_ID_RE.match(clip_id)
    if not m:
        raise StoreError(f"Invalid clip id: {clip_id}")
    return {
        "video_id": m["video_id"],
        "start_s": int(m["start_ms"]) / 1000,
        "end_s": int(m["end_ms"]) / 1000,
    }


def clip_dir(clip_id: str) -> Path:
    parse_clip_id(clip_id)  # rejects path-traversal garbage before touching the filesystem
    return config.OUTPUT_DIR / clip_id


def clip_video_path(clip_id: str) -> Path:
    return clip_dir(clip_id) / "clip.mp4"


def clip_wav_path(clip_id: str) -> Path:
    return clip_dir(clip_id) / "audio_16k.wav"


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_prelabels(clip_id: str, prelabels: dict) -> None:
    _atomic_write_json(clip_dir(clip_id) / "prelabels.json", prelabels)


def read_prelabels(clip_id: str) -> dict | None:
    p = clip_dir(clip_id) / "prelabels.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_annotation(clip_id: str) -> dict | None:
    p = clip_dir(clip_id) / "annotation.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_annotation(clip_id: str, labels: dict, annotator_id: str,
                    source_meta: dict, clip_media: dict) -> dict:
    d = clip_dir(clip_id)
    if not (d / "clip.mp4").exists():
        raise StoreError(f"Clip {clip_id} has no clip.mp4; extract it first")
    seg = parse_clip_id(clip_id)
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    existing = read_annotation(clip_id)
    prelabels = read_prelabels(clip_id) or {}
    qc = prelabels.pop("qc", None)
    record = {
        "schema_version": 1,
        "tool": {"name": config.TOOL_NAME, "version": config.TOOL_VERSION},
        "clip_id": clip_id,
        "source": {
            "url": source_meta.get("url"),
            "video_id": seg["video_id"],
            "title": source_meta.get("title"),
            "channel": source_meta.get("channel"),
            "source_duration_s": source_meta.get("duration_s"),
            "downloaded_at": source_meta.get("downloaded_at"),
        },
        "segment": {
            "start_s": seg["start_s"],
            "end_s": seg["end_s"],
            "duration_s": round(seg["end_s"] - seg["start_s"], 3),
        },
        "media": {"clip_file": "clip.mp4", "audio_file": "audio_16k.wav", **clip_media},
        # QC verification (face/voice/speaker checks) gets its own section;
        # prelabels keeps only the model suggestions the annotator corrects.
        "prelabels": prelabels or None,
        "qc": qc,
        "labels": labels,
        "annotator": {"id": annotator_id},
        "timestamps": {
            "clip_created_at": (existing or {}).get("timestamps", {}).get("clip_created_at")
            or datetime.fromtimestamp((d / "clip.mp4").stat().st_mtime)
            .astimezone().isoformat(timespec="seconds"),
            "annotation_saved_at": now,
        },
    }
    _atomic_write_json(d / "annotation.json", record)
    return record


def clip_summary(clip_id: str) -> dict:
    seg = parse_clip_id(clip_id)
    ann = read_annotation(clip_id)
    return {
        "clip_id": clip_id,
        "video_id": seg["video_id"],
        "start_s": seg["start_s"],
        "end_s": seg["end_s"],
        "annotated": ann is not None,
        "emotion": (ann or {}).get("labels", {}).get("emotion"),
    }


def list_clips(video_id: str | None = None) -> list[dict]:
    out = []
    for d in sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else []:
        if not d.is_dir() or not CLIP_ID_RE.match(d.name):
            continue
        if not (d / "clip.mp4").exists():
            continue  # aborted extraction leftovers
        if video_id and not d.name.startswith(f"{video_id}_"):
            continue
        out.append(clip_summary(d.name))
    return out


def delete_clip(clip_id: str) -> None:
    d = clip_dir(clip_id)
    if not d.exists():
        raise StoreError(f"No such clip: {clip_id}")
    if (d / "annotation.json").exists():
        raise StoreError(f"Clip {clip_id} is annotated; refusing to delete")
    shutil.rmtree(d)
