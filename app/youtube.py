"""yt-dlp integration: URL validation, duration probe, download with cache."""

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp

from . import config, media

ALLOWED_HOSTS = {
    "www.youtube.com", "youtube.com", "m.youtube.com",
    "youtu.be", "www.youtu.be", "music.youtube.com",
}


class YouTubeError(RuntimeError):
    pass


def validate_url(url: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise YouTubeError(f"Not a YouTube URL (host: {host or 'none'})")


def _ydl_opts(download: bool) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "ffmpeg_location": str(Path(config.FFMPEG).parent),
    }
    if download:
        opts.update(
            {
                "format": "bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/b[ext=mp4]/b",
                "merge_output_format": "mp4",
                "outtmpl": str(config.DOWNLOADS_DIR / "%(id)s.%(ext)s"),
            }
        )
    return opts


def info_path(video_id: str) -> Path:
    return config.DOWNLOADS_DIR / f"{video_id}.info.json"


def video_path(video_id: str) -> Path:
    return config.DOWNLOADS_DIR / f"{video_id}.mp4"


def load_video(url: str) -> dict:
    """Download (or reuse cached) video; return metadata dict stored in {id}.info.json."""
    validate_url(url)

    try:
        with yt_dlp.YoutubeDL(_ydl_opts(download=False)) as ydl:
            probe_info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise YouTubeError(str(e)) from e

    video_id = probe_info["id"]
    duration = probe_info.get("duration")
    if duration and duration > config.MAX_VIDEO_DURATION_S:
        raise YouTubeError(
            f"Video is {duration:.0f}s long; max allowed is "
            f"{config.MAX_VIDEO_DURATION_S:.0f}s. This tool targets Shorts."
        )

    mp4 = video_path(video_id)
    meta_file = info_path(video_id)
    if mp4.exists() and meta_file.exists():
        return json.loads(meta_file.read_text(encoding="utf-8"))

    try:
        with yt_dlp.YoutubeDL(_ydl_opts(download=True)) as ydl:
            ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        raise YouTubeError(str(e)) from e

    if not mp4.exists():
        raise YouTubeError("Download finished but no mp4 was produced")

    probed = media.probe(mp4)
    if probed["video_codec"] != "h264":
        tmp = mp4.with_suffix(".h264.mp4")
        media.transcode_to_h264(mp4, tmp)
        tmp.replace(mp4)
        probed = media.probe(mp4)

    meta = {
        "video_id": video_id,
        "url": probe_info.get("webpage_url") or url,
        "title": probe_info.get("title"),
        "channel": probe_info.get("channel") or probe_info.get("uploader"),
        "duration_s": probed["duration_s"],
        "width": probed["width"],
        "height": probed["height"],
        "fps": probed["fps"],
        "video_codec": probed["video_codec"],
        "audio_codec": probed["audio_codec"],
        "downloaded_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def get_meta(video_id: str) -> dict:
    meta_file = info_path(video_id)
    if not meta_file.exists():
        raise YouTubeError(f"Unknown video id: {video_id}")
    return json.loads(meta_file.read_text(encoding="utf-8"))


def get_peaks(video_id: str) -> dict:
    """Waveform peaks, computed once and cached beside the download."""
    cache = config.DOWNLOADS_DIR / f"{video_id}.peaks.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    meta = get_meta(video_id)
    peaks = media.compute_peaks(video_path(video_id), meta["duration_s"])
    cache.write_text(json.dumps(peaks), encoding="utf-8")
    return peaks
