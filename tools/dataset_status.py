"""Read-only progress and basic completeness check; no model inference."""
import argparse
import json
from collections import Counter
from pathlib import Path

LABELS = ["neutral", "angry", "sad", "happy"]


def inspect(root):
    counts, issues, pending = Counter(), [], []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        path = directory / "annotation.json"
        if not path.exists():
            if (directory / "clip.mp4").exists():
                pending.append(directory.name)
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            label = item["labels"]["emotion"]
            if label not in LABELS:
                raise ValueError(f"unknown emotion: {label}")
            counts[label] += 1
            if item.get("clip_id") != directory.name:
                issues.append(f"{directory.name}: clip_id mismatch")
            if not item["labels"].get("transcript", "").strip():
                issues.append(f"{directory.name}: empty transcript")
            if not item.get("annotator", {}).get("id", "").strip():
                issues.append(f"{directory.name}: missing annotator ID")
            for name in ("clip.mp4", "audio_16k.wav"):
                media = directory / name
                if not media.is_file() or media.stat().st_size == 0:
                    issues.append(f"{directory.name}: missing/empty {name}")
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
            issues.append(f"{directory.name}: {exc}")
    return counts, pending, issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/output"))
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"output directory does not exist: {args.root}")
    counts, pending, issues = inspect(args.root)
    for label in LABELS:
        print(f"{label:7} {counts[label]:3}/25 ({counts[label] * 4:3}%)")
    print(f"total   {sum(counts.values()):3}/100 ({sum(counts.values())}%)")
    print(f"Extracted but not annotated: {len(pending)}; basic issues: {len(issues)}")
    for item in pending:
        print("PENDING", item)
    for issue in issues:
        print("ISSUE", issue)
    print("Counts are saved labels, not a quality certification.")
    raise SystemExit(1 if issues else 0)
