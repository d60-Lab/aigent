"""
Standalone merge test using files from ./temp.

Usage examples:
  # Auto-pick newest video/audio from temp and merge with padding (recommended)
  python examples/test_merge_audio_video.py

  # Specify exact files
  python examples/test_merge_audio_video.py \
    --video temp/video_norm_20251114_000156_780423.mp4 \
    --audio temp/audio_20251114_003213_277733.mp3 \
    --no-shortest

Options:
  --video <path>   Video file to merge (defaults: latest of scene_final_*.mp4, video_norm_*.mp4, video_*.mp4)
  --audio <path>   Audio file to merge (defaults: latest of audio_padded_*.mp3, audio_*.mp3, audio_silence_*.mp3)
  --output <name>  Output filename in ./output (default: merged_debug.mp4)
  --no-shortest    Do not use -shortest (prevents truncating video)
  --no-pad         Do not pad audio to video duration
"""

import argparse
import os
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from video_processor import VideoProcessor  # type: ignore


def newest(paths: List[Path]) -> Optional[Path]:
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def auto_pick(temp_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    video_candidates = []
    audio_candidates = []
    # Collect candidates
    for pat in ("scene_final_*.mp4", "video_norm_*.mp4", "video_*.mp4"):
        video_candidates += list(temp_dir.glob(pat))
    for pat in ("audio_padded_*.mp3", "audio_*.mp3", "audio_silence_*.mp3"):
        audio_candidates += list(temp_dir.glob(pat))
    v = newest(video_candidates)
    a = newest(audio_candidates)
    return v, a


def main() -> int:
    load_dotenv()
    root = Path(__file__).parent.parent
    temp = root / "temp"
    outdir = root / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Merge a video and an audio from temp")
    parser.add_argument("--video", type=str, help="video path", default="")
    parser.add_argument("--audio", type=str, help="audio path", default="")
    parser.add_argument("--output", type=str, help="output name in ./output", default="merged_debug.mp4")
    parser.add_argument("--no-shortest", action="store_true", help="Do not use -shortest")
    parser.add_argument("--no-pad", action="store_true", help="Do not pad audio to video duration")
    args = parser.parse_args()

    vp = VideoProcessor()

    vpath = Path(args.video) if args.video else None
    apath = Path(args.audio) if args.audio else None

    if not vpath or not vpath.exists() or not apath or not apath.exists():
        auto_v, auto_a = auto_pick(temp)
        if not vpath or not vpath.exists():
            vpath = auto_v
        if not apath or not apath.exists():
            apath = auto_a

    if not vpath or not vpath.exists():
        print("❌ No video found. Specify --video or generate assets first.")
        return 1
    if not apath or not apath.exists():
        print("❌ No audio found. Specify --audio or generate assets first.")
        return 1

    print(f"📼 Video: {vpath}")
    print(f"🎧 Audio: {apath}")

    vdur = vp.get_media_duration(vpath) or 0.0
    adur = vp.get_media_duration(apath) or 0.0
    print(f"   video ~ {vdur:.2f}s, audio ~ {adur:.2f}s")

    work_audio = apath
    shortest = not args.no_shortest

    # Pad audio to video duration unless disabled
    if not args.no_pad and vdur > 0 and (adur == 0.0 or adur < 0.95 * vdur):
        padded = temp / "audio_padded_manual.mp3"
        if vp.pad_audio_to_duration(apath, padded, vdur):
            work_audio = padded
            shortest = False
            adur = vp.get_media_duration(work_audio) or adur
            print(f"   padded audio -> {padded.name} ~ {adur:.2f}s")
        else:
            print("⚠️  pad failed, will merge without padding")

    out = outdir / args.output
    ok = vp.merge_audio_video(vpath, work_audio, out, audio_volume=1.0, shortest=shortest)
    if not ok:
        print("❌ Merge failed")
        return 1

    info = vp.get_video_info(out)
    print(f"✅ Merged: {out}")
    if info:
        print(f"   merged duration ~ {info.duration:.2f}s, has_audio={info.has_audio}")
    else:
        print("   merged file info unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
