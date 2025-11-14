"""
Create an ~30s cinematic video with audio using Runway Gen‑4 (video) + hailuo‑speech‑02 (TTS).

Usage:
  # Recommended env
  export POE_API_KEY=... 
  export IMAGE_TO_VIDEO_MODEL=Runway-Gen-4-Turbo
  export TEXT_TO_AUDIO_MODEL=hailuo-speech-02

  python examples/thirty_seconds_example.py

This script keeps temp files for inspection.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_to_video_agent import PipelineConfig, TextToVideoAgentSync  # type: ignore
from video_processor import VideoProcessor  # type: ignore


def main() -> int:
    load_dotenv()

    api_key = os.getenv("POE_API_KEY")
    if not api_key:
        print("❌ POE_API_KEY not found. Set it in .env or environment.")
        return 1

    # Models (env‑overridable)
    tti = os.getenv("TEXT_TO_IMAGE_MODEL") or "FLUX-pro"
    itv = os.getenv("IMAGE_TO_VIDEO_MODEL") or "Runway-Gen-4-Turbo"
    tta = os.getenv("TEXT_TO_AUDIO_MODEL") or "hailuo-speech-02"

    # Config: cinematic + 24fps + fadeblack transitions; keep temp for verification
    config = PipelineConfig(
        output_dir=Path("./output"),
        temp_dir=Path("./temp"),
        text_to_image_model=tti,
        image_to_video_model=itv,
        text_to_audio_model=tta,
        add_transitions=True,
        # 6 段 × ~5s，5 次转场，每次 0.2s → 总长约 30 - 1 = 29s 接近 30s
        transition_duration=0.2,
        clean_temp=False,
        cinematic_mode=True,
        cinematic_fps=24,
        cinematic_transition="fadeblack",
        target_duration=30.0,
    )

    agent = TextToVideoAgentSync(api_key, config)

    # 6 scenes × ~5s ≈ 30s; short narration lines per scene
    scenes = [
        {
            "description": "Wide horizon at golden hour over a calm sea, warm sun near the waterline, gentle swells",
            "motion": "handheld slow dolly‑in toward the horizon",
            "audio_text": "黄昏的海平面缓缓起伏，金色的阳光洒满水面。",
        },
        {
            "description": "Drone rising, tilting up from shimmering water to glowing sky, light clouds",
            "motion": "rising tilt up, smooth and steady",
            "audio_text": "镜头抬升，水面与天空连成一片，云层被夕阳染亮。",
        },
        {
            "description": "Tracking parallel to shoreline, waves curl and break with soft spray, footprints visible",
            "motion": "sideways tracking parallel to shore",
            "audio_text": "沿着海岸线平移，浪花轻轻卷起，沙滩上留下脚印。",
        },
        {
            "description": "Low angle of waves rolling over dark rocks, droplets sparkling in backlight",
            "motion": "low angle push‑in toward the rocks",
            "audio_text": "低机位靠近礁石，水珠在逆光里闪烁。",
        },
        {
            "description": "Close‑up foam patterns and retreating water over wet sand, bokeh highlights",
            "motion": "macro‑like steady push‑in",
            "audio_text": "浪沫在沙面上勾勒出细腻纹理，又轻轻退去。",
        },
        {
            "description": "Silhouette couple walking into sunset along the shore, long shadows, tranquil mood",
            "motion": "handheld follow from behind, slow pace",
            "audio_text": "一对行人走向落日，海风温柔，时间也缓了下来。",
        },
    ]

    print(f"🧠 Models -> TTI={tti}, ITV={itv}, TTA={tta}")
    print("🎬 Generating ~30s cinematic video with audio…")
    result = agent.create_video(scenes, output_name="thirty_seconds_demo")

    if not result.success:
        print("❌ FAILED\nErrors:")
        for e in result.errors:
            print(" -", e)
        return 1

    print(f"✅ Output: {result.output_path}")

    # Optional: verify per‑scene durations and sum
    vp = VideoProcessor()
    total = 0.0
    for i, s in enumerate(result.scenes, 1):
        if s.final_path:
            dur = vp.get_media_duration(s.final_path) or 0.0
            total += dur
            print(f"  Scene {i}: {s.final_path.name} ~ {dur:.2f}s")
    print(f"≈ Total length: {total:.2f}s (target ~30s)")
    # 快速检查最终文件是否包含音轨
    info = vp.get_video_info(result.output_path) if result.output_path else None
    if info:
        print(f"Final has_audio={info.has_audio}")
    print("Temp kept at ./temp for inspection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
