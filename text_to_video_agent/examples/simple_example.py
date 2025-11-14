"""
Simple example: Create a single-scene video with narration
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_to_video_agent import PipelineConfig, TextToVideoAgentSync

# Load environment variables
load_dotenv()
import logging

logging.getLogger().setLevel(logging.DEBUG)


def main():
    # Get API key from environment
    api_key = os.getenv("POE_API_KEY")
    if not api_key:
        print("❌ Error: POE_API_KEY not found in environment")
        print("Please set it in .env file or export POE_API_KEY=your_key")
        return

    # Configure the agent
    # Allow overriding models from environment for reliability (e.g. ITV=Runway-Gen-4-Turbo)
    tti_model = os.getenv("TEXT_TO_IMAGE_MODEL") or "FLUX-pro"
    itv_model = os.getenv("IMAGE_TO_VIDEO_MODEL") or "FLUX-pro"
    tta_model = os.getenv("TEXT_TO_AUDIO_MODEL") or "Claude-3.5-Sonnet"

    config = PipelineConfig(
        output_dir=Path("./output"),
        temp_dir=Path("./temp"),
        video_width=1920,
        video_height=1080,
        audio_volume=1.0,
        clean_temp=True,
        # 默认允许占位资源通过，以确保第一次也能产出一个视频
        strict_generation=False,
        text_to_image_model=tti_model,
        image_to_video_model=itv_model,
        text_to_audio_model=tta_model,
    )

    # Create agent
    agent = TextToVideoAgentSync(api_key, config)

    # Generate a simple video
    print("🎬 Creating a simple video...")
    print()

    print(f"🧠 Models -> TTI={config.text_to_image_model}, ITV={config.image_to_video_model}, TTA={config.text_to_audio_model}")

    description = "A beautiful sunset over a calm ocean, with gentle waves and warm golden light"
    narration = (
        "As the day comes to an end, the sun paints the sky in brilliant hues of orange and gold."
    )

    if os.getenv("USE_PLANNER") == "1":
        print("🧭 Planner enabled: decomposing description into coherent scenes")
        result = agent.create_from_description(
            description=description,
            audio_text=narration,
            output_name="sunset_video",
        )
    else:
        result = agent.create_simple_video(
            description=description,
            audio_text=narration,
            output_name="sunset_video",
        )

    # Check result
    if result.success:
        print()
        print("=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print(f"📁 Output file: {result.output_path}")
        print(f"⏱️  Duration: {result.duration:.1f} seconds")
        print(f"🎬 Scenes: {len(result.scenes)}")
        print()
        print("Metadata:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")
    else:
        print()
        print("=" * 60)
        print("❌ FAILED")
        print("=" * 60)
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
