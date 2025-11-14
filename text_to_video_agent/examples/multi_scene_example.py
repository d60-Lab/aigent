"""
Multi-scene example: Create a video with multiple scenes and transitions
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


def main():
    # Get API key from environment
    api_key = os.getenv("POE_API_KEY")
    if not api_key:
        print("❌ Error: POE_API_KEY not found in environment")
        print("Please set it in .env file or export POE_API_KEY=your_key")
        return

    # Configure the agent with transitions and model overrides via env
    tti_model = os.getenv("TEXT_TO_IMAGE_MODEL") or "FLUX-pro"
    itv_model = os.getenv("IMAGE_TO_VIDEO_MODEL") or "Runway-Gen-4-Turbo"
    tta_model = os.getenv("TEXT_TO_AUDIO_MODEL") or "Claude-3.5-Sonnet"

    config = PipelineConfig(
        output_dir=Path("./output"),
        temp_dir=Path("./temp"),
        add_transitions=True,
        transition_duration=1.5,
        audio_volume=0.8,
        clean_temp=True,
        text_to_image_model=tti_model,
        image_to_video_model=itv_model,
        text_to_audio_model=tta_model,
    )

    # Create agent
    agent = TextToVideoAgentSync(api_key, config)

    # Define multiple scenes for a short story
    scenes = [
        {
            "description": "A peaceful mountain landscape at dawn, with mist rolling through valleys and first rays of sunlight",
            "audio_text": "In the heart of the mountains, a new day begins.",
            "motion": "slow pan from left to right across the mountain range",
        },
        {
            "description": "A crystal clear river flowing through a lush green forest, sunlight filtering through trees",
            "audio_text": "The river carries stories from the peaks to the valleys below.",
            "motion": "follow the river's flow downstream",
        },
        {
            "description": "A majestic eagle soaring high above the clouds, wings spread wide against a blue sky",
            "audio_text": "Above it all, freedom takes flight.",
            "motion": "follow the eagle's graceful glide",
        },
        {
            "description": "A cozy cabin nestled in the woods with smoke rising from the chimney at golden hour",
            "audio_text": "And as evening approaches, we find our home in nature.",
            "motion": "slow zoom in toward the cabin",
        },
    ]

    # Generate the video
    print("🎬 Creating a multi-scene video with transitions...")
    print(f"📋 Scenes to process: {len(scenes)}")
    print(
        f"🧠 Models -> TTI={config.text_to_image_model}, ITV={config.image_to_video_model}, TTA={config.text_to_audio_model}"
    )
    print()

    result = agent.create_video(scenes=scenes, output_name="mountain_story")

    # Display results
    if result.success:
        print()
        print("=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print(f"📁 Output file: {result.output_path}")
        print(f"⏱️  Total duration: {result.duration:.1f} seconds")
        print(
            f"🎬 Successful scenes: {result.metadata.get('successful_scenes', 0)}/{result.metadata.get('scenes_count', 0)}"
        )

        if result.metadata.get("failed_scenes", 0) > 0:
            print(f"⚠️  Failed scenes: {result.metadata['failed_scenes']}")

        print()
        print("Scene Details:")
        for i, scene in enumerate(result.scenes, 1):
            status = "✓" if scene.final_path else "✗"
            print(f"  {status} Scene {i}: {scene.description[:50]}...")

        print()
        print("Models Used:")
        config = result.metadata.get("config", {})
        print(f"  Text→Image: {config.get('text_to_image_model', 'N/A')}")
        print(f"  Image→Video: {config.get('image_to_video_model', 'N/A')}")
        print(f"  Text→Audio: {config.get('text_to_audio_model', 'N/A')}")

    else:
        print()
        print("=" * 60)
        print("❌ FAILED")
        print("=" * 60)
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")

        print()
        print("Partial Results:")
        for i, scene in enumerate(result.scenes, 1):
            status = "✓" if scene.final_path else "✗"
            print(f"  {status} Scene {i}: {scene.description[:50]}...")


if __name__ == "__main__":
    main()
