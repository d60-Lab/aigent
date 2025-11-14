"""
Example: Test Image -> Video via PoeClientSync
Generates a small placeholder PNG first (via Pillow), then requests video.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poe_client import PoeClientSync  # type: ignore


def _make_test_png(path: Path):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("❌ Pillow not available; please install requirements.txt")
        return False

    img = Image.new("RGB", (640, 360), (32, 64, 96))
    d = ImageDraw.Draw(img)
    d.text((20, 20), "Test Image", fill=(240, 240, 240))
    img.save(path, format="PNG")
    return True


def main():
    load_dotenv()
    api_key = os.getenv("POE_API_KEY")
    if not api_key:
        print("❌ POE_API_KEY not found. Set it in .env or environment.")
        return 1

    base_dir = Path(__file__).parent.parent
    out_dir = base_dir / "output"
    temp_dir = base_dir / "temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    image_path = temp_dir / "test_input_image.png"
    if not image_path.exists():
        if not _make_test_png(image_path):
            return 1

    out_path = out_dir / "test_image_to_video.mp4"

    client = PoeClientSync(api_key)
    model = os.getenv("IMAGE_TO_VIDEO_MODEL") or "FLUX-pro"

    motion = "slow cinematic pan from left to right"
    print(f"🎬 Generating video from image {image_path.name} with motion: {motion}")
    print(f"🧠 Using bot/model: {model}")

    description = "A serene lake at sunset with purple-orange sky, calm water, gentle waves"
    res = client.image_to_video(
        image_path=image_path,
        description=description,
        motion_prompt=motion,
        model=model,
        output_path=out_path,
    )

    print("success:", res.success)
    print("placeholder:", res.metadata.get("placeholder"), "reason:", res.metadata.get("placeholder_reason"))
    print("resolved_video_url:", res.metadata.get("resolved_video_url"))
    print("extracted_urls:", res.metadata.get("extracted_urls"))
    print("output:", str(out_path), "exists:", out_path.exists(), "size:", out_path.stat().st_size if out_path.exists() else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
