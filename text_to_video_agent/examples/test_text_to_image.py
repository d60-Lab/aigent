"""
Example: Test Text -> Image via PoeClientSync
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poe_client import PoeClientSync  # type: ignore


def main():
    load_dotenv()
    api_key = os.getenv("POE_API_KEY")
    if not api_key:
        print("❌ POE_API_KEY not found. Set it in .env or environment.")
        return 1

    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_text_to_image.png"

    client = PoeClientSync(api_key)
    model = os.getenv("TEXT_TO_IMAGE_MODEL") or "FLUX-pro"
    prompt = "A serene lake at sunset with purple-orange sky"
    print(f"🖼️  Generating image for: {prompt}")
    print(f"🧠 Using bot/model: {model}")

    res = client.text_to_image(prompt=prompt, model=model, output_path=out_path)

    print("success:", res.success)
    print("placeholder:", res.metadata.get("placeholder"), "reason:", res.metadata.get("placeholder_reason"))
    print("chunks:", res.metadata.get("chunks_count"))
    preview = (res.metadata.get("last_chunk") or "")[:200]
    print("last_chunk_preview:", preview.replace("\n", " "))
    print("resolved_image_url:", res.metadata.get("resolved_image_url"))
    print("extracted_urls:", res.metadata.get("extracted_urls"))
    print("output:", str(out_path), "exists:", out_path.exists(), "size:", out_path.stat().st_size if out_path.exists() else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
