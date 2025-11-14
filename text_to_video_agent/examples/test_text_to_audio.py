"""
Example: Test Text -> Audio via PoeClientSync with a specific model.
Usage:
  TEXT_TO_AUDIO_MODEL=hailuo-speech-02 python examples/test_text_to_audio.py
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

    model = os.getenv("TEXT_TO_AUDIO_MODEL") or "hailuo-speech-02"
    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_text_to_audio.mp3"

    client = PoeClientSync(api_key)
    voice = os.getenv("VOICE_STYLE") or "zh-CN-Neutral"
    text = os.getenv("TTS_TEXT") or "海面上晚霞温柔而宁静，浪花轻轻拍打着沙滩。"

    print(f"🎙️  Generating audio with model={model}, voice={voice}")
    res = client.text_to_audio(text=text, voice=voice, model=model, output_path=out_path)

    print("success:", res.success)
    print("placeholder:", res.metadata.get("placeholder"), "reason:", res.metadata.get("placeholder_reason"))
    print("resolved_audio_url:", res.metadata.get("resolved_audio_url"))
    print("output:", str(out_path), "exists:", out_path.exists(), "size:", out_path.stat().st_size if out_path.exists() else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
