"""
Poe API Client (Refactored)

This client uses the official fastapi_poe SDK to interact with Poe bots instead of
calling undocumented HTTP endpoints. It provides higher-level helpers for:

- Text → Image
- Image → Video (simulated since direct media upload isn't exposed via SDK; can be adapted)
- Text → Audio

Features:
- Robust retry with exponential backoff
- Streaming collection & timeout control
- URL extraction (regex + JSON parsing attempts)
- Graceful fallbacks with placeholder asset generation
- Unified GenerationResult dataclass with metadata
- Synchronous wrapper convenience class

Notes:
1. The Poe fastapi_poe SDK currently exposes functions like `get_bot_response` for
   streaming text responses. Media generation must be invoked via prompt engineering.
2. Since direct binary/media APIs aren't public, this client relies on the model returning
   either a URL or JSON describing the asset. If none found, a placeholder file is created.
3. For real production usage, replace the prompt templates with ones matching the
   actual bot/media provider contract or use official media endpoints if/when available.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
from fastapi_poe import ProtocolMessage, get_bot_response

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:  # Pillow is optional but listed in requirements
    Image = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("poe_client")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    success: bool
    data: Optional[bytes] = None  # Raw binary (image/audio/video)
    text: Optional[str] = None  # Final textual response
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _ensure_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return _extract_first_json(text)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_URL_REGEX = re.compile(
    r"(https?://[^\s\"'<>]+)",
    re.IGNORECASE,
)

# data:image/png;base64,... extractor
_DATA_IMAGE_REGEX = re.compile(
    r"data:(image/(?:png|jpeg|jpg|gif|webp));base64,([A-Za-z0-9+/=]+)",
    re.IGNORECASE,
)

_JSON_OBJECT_REGEX = re.compile(r"\{.*?\}", re.DOTALL)


def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    found = _URL_REGEX.findall(text)
    # Deduplicate preserving order
    seen = set()
    result = []
    for u in found:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to extract and parse the first JSON object from arbitrary text."""
    if not text:
        return None
    # Quick direct attempt
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback: find first likely JSON object substring
    for match in _JSON_OBJECT_REGEX.finditer(text):
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _safe_json_get(d: Optional[Dict[str, Any]], key: str) -> Optional[str]:
    if not d:
        return None
    val = d.get(key)
    return val if isinstance(val, str) and val.strip() else None


def _maybe_decode_data_url(text: str) -> Optional[bytes]:
    """If a data:image/...;base64 URL is present, decode and return bytes."""
    if not text:
        return None
    m = _DATA_IMAGE_REGEX.search(text)
    if not m:
        return None
    b64 = m.group(2)
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


def _sniff_content_type(data: Optional[bytes]) -> str:
    """
    Very lightweight content sniffing to classify downloaded bytes.
    Returns one of: "image" | "video" | "audio" | "unknown"
    """
    if not data or len(data) < 12:
        return "unknown"

    head = data[:64]
    # Images
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if head.startswith(b"\xff\xd8\xff"):
        return "image"  # JPEG
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image"
    if head.startswith(b"RIFF") and b"WEBP" in head[:16]:
        return "image"

    # Audio (WAV, MP3, AAC/ADTS, M4A as MP4)
    if head.startswith(b"RIFF") and b"WAVE" in head[:16]:
        return "audio"
    if head.startswith(b"ID3") or head[:2] == b"\xff\xfb":
        return "audio"  # MP3 variants
    if head[:2] == b"\xff\xf1" or head[:2] == b"\xff\xf9":
        return "audio"  # AAC ADTS

    # MP4 / MOV / M4A family typically contains 'ftyp' box near start
    if b"ftyp" in head[:32]:
        # Could be video or audio (m4a). We'll treat as video for video path; audio path adds its own checks.
        return "video"

    # Matroska (MKV)
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "video"

    return "unknown"


def _backoff_sleep(attempt: int, base: float = 0.5, cap: float = 8.0):
    """Exponential backoff with jitter."""
    exp = min(cap, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0, exp / 4)
    time.sleep(exp + jitter)


# ---------------------------------------------------------------------------
# Poe Client
# ---------------------------------------------------------------------------


class PoeClient:
    """
    High-level Poe client using fastapi_poe SDK streaming interface.
    """

    def __init__(
        self,
        api_key: str,
        default_image_bot: str = "nano-banana",
        default_video_bot: str = "nano-banana",
        default_audio_bot: str = "nano-banana",
        max_concurrency: int = 3,
        simulate: bool = False,
    ):
        """
        simulate:
            When True, all bot calls are short‑circuited and placeholder assets are returned
            without performing any network requests. Useful for offline testing.
        """
        self.api_key = api_key
        self.default_image_bot = default_image_bot
        self.default_video_bot = default_video_bot
        self.default_audio_bot = default_audio_bot
        self._sem = asyncio.Semaphore(max_concurrency)
        self.simulate = simulate
        self._compat_base_url = "https://api.poe.com/v1"

    # ---------------------- Core Streaming Call ---------------------------

    async def _call_bot(
        self,
        bot_name: str,
        messages: Sequence[ProtocolMessage],
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> GenerationResult:
        """
        Core streaming call with retries.
        """
        # Simulation short‑circuit: return a synthetic response immediately.
        if self.simulate:
            return GenerationResult(
                success=True,
                text=f"[SIMULATED RESPONSE for {bot_name}]",
                metadata={
                    "bot_name": bot_name,
                    "simulated": True,
                    "chunks_count": 0,
                    "attempts": 0,
                    "timeout_seconds": timeout,
                    "placeholder": True,
                    "placeholder_reason": "simulation_mode",
                },
            )

        # Collects streaming partials and returns final aggregated text.
        # Handles timeout & generic networking errors.
        # (Removed stray triple quotes ending)
        attempt = 0
        last_error: Optional[str] = None

        while attempt < max_retries:
            attempt += 1
            try:
                async with self._sem:
                    logger.info(f"[{bot_name}] Attempt {attempt}/{max_retries}")
                    # Stream partials
                    chunks: List[str] = []

                    candidate_urls: List[str] = []
                    final_event_raw: Optional[str] = None
                    final_attachments: Optional[Any] = None
                    final_attachment_urls: List[str] = []

                    async def _stream():
                        async for partial in get_bot_response(
                            messages=list(messages),
                            bot_name=bot_name,
                            api_key=self.api_key,
                        ):
                            # Some streaming implementations expose is_final / stop_reason, keep defensively.
                            is_final = False
                            if hasattr(partial, "is_final"):
                                try:
                                    is_final = bool(getattr(partial, "is_final"))
                                except Exception:
                                    is_final = False

                            if hasattr(partial, "text"):
                                chunk = getattr(partial, "text")
                            else:
                                chunk = str(partial)

                            if chunk:
                                chunks.append(chunk)
                                # Incremental URL harvesting (helps debugging models that output progress logs first)
                                for m in _URL_REGEX.findall(chunk):
                                    if m not in candidate_urls:
                                        candidate_urls.append(m)

                            # Attempt to harvest URLs from attachments/content if available
                            try:
                                atts = getattr(partial, "attachments", None)
                                if atts and isinstance(atts, (list, tuple)):
                                    for att in atts:
                                        try:
                                            if isinstance(att, dict):
                                                for v in att.values():
                                                    if isinstance(v, str):
                                                        for m in _URL_REGEX.findall(v):
                                                            if m not in candidate_urls:
                                                                candidate_urls.append(m)
                                            else:
                                                s = str(att)
                                                for m in _URL_REGEX.findall(s):
                                                    if m not in candidate_urls:
                                                        candidate_urls.append(m)
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                            if is_final:
                                # Capture the raw final event text separately (may differ from simple join)
                                final_event_raw = chunk
                                try:
                                    final_attachments = getattr(partial, "attachments", None)
                                    if final_attachments:
                                        # try to collect URLs from attachments
                                        if isinstance(final_attachments, (list, tuple)):
                                            for att in final_attachments:
                                                try:
                                                    if isinstance(att, dict):
                                                        url = att.get("url") or att.get("link")
                                                        if isinstance(url, str):
                                                            final_attachment_urls.append(url)
                                                    else:
                                                        # fallback to string parse
                                                        s = str(att)
                                                        for m in _URL_REGEX.findall(s):
                                                            final_attachment_urls.append(m)
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass

                    await asyncio.wait_for(_stream(), timeout=timeout)

                    full_text = "".join(chunks)

                    metadata = {
                        "bot_name": bot_name,
                        "chunks_count": len(chunks),
                        "raw_chunks": chunks,
                        "attempts": attempt,
                        "timeout_seconds": timeout,
                        "stream_detected_urls": candidate_urls,
                        "final_event_raw": final_event_raw,
                        "final_attachments": final_attachments,
                        "final_attachment_urls": final_attachment_urls,
                        "last_chunk": (chunks[-1] if chunks else None),
                    }

                    return GenerationResult(
                        success=True,
                        text=full_text,
                        metadata=metadata,
                    )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s (attempt {attempt})"
                logger.error(last_error)
            except Exception as e:
                last_error = f"Streaming error (attempt {attempt}): {e}"
                logger.error(last_error)

            if attempt < max_retries:
                _backoff_sleep(attempt)

        return GenerationResult(success=False, error=last_error or "Unknown failure")

    # ---------------------- Download Helpers ------------------------------

    async def _download_bytes(
        self,
        url: str,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> Optional[bytes]:
        attempt = 0
        last_error: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=timeout) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            return data
                        else:
                            last_error = f"HTTP {resp.status} on {url}"
                            logger.warning(last_error)
            except asyncio.TimeoutError:
                last_error = f"Download timeout {timeout}s"
                logger.warning(last_error)
            except Exception as e:
                last_error = f"Download error: {e}"
                logger.warning(last_error)
            if attempt < max_retries:
                _backoff_sleep(attempt)
        logger.error(f"Failed to download after {attempt} attempts: {last_error}")
        return None

    # ---------------------- Placeholder Generators -----------------------

    def _placeholder_image(
        self, text: str, width: int = 768, height: int = 512
    ) -> bytes:
        if Image is None:
            # Fallback simple binary
            return f"PLACEHOLDER_IMAGE:{text[:60]}".encode("utf-8")
        img = Image.new("RGB", (width, height), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        label = f"Placeholder\n{text[:40]}"
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.multiline_text((20, 20), label, fill=(220, 220, 220), font=font, spacing=4)
        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _placeholder_audio(self, duration_sec: float = 2.0) -> bytes:
        """
        Generate a silent WAV placeholder. (Small ~176kB)
        """
        import struct
        import wave

        sample_rate = 16000
        n_frames = int(duration_sec * sample_rate)
        buf = []
        # Silence samples
        for _ in range(n_frames):
            buf.append(struct.pack("<h", 0))
        raw = b"".join(buf)
        from io import BytesIO

        out = BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(raw)
        return out.getvalue()

    def _placeholder_video(
        self,
        width: int = 640,
        height: int = 360,
        duration_sec: float = 1.5,
        text: str = "Placeholder",
    ) -> bytes:
        """
        Generate a valid MP4 placeholder using ffmpeg so media players (e.g. QuickTime) can open it.
        Falls back to simple bytes if ffmpeg is unavailable or fails.

        Args:
            width: video width
            height: video height
            duration_sec: length of placeholder clip
            text: overlay text

        Returns:
            MP4 file bytes
        """
        try:
            import shutil

            if not shutil.which("ffmpeg"):
                raise RuntimeError("ffmpeg not found")

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            # Use a solid color background and overlay text
            cmd = [
                "ffmpeg",
                "-y",  # overwrite temp output if exists
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c=0x202020:s={width}x{height}:d={duration_sec}",
                "-vf",
                f"drawtext=text='{text}':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=(h-text_h)/2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp_path),
            ]
            subprocess.run(cmd, check=True)
            data = tmp_path.read_bytes()
            tmp_path.unlink(missing_ok=True)
            return data
        except Exception as e:
            logger.warning(
                f"Failed to generate ffmpeg placeholder video: {e}; using fallback bytes."
            )
            return f"PLACEHOLDER_VIDEO {width}x{height} ERROR={e}".encode("utf-8")

    # ---------------------- Prompt Templates ------------------------------

    def _image_prompt(self, description: str, lang: str = "zh") -> str:
        # 中文提示，强制返回直链
        return (
            "你是一名图像生成助手。\n"
            "请根据描述生成图片，并且在回复中只给出一个可直接下载的图片 URL（http/https）。\n"
            "如果使用 Markdown，也必须把 URL 直接写出来，例如：![alt](https://...)。\n"
            "不要返回占位或引用，必须是最终直链。\n"
            f"描述（请按{lang}语境理解）：{description}\n"
        )

    def _video_prompt(
        self, description: str, motion: str, image_b64: Optional[str]
    ) -> str:
        base = (
            "You are a video generation assistant.\n"
            "Create a short video and respond with a direct downloadable video URL (mp4/mov).\n"
            "If you include Markdown, ensure the URL appears literally in the text.\n"
            f"Visual base description: {description}\n"
            f"Motion instructions: {motion or 'smooth cinematic motion'}\n"
        )
        if image_b64:
            base += f"Reference image (base64 PNG truncated): {image_b64[:120]}...\n"
        return base

    def _audio_prompt(self, text: str, voice: str) -> str:
        return (
            "You are a text-to-speech assistant.\n"
            "Synthesize speech and respond with a direct downloadable audio URL (mp3/wav/m4a).\n"
            "If you include Markdown, ensure the URL is present literally in the text.\n"
            f"Voice style: {voice}\n"
            f"Text to speak: {text}\n"
        )

    def _audio_compat_prompt(self, text: str, voice: str) -> str:
        return (
            "Create a text-to-speech audio clip for the following text.\n"
            "Respond with a direct downloadable audio URL (http/https) in mp3/wav/m4a format.\n"
            "Include the URL literally in the message (plain text or Markdown).\n"
            f"Voice style: {voice}\n"
            f"Text: {text}\n"
        )

    def _build_video_prompt(
        self,
        model: str,
        description: str,
        motion: str,
        image_b64: Optional[str],
        clip_seconds: Optional[int] = None,
        lang: str = "zh",
    ) -> str:
        """Build a Runway Gen-4 friendly prompt focusing on motion and single-scene output."""
        # Core motion-focused structure per Runway Gen-4 guide
        camera_motion = motion or "handheld camera gently tracks the subject"
        dur_line = (
            f"Generate a single-scene video clip (about {clip_seconds} seconds)."
            if clip_seconds
            else "Generate a single-scene video clip (about 5–10 seconds)."
        )
        parts = [
            dur_line,
            "请使用中文风格描述运动（主体/镜头/场景），不要使用否定语气。",
            "不要包含多次剪切或复杂镜头，仅保留一个连续镜头。",
            f"场景描述：{description}",
            f"镜头运动：{camera_motion}",
            "风格：电影质感，真实自然。",
        ]
        if image_b64:
            parts.append(
                f"Reference image (base64 PNG truncated): {image_b64[:120]}..."
            )
        # Response constraint for downstream downloader
        parts.append(
            "请在回复中直接给出可下载的视频 URL（http/https，mp4/mov），可使用 Markdown，但必须直接包含 URL。"
        )
        return "\n".join(parts)

    # ---------------------- Public Generation Methods --------------------

    async def _compat_chat(
        self,
        model: str,
        user_content: str,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> Optional[str]:
        """Call Poe OpenAI-compatible chat.completions to get a single text content.
        Returns the message content string if successful, otherwise None.
        """
        attempt = 0
        last_err: Optional[str] = None
        while attempt < max_retries:
            attempt += 1
            try:
                url = f"{self._compat_base_url}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": user_content}],
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, json=payload, headers=headers, timeout=timeout
                    ) as resp:
                        if resp.status != 200:
                            last_err = f"HTTP {resp.status}"
                            logger.warning(f"compat chat HTTP {resp.status}")
                        else:
                            data = await resp.json()
                            try:
                                content = data["choices"][0]["message"]["content"]
                                if isinstance(content, str) and content.strip():
                                    return content
                            except Exception as e:
                                last_err = f"parse error: {e}"
            except Exception as e:
                last_err = str(e)
                logger.warning(f"compat chat error: {e}")
            if attempt < max_retries:
                _backoff_sleep(attempt)
        logger.warning(f"compat chat failed after {max_retries} attempts: {last_err}")
        return None

    async def plan_scenes(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_scenes: int = 4,
        target_duration: float = 20.0,
        timeout: float = 60.0,
        retries: int = 3,
        free: bool = False,
    ) -> Dict[str, Any]:
        """Ask LLM to plan scenes (Chinese). Returns a dict with keys: scenes (list), transition (optional).

        When free=True, the model decides number of scenes and (optionally) per‑scene durations based on the story,
        without constraints on max scenes or total duration.
        """
        model = model or self.default_audio_bot
        schema_hint = (
            "仅返回 JSON，格式：{\n"
            "  \"scenes\": [  // 场景列表\n"
            "    {\"description\": string, \"motion\": string, \"audio_text\": string?, \"duration\": number?} // duration 可选，单位秒\n"
            "  ],\n"
            "  \"transition\": {\"type\": string, \"duration\": number} (可选)\n"
            "}\n"
        )
        beats = "请覆盖完整叙事节奏：开端/背景 → 事件或转折 → 反转或发展 → 高潮/解决 → 余韵/主题点题。"
        if free:
            instruction = (
                "将中文故事拆解为若干个自然场景（不限定个数和总时长），" \
                "每个场景给出1句中文画面描述 + 1句中文旁白 + 简洁镜头运动，可选提供该场景时长（秒）。\n"
                "镜头要求：简单、连续、可拍摄的移动（推/拉/摇/移/跟拍 之一），风格统一、色调连贯。\n"
                f"用户故事：{prompt}\n"
                f"{beats}\n"
                f"{schema_hint}"
            )
        else:
            instruction = (
                f"把用户的中文故事拆成 {max_scenes} 个以内的场景，每个场景 1 句话述画面，另给一句中文旁白。\n"
                f"镜头要求：简单、连续、可拍摄的移动（如 推/拉/摇/移/跟拍 之一）。\n"
                f"目标总时长约 {target_duration} 秒，各场景尽量均衡分配。\n"
                f"用户故事：{prompt}\n"
                f"{beats}\n"
                f"{schema_hint}"
            )
        # Try compat first
        txt = await self._compat_chat(model, instruction, timeout=timeout, max_retries=retries)
        parsed = _ensure_json(txt)
        if parsed and isinstance(parsed.get("scenes"), list) and parsed["scenes"]:
            return parsed
        # Fallback to streaming
        sys_msg = ProtocolMessage(role="system", content="Return ONLY JSON per the user's schema.")
        user_msg = ProtocolMessage(role="user", content=instruction)
        res = await self._call_bot(bot_name=model, messages=[sys_msg, user_msg], timeout=timeout, max_retries=retries)
        parsed = _ensure_json(res.text)
        if parsed and isinstance(parsed.get("scenes"), list) and parsed["scenes"]:
            return parsed
        # Last resort: single scene
        return {"scenes": [{"description": prompt, "motion": ""}]} 

    async def text_to_image(
        self,
        prompt: str,
        model: str = None,
        output_path: Optional[Path] = None,
        timeout: float = 120.0,
        retries: int = 3,
    ) -> GenerationResult:
        model = model or self.default_image_bot
        if self.simulate:
            # Deterministic placeholder in simulation mode (image)
            bin_data = self._placeholder_image(prompt)
            if output_path:
                output_path.write_bytes(bin_data)
            return GenerationResult(
                success=True,
                data=bin_data,
                text="[SIMULATED IMAGE]",
                metadata={
                    "placeholder": True,
                    "placeholder_reason": "simulation_mode",
                    "bot_name": model,
                    "simulated": True,
                },
            )
        # First attempt: OpenAI-compatible Chat Completions (often returns a direct poecdn URL)
        compat_text = await self._compat_chat(
            model, prompt, timeout=min(timeout, 60.0), max_retries=retries
        )
        if compat_text:
            urls = _extract_urls(compat_text)
            bin_data: Optional[bytes] = None
            if urls:
                # Prefer image-like URL
                image_url = None
                for u in urls:
                    if re.search(r"\.(png|jpg|jpeg|gif|webp)(\?|$)", u, re.IGNORECASE):
                        image_url = u
                        break
                if not image_url:
                    image_url = urls[0]
                bin_data = await self._download_bytes(image_url)
                if bin_data and _sniff_content_type(bin_data) == "image":
                    if output_path:
                        output_path.write_bytes(bin_data)
                    return GenerationResult(
                        success=True,
                        data=bin_data,
                        text=compat_text,
                        metadata={
                            "placeholder": False,
                            "resolved_image_url": image_url,
                            "from": "compat",
                        },
                    )
            # If compat returned but no usable URL, fall through to streaming

        system_msg = ProtocolMessage(role="system", content="你必须严格按用户格式要求作答。")
        user_msg = ProtocolMessage(role="user", content=self._image_prompt(prompt, lang="zh"))

        result = await self._call_bot(
            bot_name=model,
            messages=[system_msg, user_msg],
            timeout=timeout,
            max_retries=retries,
        )

        if not result.success:
            # Placeholder
            placeholder = self._placeholder_image(prompt)
            if output_path:
                output_path.write_bytes(placeholder)
            result.data = placeholder
            result.metadata["placeholder"] = True
            return result

        # Prefer final event chunk for parsing (progress logs spam earlier chunks)
        source_text = (
            result.metadata.get("final_event_raw")
            or result.metadata.get("last_chunk")
            or result.text
            or ""
        )
        full_text = result.text or ""

        # Detect progress-only streaming (no URLs / no JSON), e.g. "Generating image (0s elapsed)..."
        progress_lines = [ln.strip() for ln in source_text.splitlines() if ln.strip()]
        if (
            progress_lines
            and all(
                re.match(r"^Generating image \(\d+s elapsed\)$", ln)
                for ln in progress_lines
            )
            and not any("http" in ln for ln in progress_lines)
        ):
            result.metadata["placeholder"] = True
            result.metadata["placeholder_reason"] = "progress_only"

        json_obj = _extract_first_json(source_text) or _extract_first_json(full_text)
        image_url = _safe_json_get(json_obj, "image_url")
        # Collect URLs from text and metadata attachments
        urls = list({*(_extract_urls(source_text)), *(_extract_urls(full_text))})
        att_urls = result.metadata.get("final_attachment_urls") or []
        if att_urls:
            urls.extend(att_urls)
            # de-dup preserve order
            seen = set()
            urls = [u for u in urls if not (u in seen or seen.add(u))]
        if not image_url and urls:
            image_url = urls[0]

        bin_data: Optional[bytes] = None
        # Try data URL first (if any)
        if not image_url:
            bin_data = _maybe_decode_data_url(source_text)
        if image_url:
            bin_data = await self._download_bytes(image_url)
            if not bin_data:
                logger.warning(
                    "Image URL found but download failed; using placeholder."
                )
                result.metadata["placeholder_reason"] = "download_failed"
            else:
                detected = _sniff_content_type(bin_data)
                if detected != "image":
                    logger.warning(
                        f"Downloaded content does not look like an image (detected={detected}); discarding."
                    )
                    bin_data = None
                    result.metadata["placeholder_reason"] = (
                        f"content_mismatch:{detected}"
                    )

        if not bin_data:
            bin_data = self._placeholder_image(prompt)
            result.metadata["placeholder"] = True
            result.metadata.setdefault("placeholder_reason", "no_url_extracted")

        if output_path:
            output_path.write_bytes(bin_data)

        result.data = bin_data
        result.metadata.update(
            {
                "parsed_json": json_obj,
                "extracted_urls": urls,
                "resolved_image_url": image_url,
            }
        )
        return result

    async def image_to_video(
        self,
        image_path: Path,
        description: str,
        motion_prompt: str = "",
        model: str = None,
        output_path: Optional[Path] = None,
        timeout: float = 180.0,
        retries: int = 3,
        clip_seconds: Optional[int] = None,
    ) -> GenerationResult:
        model = model or self.default_video_bot

        image_b64 = None
        try:
            data = image_path.read_bytes()
            # Limit size for prompt (avoid huge)
            truncated = data[:200_000]  # ~200KB
            image_b64 = base64.b64encode(truncated).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to read image for base64 embedding: {e}")

        if self.simulate:
            # Deterministic placeholder in simulation mode (video)
            bin_data = self._placeholder_video()
            if output_path:
                output_path.write_bytes(bin_data)
            return GenerationResult(
                success=True,
                data=bin_data,
                text="[SIMULATED VIDEO]",
                metadata={
                    "placeholder": True,
                    "placeholder_reason": "simulation_mode",
                    "bot_name": model,
                    "simulated": True,
                },
            )
        # First attempt: OpenAI-compatible Chat Completions for a direct mp4 URL
        compat_prompt = self._build_video_prompt(
            model,
            description or f"Derived from input image {image_path.name}",
            motion_prompt or "smooth cinematic motion",
            image_b64,
            clip_seconds=clip_seconds,
            lang="zh",
        )
        compat_text = await self._compat_chat(
            model, compat_prompt, timeout=min(timeout, 90.0), max_retries=retries
        )
        if compat_text:
            urls = _extract_urls(compat_text)
            video_url = None
            for u in urls:
                if re.search(r"\.(mp4|mov|mkv)(\?|$)", u, re.IGNORECASE):
                    video_url = u
                    break
            if not video_url and urls:
                video_url = urls[0]
            if video_url:
                bin_data = await self._download_bytes(video_url)
                if bin_data and _sniff_content_type(bin_data) in ("video",):
                    if output_path:
                        output_path.write_bytes(bin_data)
                    return GenerationResult(
                        success=True,
                        data=bin_data,
                        text=compat_text,
                        metadata={
                            "placeholder": False,
                            "resolved_video_url": video_url,
                            "from": "compat",
                        },
                    )

        system_msg = ProtocolMessage(
            role="system", content="You format responses exactly as instructed."
        )
        user_msg = ProtocolMessage(
            role="user",
            content=self._build_video_prompt(
                model=model,
                description=description or f"Derived from input image {image_path.name}",
                motion=motion_prompt or "smooth cinematic motion",
                image_b64=image_b64,
                clip_seconds=clip_seconds,
                lang="zh",
            ),
        )

        result = await self._call_bot(
            bot_name=model,
            messages=[system_msg, user_msg],
            timeout=timeout,
            max_retries=retries,
        )

        if not result.success:
            placeholder = self._placeholder_video()
            if output_path:
                output_path.write_bytes(placeholder)
            result.data = placeholder
            result.metadata["placeholder"] = True
            return result

        # Prefer final chunk for parsing
        source_text = (
            result.metadata.get("final_event_raw")
            or result.metadata.get("last_chunk")
            or result.text
            or ""
        )
        full_text = result.text or ""

        # Detect progress-only video generation logs
        progress_lines = [ln.strip() for ln in source_text.splitlines() if ln.strip()]
        if (
            progress_lines
            and all(
                re.match(r"^Generating video \(\d+s elapsed\)$", ln)
                for ln in progress_lines
            )
            and not any("http" in ln for ln in progress_lines)
        ):
            result.metadata["placeholder"] = True
            result.metadata["placeholder_reason"] = "progress_only"

        json_obj = _extract_first_json(source_text) or _extract_first_json(full_text)
        video_url = _safe_json_get(json_obj, "video_url")
        urls = list({*(_extract_urls(source_text)), *(_extract_urls(full_text))})
        att_urls = result.metadata.get("final_attachment_urls") or []
        if att_urls:
            urls.extend(att_urls)
            seen = set()
            urls = [u for u in urls if not (u in seen or seen.add(u))]
        if not video_url and urls:
            # Prefer mp4-like extension
            for u in urls:
                if re.search(r"\.(mp4|mov|mkv)(\?|$)", u, re.IGNORECASE):
                    video_url = u
                    break
            if not video_url and urls:
                video_url = urls[0]

        bin_data: Optional[bytes] = None
        if video_url:
            bin_data = await self._download_bytes(video_url)
            if not bin_data:
                logger.warning(
                    "Video URL found but download failed; using placeholder."
                )
                result.metadata["placeholder_reason"] = "download_failed"
            else:
                detected = _sniff_content_type(bin_data)
                if detected != "video":
                    logger.warning(
                        f"Downloaded content does not look like a video (detected={detected}); discarding."
                    )
                    bin_data = None
                    result.metadata["placeholder_reason"] = (
                        f"content_mismatch:{detected}"
                    )

        if not bin_data:
            bin_data = self._placeholder_video()
            result.metadata["placeholder"] = True
            result.metadata.setdefault("placeholder_reason", "no_url_extracted")

        if output_path:
            output_path.write_bytes(bin_data)

        result.data = bin_data
        result.metadata.update(
            {
                "parsed_json": json_obj,
                "extracted_urls": urls,
                "resolved_video_url": video_url,
            }
        )
        return result

    async def text_to_audio(
        self,
        text: str,
        voice: str = "en-US-Neural",
        model: str = None,
        output_path: Optional[Path] = None,
        timeout: float = 120.0,
        retries: int = 3,
    ) -> GenerationResult:
        model = model or self.default_audio_bot
        # First attempt: OpenAI-compatible chat returning direct audio URL
        is_hailuo = bool(model and ("hailuo" in model.lower()))
        compat_payload = text if is_hailuo else self._audio_compat_prompt(text, voice)
        compat_text = await self._compat_chat(
            model,
            compat_payload,
            timeout=min(timeout, 60.0),
            max_retries=retries,
        )
        if compat_text:
            urls = _extract_urls(compat_text)
            audio_url = None
            for u in urls:
                if re.search(r"\.(mp3|wav|m4a|aac)(\?|$)", u, re.IGNORECASE):
                    audio_url = u
                    break
            if not audio_url and urls:
                audio_url = urls[0]
            if audio_url:
                data = await self._download_bytes(audio_url)
                if data and _sniff_content_type(data) in ("audio", "video"):
                    if output_path:
                        output_path.write_bytes(data)
                    return GenerationResult(
                        success=True,
                        data=data,
                        text=compat_text,
                        metadata={
                            "placeholder": False,
                            "resolved_audio_url": audio_url,
                            "from": "compat",
                            "voice_requested": voice,
                        },
                    )

        # Fallback to streaming path
        if is_hailuo:
            # Send raw text only; this model expects just the content
            messages = [ProtocolMessage(role="user", content=text)]
        else:
            system_msg = ProtocolMessage(
                role="system", content="You format responses exactly as instructed."
            )
            user_msg = ProtocolMessage(role="user", content=self._audio_prompt(text, voice))
            messages = [system_msg, user_msg]

        result = await self._call_bot(
            bot_name=model,
            messages=messages,
            timeout=timeout,
            max_retries=retries,
        )

        if not result.success:
            placeholder = self._placeholder_audio()
            if output_path:
                output_path.write_bytes(placeholder)
            result.data = placeholder
            result.metadata["placeholder"] = True
            return result

        # Prefer final chunk for parsing
        source_text = (
            result.metadata.get("final_event_raw")
            or result.metadata.get("last_chunk")
            or result.text
            or ""
        )

        # Detect progress-only audio generation logs
        progress_lines = [ln.strip() for ln in source_text.splitlines() if ln.strip()]
        if (
            progress_lines
            and all(
                re.match(r"^Generating audio \(\d+s elapsed\)$", ln)
                for ln in progress_lines
            )
            and not any("http" in ln for ln in progress_lines)
        ):
            result.metadata["placeholder"] = True
            result.metadata["placeholder_reason"] = "progress_only"

        json_obj = _extract_first_json(source_text)
        audio_url = _safe_json_get(json_obj, "audio_url")
        urls = _extract_urls(source_text)
        if not audio_url and urls:
            for u in urls:
                if re.search(r"\.(mp3|wav|m4a|aac)(\?|$)", u, re.IGNORECASE):
                    audio_url = u
                    break
            if not audio_url and urls:
                audio_url = urls[0]

        bin_data: Optional[bytes] = None
        if audio_url:
            bin_data = await self._download_bytes(audio_url)
            if not bin_data:
                logger.warning(
                    "Audio URL found but download failed; using placeholder."
                )
                result.metadata["placeholder_reason"] = "download_failed"
            else:
                detected = _sniff_content_type(bin_data)
                # Accept "audio" OR small MP4 container (video) if TTS returned mp4 with audio-only
                if detected not in ("audio", "video"):
                    logger.warning(
                        f"Downloaded content does not look like audio (detected={detected}); discarding."
                    )
                    bin_data = None
                    result.metadata["placeholder_reason"] = (
                        f"content_mismatch:{detected}"
                    )

        if not bin_data:
            bin_data = self._placeholder_audio()
            result.metadata["placeholder"] = True
            result.metadata.setdefault("placeholder_reason", "no_url_extracted")

        if output_path:
            output_path.write_bytes(bin_data)

        result.data = bin_data
        result.metadata.update(
            {
                "parsed_json": json_obj,
                "extracted_urls": urls,
                "resolved_audio_url": audio_url,
                "voice_requested": voice,
            }
        )
        return result


# ---------------------------------------------------------------------------
# Synchronous Wrapper
# ---------------------------------------------------------------------------


class PoeClientSync:
    def __init__(self, api_key: str, **kwargs):
        self._client = PoeClient(api_key, **kwargs)

    def text_to_image(
        self,
        prompt: str,
        model: str = None,
        output_path: Optional[Path] = None,
        timeout: float = 120.0,
        retries: int = 3,
    ) -> GenerationResult:
        return asyncio.run(
            self._client.text_to_image(
                prompt=prompt,
                model=model,
                output_path=output_path,
                timeout=timeout,
                retries=retries,
            )
        )

    def image_to_video(
        self,
        image_path: Path,
        description: str,
        motion_prompt: str = "",
        model: str = None,
        output_path: Optional[Path] = None,
        timeout: float = 180.0,
        retries: int = 3,
    ) -> GenerationResult:
        return asyncio.run(
            self._client.image_to_video(
                image_path=image_path,
                description=description,
                motion_prompt=motion_prompt,
                model=model,
                output_path=output_path,
                timeout=timeout,
                retries=retries,
            )
        )

    def text_to_audio(
        self,
        text: str,
        voice: str = "en-US-Neural",
        model: str = None,
        output_path: Optional[Path] = None,
        timeout: float = 120.0,
        retries: int = 3,
    ) -> GenerationResult:
        return asyncio.run(
            self._client.text_to_audio(
                text=text,
                voice=voice,
                model=model,
                output_path=output_path,
                timeout=timeout,
                retries=retries,
            )
        )


# ---------------------------------------------------------------------------
# Manual quick test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    api_key = os.getenv("POE_API_KEY")
    if not api_key:
        print("POE_API_KEY not set; export it before running.")
        raise SystemExit(1)

    client = PoeClientSync(api_key)

    print("=== IMAGE GEN TEST ===")
    img_res = client.text_to_image(
        "A futuristic cityscape at dusk with neon reflections",
        output_path=Path("test_image.png"),
    )
    print(
        "Image success:",
        img_res.success,
        "placeholder:",
        img_res.metadata.get("placeholder"),
    )
    print("Meta:", {k: v for k, v in img_res.metadata.items() if k != "raw_chunks"})

    print("\n=== AUDIO GEN TEST ===")
    aud_res = client.text_to_audio(
        "The quick brown fox jumps over the lazy dog.",
        voice="en-US-Neural",
        output_path=Path("test_audio.wav"),
    )
    print(
        "Audio success:",
        aud_res.success,
        "placeholder:",
        aud_res.metadata.get("placeholder"),
    )
    print("Meta:", {k: v for k, v in aud_res.metadata.items() if k != "raw_chunks"})

    if Path("test_image.png").exists():
        print("Generated/test placeholder image saved.")
    if Path("test_audio.wav").exists():
        print("Generated/test placeholder audio saved.")
