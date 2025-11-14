"""
Text-to-Video AI Agent
Orchestrates the complete pipeline from text to final video with audio
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from poe_client import GenerationResult, PoeClient
from video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoScene:
    """Represents a single video scene"""

    description: str
    image_path: Optional[Path] = None
    video_path: Optional[Path] = None
    audio_text: Optional[str] = None
    audio_path: Optional[Path] = None
    final_path: Optional[Path] = None


@dataclass
class PipelineConfig:
    """Configuration for the video generation pipeline

    strict_generation:
        When True, any placeholder media (image/video/audio) returned from PoeClient
        will cause the corresponding scene step to fail immediately instead of
        proceeding with fallback placeholder assets.
    """

    text_to_image_model: str = "FLUX-pro"
    image_to_video_model: str = "FLUX-pro"
    text_to_audio_model: str = "Claude-3.5-Sonnet"
    output_dir: Path = Path("./output")
    temp_dir: Path = Path("./temp")
    video_width: int = 1920
    video_height: int = 1080
    audio_volume: float = 1.0
    add_transitions: bool = True
    transition_duration: float = 1.0
    clean_temp: bool = True
    strict_generation: bool = False
    # Planner
    use_planner: bool = False
    planner_model: str = "Claude-3.5-Sonnet"
    max_scenes: int = 4
    target_duration: float = 20.0  # seconds total
    planner_free: bool = False  # when True, let planner decide scenes/durations freely
    # Cinematic mode
    cinematic_mode: bool = False
    cinematic_fps: int = 24
    cinematic_transition: str = "fadeblack"  # xfade transition name
    # Language & fallbacks
    preferred_language: str = "zh"
    local_image_to_video_fallback: bool = True
    # Narration
    auto_narrate: bool = True  # when True and scene has no audio_text, use description as narration


@dataclass
class PipelineResult:
    """Result from the complete pipeline"""

    success: bool
    output_path: Optional[Path] = None
    scenes: List[VideoScene] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextToVideoAgent:
    """
    AI Agent that converts text descriptions into complete videos with audio

    Pipeline:
    1. Text → Image (for each scene)
    2. Image → Video (for each scene)
    3. Text → Audio (narration/music)
    4. Merge Audio + Video
    5. Concatenate multiple scenes (optional)
    """

    def __init__(self, api_key: str, config: Optional[PipelineConfig] = None):
        self.poe_client = PoeClient(api_key)
        self.video_processor = VideoProcessor()
        self.config = config or PipelineConfig()
        self._transition_override: Optional[float] = None

        # Create directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("TextToVideoAgent initialized")

    def _get_temp_path(self, prefix: str, extension: str) -> Path:
        """Generate temporary file path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.config.temp_dir / f"{prefix}_{timestamp}.{extension}"

    def _get_output_path(self, name: str, extension: str) -> Path:
        """Generate output file path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            c for c in name if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        safe_name = safe_name[:50]  # Limit length
        return self.config.output_dir / f"{safe_name}_{timestamp}.{extension}"

    async def _generate_image(self, description: str) -> Optional[Path]:
        """Generate image from text description"""
        logger.info(f"📸 Generating image: {description[:60]}...")

        image_path = self._get_temp_path("image", "png")

        result = await self.poe_client.text_to_image(
            prompt=description,
            model=self.config.text_to_image_model,
            output_path=image_path,
        )

        # Detailed diagnostics
        logger.debug(
            "[IMAGE] success=%s placeholder=%s placeholder_reason=%s error=%s",
            result.success,
            result.metadata.get("placeholder"),
            result.metadata.get("placeholder_reason"),
            result.error,
        )
        logger.debug("[IMAGE] metadata_keys=%s", list(result.metadata.keys()))
        logger.debug("[IMAGE] text_preview=%r", (result.text or "")[:300])

        # Fail fast if strict_generation enabled and placeholder detected
        if result.metadata.get("placeholder") and self.config.strict_generation:
            logger.error(
                "✗ Image generation returned placeholder; strict_generation=True, aborting scene."
            )
            logger.error(f"Raw text preview: {(result.text or '')[:400]}")
            return None
        # Accept placeholder output when not strict, as long as file exists
        if image_path.exists():
            if result.success:
                logger.info(f"✓ Image generated: {image_path.name}")
            else:
                logger.warning(
                    "Using placeholder image due to generation failure; proceed (strict_generation=False)."
                )
            return image_path
        logger.error(f"✗ Image generation failed: {result.error}")
        logger.error(f"Raw text preview: {(result.text or '')[:400]}")
        return None

    async def _generate_video(
        self,
        image_path: Path,
        description: str,
        motion_prompt: str = "",
        clip_seconds: Optional[int] = None,
    ) -> Optional[Path]:
        """Generate video from image"""
        logger.info(f"🎬 Generating video from {image_path.name}...")

        video_path = self._get_temp_path("video", "mp4")

        result = await self.poe_client.image_to_video(
            image_path=image_path,
            description=description,
            motion_prompt=motion_prompt or "smooth cinematic movement",
            model=self.config.image_to_video_model,
            output_path=video_path,
            clip_seconds=clip_seconds,
        )

        # Detailed diagnostics
        logger.debug(
            "[VIDEO] success=%s placeholder=%s placeholder_reason=%s error=%s",
            result.success,
            result.metadata.get("placeholder"),
            result.metadata.get("placeholder_reason"),
            result.error,
        )
        logger.debug("[VIDEO] metadata_keys=%s", list(result.metadata.keys()))
        logger.debug("[VIDEO] text_preview=%r", (result.text or "")[:300])

        if result.metadata.get("placeholder") and self.config.strict_generation:
            logger.error(
                "✗ Video generation returned placeholder; strict_generation=True."
            )
            # Fallback: synthesize simple motion clip from image
            if self.config.local_image_to_video_fallback:
                fb_path = self._get_temp_path("video_fb", "mp4")
                ok = self.video_processor.create_video_from_image(
                    image_path=image_path,
                    output_path=fb_path,
                    width=self.config.video_width,
                    height=self.config.video_height,
                    fps=self.config.cinematic_fps if self.config.cinematic_mode else 24,
                    duration=float(clip_seconds or 5),
                    motion="zoom_in_slow",
                )
                if ok:
                    logger.info("✓ Used local fallback video from image")
                    return fb_path
            logger.error(f"Raw text preview: {(result.text or '')[:400]}")
            return None
        # Accept placeholder output when not strict, as long as file exists
        if video_path.exists():
            if result.success:
                logger.info(f"✓ Video generated: {video_path.name}")
            else:
                logger.warning(
                    "Using placeholder video due to generation failure; proceed (strict_generation=False)."
                )
            return video_path
        logger.error(f"✗ Video generation failed: {result.error}")
        logger.error(f"Raw text preview: {(result.text or '')[:400]}")
        return None

    async def _generate_audio(self, text: str) -> Optional[Path]:
        """Generate audio from text"""
        logger.info(f"🎵 Generating audio: {text[:60]}...")

        audio_path = self._get_temp_path("audio", "mp3")

        result = await self.poe_client.text_to_audio(
            text=text, model=self.config.text_to_audio_model, output_path=audio_path
        )

        # Detailed diagnostics
        logger.debug(
            "[AUDIO] success=%s placeholder=%s placeholder_reason=%s error=%s",
            result.success,
            result.metadata.get("placeholder"),
            result.metadata.get("placeholder_reason"),
            result.error,
        )
        logger.debug("[AUDIO] metadata_keys=%s", list(result.metadata.keys()))
        logger.debug("[AUDIO] text_preview=%r", (result.text or "")[:300])

        if result.metadata.get("placeholder") and self.config.strict_generation:
            logger.error(
                "✗ Audio generation returned placeholder; strict_generation=True, aborting scene."
            )
            logger.error(f"Raw text preview: {(result.text or '')[:400]}")
            return None
        # Accept placeholder output when not strict, as long as file exists
        if audio_path.exists():
            if result.success:
                logger.info(f"✓ Audio generated: {audio_path.name}")
            else:
                logger.warning(
                    "Using placeholder audio due to generation failure; proceed (strict_generation=False)."
                )
            return audio_path
        logger.error(f"✗ Audio generation failed: {result.error}")
        logger.error(f"Raw text preview: {(result.text or '')[:400]}")
        return None

    async def create_scene(
        self,
        description: str,
        audio_text: Optional[str] = None,
        motion_prompt: Optional[str] = None,
        ref_image: Optional[Path] = None,
        clip_seconds: Optional[int] = None,
    ) -> VideoScene:
        """
        Create a complete video scene

        Args:
            description: Visual description for image/video generation
            audio_text: Optional narration text
            motion_prompt: Optional motion description for video

        Returns:
            VideoScene with all generated assets
        """
        # Auto narrate if enabled and no audio text provided
        if self.config.auto_narrate and not audio_text:
            audio_text = description

        scene = VideoScene(description=description, audio_text=audio_text)

        # Step 1: Generate or reuse reference image
        if self.config.cinematic_mode and ref_image and ref_image.exists():
            scene.image_path = ref_image
            logger.info("📸 Using reference frame from previous scene for continuity")
        else:
            scene.image_path = await self._generate_image(description)
        if not scene.image_path:
            return scene

        # Step 2: Generate video from image
        scene.video_path = await self._generate_video(
            scene.image_path,
            description,
            motion_prompt or "",
            clip_seconds=clip_seconds,
        )
        if not scene.video_path:
            return scene

        # Optional: normalize to cinematic fps/resolution for consistency
        if self.config.cinematic_mode:
            normalized = self._get_temp_path("video_norm", "mp4")
            ok = self.video_processor.normalize_video(
                video_path=scene.video_path,
                output_path=normalized,
                width=self.config.video_width,
                height=self.config.video_height,
                fps=self.config.cinematic_fps,
            )
            if ok:
                scene.video_path = normalized

        # Step 3: Generate audio if text provided
        if audio_text:
            scene.audio_path = await self._generate_audio(audio_text)

            # Step 4: Merge audio with video
            if scene.video_path:
                final_path = self._get_temp_path("scene_final", "mp4")
                # Ensure audio length not truncating the video; if missing/short, pad to video duration
                shortest = True
                vdur = self.video_processor.get_media_duration(scene.video_path) or 0.0
                adur = 0.0
                if scene.audio_path:
                    adur = self.video_processor.get_media_duration(scene.audio_path) or 0.0
                if vdur > 0 and (not scene.audio_path or adur == 0.0):
                    try:
                        silent = self._get_temp_path("audio_silence", "mp3")
                        if self.video_processor.generate_silence(silent, max(vdur - 0.05, 0.5)):
                            scene.audio_path = silent
                            shortest = False
                    except Exception:
                        shortest = False
                elif vdur > 0 and adur < (0.95 * vdur):
                    try:
                        padded = self._get_temp_path("audio_padded", "mp3")
                        if self.video_processor.pad_audio_to_duration(scene.audio_path, padded, vdur):
                            scene.audio_path = padded
                            shortest = False
                    except Exception:
                        shortest = False

                if scene.audio_path:
                    success = self.video_processor.merge_audio_video(
                        video_path=scene.video_path,
                        audio_path=scene.audio_path,
                        output_path=final_path,
                        audio_volume=self.config.audio_volume,
                        shortest=shortest,
                    )
                    if success:
                        scene.final_path = final_path
                    else:
                        scene.final_path = scene.video_path
                else:
                    scene.final_path = scene.video_path
        else:
            scene.final_path = scene.video_path

        return scene

    async def create_video(
        self, scenes: List[Dict[str, str]], output_name: str = "generated_video"
    ) -> PipelineResult:
        """
        Create complete video from multiple scenes

        Args:
            scenes: List of scene dictionaries with 'description', 'audio_text', 'motion'
            output_name: Base name for output file

        Returns:
            PipelineResult with final video path and metadata

        Example:
            scenes = [
                {
                    "description": "A serene mountain landscape at sunrise",
                    "audio_text": "Welcome to the mountains",
                    "motion": "slow pan across the horizon"
                },
                {
                    "description": "A flowing river through the valley",
                    "audio_text": "Nature's beauty surrounds us",
                    "motion": "follow the river downstream"
                }
            ]
        """
        start_time = datetime.now()
        result = PipelineResult(success=False)

        try:
            logger.info(
                f"🚀 Starting video generation pipeline with {len(scenes)} scenes"
            )

            # Decide per-scene clip duration hint for Runway (5s or 10s)
            clip_hint: Optional[int] = None
            if self.config.target_duration and len(scenes) > 0:
                per = self.config.target_duration / len(scenes)
                clip_hint = 10 if per >= 8 else 5

            # Generate all scenes
            scene_objects = []
            prev_ref: Optional[Path] = None
            for i, scene_data in enumerate(scenes, 1):
                logger.info(f"📍 Processing scene {i}/{len(scenes)}")

                scene = await self.create_scene(
                    description=scene_data.get("description", ""),
                    audio_text=scene_data.get("audio_text"),
                    motion_prompt=scene_data.get("motion"),
                    ref_image=prev_ref,
                    clip_seconds=clip_hint,
                )

                scene_objects.append(scene)
                result.scenes.append(scene)

                if not scene.final_path:
                    error_msg = f"Failed to create scene {i}"
                    result.errors.append(error_msg)
                    logger.error(f"✗ {error_msg}")

                # Update reference frame for next scene (continuity)
                if self.config.cinematic_mode and scene.video_path:
                    try:
                        frame_path = self._get_temp_path("ref_frame", "png")
                        if self.video_processor.extract_last_frame(
                            scene.video_path, frame_path
                        ):
                            prev_ref = frame_path
                    except Exception as _:
                        pass

            # Collect valid scene videos
            valid_videos = [s.final_path for s in scene_objects if s.final_path]

            if not valid_videos:
                result.errors.append("No valid videos generated")
                logger.error("✗ Pipeline failed: No valid videos")
                return result

            # Concatenate if multiple scenes
            if len(valid_videos) > 1:
                logger.info(f"🔗 Concatenating {len(valid_videos)} videos")

                output_path = self._get_output_path(output_name, "mp4")

                fade_dur = (
                    self._transition_override
                    if self._transition_override is not None
                    else self.config.transition_duration
                )
                if self.config.add_transitions:
                    success = self.video_processor.add_fade_transition(
                        video_paths=valid_videos,
                        output_path=output_path,
                        fade_duration=fade_dur,
                        transition_type=(
                            self.config.cinematic_transition
                            if self.config.cinematic_mode
                            else "fade"
                        ),
                    )
                else:
                    success = self.video_processor.concatenate_videos(
                        video_paths=valid_videos, output_path=output_path
                    )

                if success:
                    result.output_path = output_path
                    result.success = True
                else:
                    result.errors.append("Failed to concatenate videos")
            else:
                # Single scene - just copy to output
                output_path = self._get_output_path(output_name, "mp4")
                valid_videos[0].rename(output_path)
                result.output_path = output_path
                result.success = True

            # Calculate duration
            end_time = datetime.now()
            result.duration = (end_time - start_time).total_seconds()

            # Add metadata
            result.metadata = {
                "scenes_count": len(scenes),
                "successful_scenes": len(valid_videos),
                "failed_scenes": len(scenes) - len(valid_videos),
                "config": {
                    "text_to_image_model": self.config.text_to_image_model,
                    "image_to_video_model": self.config.image_to_video_model,
                    "text_to_audio_model": self.config.text_to_audio_model,
                },
            }

            if result.success:
                logger.info(
                    f"✅ Pipeline completed successfully in {result.duration:.1f}s"
                )
                logger.info(f"📁 Output: {result.output_path}")
            else:
                logger.error(f"❌ Pipeline completed with errors")

            return result

        except Exception as e:
            logger.error(f"❌ Pipeline error: {str(e)}")
            result.errors.append(str(e))
            return result
        finally:
            # Clean up temporary files
            if self.config.clean_temp:
                self._cleanup_temp_files()
            # reset any overrides
            self._transition_override = None

    async def plan_scenes(self, description: str) -> List[Dict[str, str]]:
        """Use LLM planner to split a description into coherent scenes.

        Returns a list of scene dicts with keys: description, motion, audio_text (optional), duration (optional)
        """
        try:
            plan = await self.poe_client.plan_scenes(
                prompt=description,
                model=self.config.planner_model,
                max_scenes=self.config.max_scenes,
                target_duration=self.config.target_duration,
                free=self.config.planner_free,
            )
            scenes = plan.get("scenes") if isinstance(plan, dict) else None
            # capture transition override if provided
            trans = plan.get("transition") if isinstance(plan, dict) else None
            if isinstance(trans, dict):
                dur = trans.get("duration")
                try:
                    if dur is not None:
                        self._transition_override = float(dur)
                except Exception:
                    pass
            if isinstance(scenes, list) and scenes:
                # normalize keys
                norm = []
                for s in scenes:
                    if not isinstance(s, dict):
                        continue
                    norm.append(
                        {
                            "description": s.get("description") or description,
                            "motion": s.get("motion") or "",
                            "audio_text": s.get("audio_text"),
                        }
                    )
                return norm
        except Exception as e:
            logger.warning(f"Planner failed, fallback to single scene: {e}")
        # Fallback single-scene
        return [{"description": description}]

    async def create_from_description(
        self,
        description: str,
        audio_text: Optional[str] = None,
        output_name: str = "generated_video",
    ) -> PipelineResult:
        """Create video from a high-level description using planner if enabled."""
        if self.config.use_planner:
            planned = await self.plan_scenes(description)
            # If user passed a single audio_text, apply it to first scene only
            if audio_text and planned:
                planned[0]["audio_text"] = audio_text
            return await self.create_video(planned, output_name)
        else:
            return await self.create_video(
                scenes=[{"description": description, "audio_text": audio_text}],
                output_name=output_name,
            )

    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for file in self.config.temp_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            logger.info("🧹 Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to clean temp files: {str(e)}")

    async def create_simple_video(
        self,
        description: str,
        audio_text: Optional[str] = None,
        output_name: str = "video",
    ) -> PipelineResult:
        """
        Create a simple single-scene video

        Args:
            description: Visual description
            audio_text: Optional narration
            output_name: Output file name

        Returns:
            PipelineResult
        """
        scene_data = {"description": description, "audio_text": audio_text}
        return await self.create_video([scene_data], output_name)


# Synchronous wrapper
class TextToVideoAgentSync:
    """Synchronous wrapper for TextToVideoAgent"""

    def __init__(self, api_key: str, config: Optional[PipelineConfig] = None):
        self.agent = TextToVideoAgent(api_key, config)

    def create_video(
        self, scenes: List[Dict[str, str]], output_name: str = "generated_video"
    ) -> PipelineResult:
        return asyncio.run(self.agent.create_video(scenes, output_name))

    def create_simple_video(
        self,
        description: str,
        audio_text: Optional[str] = None,
        output_name: str = "video",
    ) -> PipelineResult:
        return asyncio.run(
            self.agent.create_simple_video(description, audio_text, output_name)
        )

    def create_from_description(
        self,
        description: str,
        audio_text: Optional[str] = None,
        output_name: str = "generated_video",
    ) -> PipelineResult:
        return asyncio.run(
            self.agent.create_from_description(description, audio_text, output_name)
        )

    def plan_scenes_sync(self, description: str) -> List[Dict[str, str]]:
        return asyncio.run(self.agent.plan_scenes(description))
