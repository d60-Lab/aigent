"""
Video processing utilities using FFmpeg
Handles video merging, audio mixing, and concatenation
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Video file information"""

    path: Path
    duration: float
    width: int
    height: int
    has_audio: bool


class VideoProcessor:
    """FFmpeg-based video processing"""

    def __init__(self):
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Check if FFmpeg is installed"""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            logger.info("FFmpeg is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg: "
                "https://ffmpeg.org/download.html"
            )

    def _run_ffmpeg(
        self, args: List[str], description: str = "FFmpeg operation"
    ) -> bool:
        """Run FFmpeg command"""
        cmd = ["ffmpeg", "-y"] + args  # -y to overwrite output files

        logger.info(f"Running: {description}")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            logger.info(f"✓ {description} completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {description} failed")
            logger.error(f"Error: {e.stderr}")
            return False

    def get_video_info(self, video_path: Path) -> Optional[VideoInfo]:
        """Get video file information using ffprobe"""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,width,height,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1",
                str(video_path),
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            # Parse output
            output = result.stdout
            width = height = 0
            duration = 0.0
            has_audio = False

            for line in output.split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key == "width":
                        width = int(value)
                    elif key == "height":
                        height = int(value)
                    elif key == "duration":
                        try:
                            duration = float(value)
                        except ValueError:
                            pass
                    elif key == "codec_type" and value == "audio":
                        has_audio = True

            return VideoInfo(
                path=video_path,
                duration=duration,
                width=width,
                height=height,
                has_audio=has_audio,
            )
        except Exception as e:
            logger.error(f"Failed to get video info: {str(e)}")
            return None

    def merge_audio_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        audio_volume: float = 1.0,
        shortest: bool = True,
    ) -> bool:
        """
        Merge audio into video

        Args:
            video_path: Input video file
            audio_path: Input audio file
            output_path: Output video file
            audio_volume: Audio volume multiplier (0.0 to 1.0+)

        Returns:
            True if successful
        """
        # Use filter_complex to ensure we always attach the external audio,
        # apply volume on the correct stream, and standardize sample rate/channels.
        args = [
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            f"[1:a:0]volume={audio_volume}[ac]",
            "-map",
            "0:v:0",
            "-map",
            "[ac]",
            "-c:v",
            "copy",  # Copy video codec (no re-encoding)
            "-c:a",
            "aac",  # Audio codec
            "-ar",
            "48000",  # unify sample rate for downstream acrossfade
            "-ac",
            "2",      # stereo
            "-b:a",
            "192k",  # Audio bitrate
            "-movflags",
            "+faststart",
        ]
        if shortest:
            args.append("-shortest")
        args.append(str(output_path))

        # First attempt: keep video stream as-is (copy)
        ok = self._run_ffmpeg(args, f"Merging audio into video: {output_path.name}")
        if not ok:
            return False

        info = self.get_video_info(output_path)
        if info and info.has_audio:
            return True

        # Fallback: re-encode video to ensure container/stream compatibility
        args_fallback = [
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            f"[1:a:0]volume={audio_volume}[ac]",
            "-map",
            "0:v:0",
            "-map",
            "[ac]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
        ]
        if shortest:
            args_fallback.append("-shortest")
        args_fallback.append(str(output_path))

        ok2 = self._run_ffmpeg(args_fallback, f"Merging (fallback re-encode) audio into video: {output_path.name}")
        if not ok2:
            return False
        info2 = self.get_video_info(output_path)
        return bool(info2 and info2.has_audio)

    def get_media_duration(self, media_path: Path) -> Optional[float]:
        """Return media duration in seconds using ffprobe format duration."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except Exception:
            return None

    def generate_silence(self, output_path: Path, duration: float, sample_rate: int = 48000) -> bool:
        """Generate a silent audio track of given duration (mp3)."""
        args = [
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl=mono",
            "-t",
            f"{duration:.2f}",
            "-q:a",
            "9",
            "-acodec",
            "libmp3lame",
            str(output_path),
        ]
        return self._run_ffmpeg(args, f"Generating {duration:.2f}s silent audio")

    def pad_audio_to_duration(self, input_audio: Path, output_audio: Path, duration: float) -> bool:
        """Pad an audio clip with silence to reach target duration (seconds)."""
        if duration <= 0:
            return False
        args = [
            "-i",
            str(input_audio),
            "-af",
            f"apad=pad_dur={duration:.2f}",
            "-t",
            f"{duration:.2f}",
            str(output_audio),
        ]
        return self._run_ffmpeg(args, f"Padding audio to {duration:.2f}s")

    def concatenate_videos(
        self,
        video_paths: List[Path],
        output_path: Path,
        transition: Optional[str] = None,
    ) -> bool:
        """
        Concatenate multiple videos into one

        Args:
            video_paths: List of input video files
            output_path: Output video file
            transition: Optional transition effect (fade, wipe, etc.)

        Returns:
            True if successful
        """
        if len(video_paths) < 2:
            logger.error("Need at least 2 videos to concatenate")
            return False

        # Create temporary file list for FFmpeg concat demuxer
        concat_file = output_path.parent / "concat_list.txt"

        try:
            with open(concat_file, "w") as f:
                for video_path in video_paths:
                    f.write(f"file '{video_path.absolute()}'\n")

            args = [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",  # Copy codecs (fast)
                str(output_path),
            ]

            success = self._run_ffmpeg(args, f"Concatenating {len(video_paths)} videos")

            return success
        finally:
            # Clean up temporary file
            if concat_file.exists():
                concat_file.unlink()

    def add_fade_transition(
        self,
        video_paths: List[Path],
        output_path: Path,
        fade_duration: float = 1.0,
        transition_type: str = "fade",
    ) -> bool:
        """
        Concatenate videos with fade transitions

        Args:
            video_paths: List of input video files
            output_path: Output video file
            fade_duration: Duration of fade effect in seconds

        Returns:
            True if successful
        """
        if len(video_paths) < 2:
            logger.error("Need at least 2 videos for transitions")
            return False

        # Build complex filter for xfade (video) + acrossfade (audio)
        filter_parts: List[str] = []
        input_parts: List[str] = []

        # Inputs
        for vp in video_paths:
            input_parts.extend(["-i", str(vp)])

        # Durations for offsets
        durations: List[float] = []
        for vp in video_paths:
            info = self.get_video_info(vp)
            dur = info.duration if info and info.duration and info.duration > 0 else 5.0
            durations.append(dur)

        # Chain pairwise
        v_prev = "0:v"
        a_prev = "0:a"
        cumulative = durations[0]
        for i in range(1, len(video_paths)):
            v_cur = f"{i}:v"
            a_cur = f"{i}:a"
            v_label = f"vv{i}" if i < len(video_paths) - 1 else "vout"
            a_label = f"aa{i}" if i < len(video_paths) - 1 else "aout"
            offset = max(0.0, cumulative - fade_duration)
            filter_parts.append(
                f"[{v_prev}][{v_cur}]xfade=transition={transition_type}:duration={fade_duration}:offset={offset}[{v_label}]"
            )
            filter_parts.append(
                f"[{a_prev}][{a_cur}]acrossfade=d={fade_duration}:c1=tri:c2=tri[{a_label}]"
            )
            v_prev = v_label
            a_prev = a_label
            cumulative = cumulative + durations[i] - fade_duration

        filter_complex = ";".join(filter_parts)

        args = input_parts + [
            "-filter_complex",
            filter_complex,
            "-map",
            f"[{v_prev}]",
            "-map",
            f"[{a_prev}]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]

        return self._run_ffmpeg(args, f"Adding fade transitions between {len(video_paths)} videos")

    def normalize_video(
        self,
        video_path: Path,
        output_path: Path,
        width: int,
        height: int,
        fps: int = 24,
    ) -> bool:
        """Re-encode video to target resolution and fps with sane defaults (libx264, yuv420p)."""
        scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        vf = f"{scale_filter},fps={fps}"
        args = [
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        return self._run_ffmpeg(args, f"Normalizing video to {width}x{height}@{fps}fps")

    def extract_last_frame(self, video_path: Path, output_image: Path) -> bool:
        """Extract the last frame of a video to an image file (PNG)."""
        info = self.get_video_info(video_path)
        # Try to seek near the end; if info missing, use -sseof -0.1
        if info and info.duration and info.duration > 0.2:
            ts = max(0.0, info.duration - 0.1)
            args = [
                "-ss",
                f"{ts:.2f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(output_image),
            ]
        else:
            args = [
                "-sseof",
                "-0.1",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(output_image),
            ]
        return self._run_ffmpeg(args, f"Extracting last frame from {video_path.name}")

    def create_video_from_image(
        self,
        image_path: Path,
        output_path: Path,
        width: int,
        height: int,
        fps: int = 24,
        duration: float = 5.0,
        motion: str = "zoom_in_slow",
    ) -> bool:
        """Create a simple Ken Burns-style clip from a still image as a fallback.

        motion: zoom_in_slow | zoom_out_slow | pan_left | pan_right
        """
        scale = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        if motion == "zoom_out_slow":
            vf = f"{scale},zoompan=z='if(lte(on,1),1.05,max(1.0,pzoom-0.0008))':d={int(duration*fps)}:s={width}x{height},fps={fps}"
        elif motion == "pan_left":
            vf = f"{scale},zoompan=x='iw*(1-on/{int(duration*fps)})':z=1.0:d={int(duration*fps)}:s={width}x{height},fps={fps}"
        elif motion == "pan_right":
            vf = f"{scale},zoompan=x='iw*(-1+on/{int(duration*fps)})':z=1.0:d={int(duration*fps)}:s={width}x{height},fps={fps}"
        else:  # zoom_in_slow
            vf = f"{scale},zoompan=z='min(1.05,pzoom+0.0008)':d={int(duration*fps)}:s={width}x{height},fps={fps}"

        args = [
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-vf",
            vf,
            "-t",
            f"{duration:.2f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        return self._run_ffmpeg(args, f"Creating fallback video from image: {image_path.name}")

    def resize_video(
        self,
        video_path: Path,
        output_path: Path,
        width: int,
        height: int,
        keep_aspect: bool = True,
    ) -> bool:
        """
        Resize video to specified dimensions

        Args:
            video_path: Input video file
            output_path: Output video file
            width: Target width
            height: Target height
            keep_aspect: Keep aspect ratio (pad if needed)

        Returns:
            True if successful
        """
        if keep_aspect:
            scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        else:
            scale_filter = f"scale={width}:{height}"

        args = [
            "-i",
            str(video_path),
            "-vf",
            scale_filter,
            "-c:a",
            "copy",  # Copy audio
            str(output_path),
        ]

        return self._run_ffmpeg(args, f"Resizing video to {width}x{height}")

    def extract_audio(
        self, video_path: Path, output_path: Path, format: str = "mp3"
    ) -> bool:
        """
        Extract audio from video

        Args:
            video_path: Input video file
            output_path: Output audio file
            format: Audio format (mp3, wav, aac, etc.)

        Returns:
            True if successful
        """
        args = [
            "-i",
            str(video_path),
            "-vn",  # No video
            "-acodec",
            "libmp3lame" if format == "mp3" else "copy",
            str(output_path),
        ]

        return self._run_ffmpeg(args, f"Extracting audio from video")
