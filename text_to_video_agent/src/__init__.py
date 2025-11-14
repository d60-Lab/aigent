"""
Text-to-Video AI Agent

A complete AI agent that generates videos from text descriptions using:
- Text-to-Image generation
- Image-to-Video generation
- Text-to-Audio generation
- Video processing and concatenation
"""

from .poe_client import GenerationResult, PoeClient, PoeClientSync
from .text_to_video_agent import (
    PipelineConfig,
    PipelineResult,
    TextToVideoAgent,
    TextToVideoAgentSync,
    VideoScene,
)
from .video_processor import VideoInfo, VideoProcessor

__version__ = "0.1.0"
__all__ = [
    "TextToVideoAgent",
    "TextToVideoAgentSync",
    "PipelineConfig",
    "PipelineResult",
    "VideoScene",
    "PoeClient",
    "PoeClientSync",
    "GenerationResult",
    "VideoProcessor",
    "VideoInfo",
]
