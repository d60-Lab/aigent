"""
Unit tests for VideoProcessor
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_processor import VideoProcessor


class TestVideoProcessor(unittest.TestCase):
    """Test cases for VideoProcessor"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.processor = VideoProcessor()

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    def test_ffmpeg_available(self):
        """Test if FFmpeg is available"""
        # This should not raise an exception
        processor = VideoProcessor()
        self.assertIsNotNone(processor)

    def test_create_test_video(self):
        """Create a test video file using FFmpeg"""
        output = self.temp_dir / "test.mp4"

        # Create a 3-second test video (color bars)
        success = self.processor._run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=3:size=640x480:rate=30",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            "Creating test video",
        )

        self.assertTrue(success)
        self.assertTrue(output.exists())
        self.assertGreater(output.stat().st_size, 0)

    def test_get_video_info(self):
        """Test getting video information"""
        # First create a test video
        video_path = self.temp_dir / "info_test.mp4"
        self.processor._run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=2:size=640x480:rate=30",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            "Creating test video for info",
        )

        # Get video info
        info = self.processor.get_video_info(video_path)

        self.assertIsNotNone(info)
        self.assertEqual(info.width, 640)
        self.assertEqual(info.height, 480)
        self.assertGreater(info.duration, 0)

    def test_resize_video(self):
        """Test video resizing"""
        # Create source video
        source = self.temp_dir / "source.mp4"
        self.processor._run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=640x480:rate=30",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            "Creating source video",
        )

        # Resize
        output = self.temp_dir / "resized.mp4"
        success = self.processor.resize_video(
            video_path=source,
            output_path=output,
            width=320,
            height=240,
            keep_aspect=False,
        )

        self.assertTrue(success)
        self.assertTrue(output.exists())

        # Verify dimensions
        info = self.processor.get_video_info(output)
        self.assertEqual(info.width, 320)
        self.assertEqual(info.height, 240)

    def test_concatenate_videos(self):
        """Test video concatenation"""
        # Create two test videos
        video1 = self.temp_dir / "video1.mp4"
        video2 = self.temp_dir / "video2.mp4"

        for video in [video1, video2]:
            self.processor._run_ffmpeg(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=duration=1:size=640x480:rate=30",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ],
                f"Creating {video.name}",
            )

        # Concatenate
        output = self.temp_dir / "concatenated.mp4"
        success = self.processor.concatenate_videos(
            video_paths=[video1, video2], output_path=output
        )

        self.assertTrue(success)
        self.assertTrue(output.exists())

        # Verify duration is approximately sum of inputs
        info = self.processor.get_video_info(output)
        self.assertGreater(info.duration, 1.5)  # Should be ~2 seconds


class TestVideoProcessorEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        self.processor = VideoProcessor()
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_nonexistent_video_info(self):
        """Test getting info for non-existent video"""
        info = self.processor.get_video_info(Path("/nonexistent/video.mp4"))
        self.assertIsNone(info)

    def test_concatenate_single_video(self):
        """Test concatenating a single video (should fail)"""
        video = self.temp_dir / "single.mp4"
        output = self.temp_dir / "output.mp4"

        success = self.processor.concatenate_videos(
            video_paths=[video], output_path=output
        )

        self.assertFalse(success)

    def test_concatenate_empty_list(self):
        """Test concatenating empty list (should fail)"""
        output = self.temp_dir / "output.mp4"

        success = self.processor.concatenate_videos(video_paths=[], output_path=output)

        self.assertFalse(success)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestVideoProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestVideoProcessorEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
