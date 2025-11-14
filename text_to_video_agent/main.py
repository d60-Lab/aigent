#!/usr/bin/env python3
"""
Text-to-Video AI Agent - Main Entry Point
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from text_to_video_agent import PipelineConfig, TextToVideoAgentSync


def print_banner():
    """Print application banner"""
    print()
    print("=" * 70)
    print("   TEXT-TO-VIDEO AI AGENT")
    print("   将文本描述转换为完整视频")
    print("=" * 70)
    print()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Text-to-Video AI Agent - 文生视频 AI 助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成单场景视频
  python main.py --description "美丽的日落" --audio "一天结束了"

  # 使用配置文件生成多场景视频
  python main.py --config scenes.json

  # 指定输出文件名
  python main.py --description "山景" --output my_video
        """,
    )

    parser.add_argument("--description", "-d", help="视频场景描述（单场景模式）")

    parser.add_argument("--audio", "-a", help="音频旁白文本（可选）")

    parser.add_argument("--motion", "-m", help="运动描述（可选）")

    parser.add_argument(
        "--output",
        "-o",
        default="generated_video",
        help="输出文件名（默认: generated_video）",
    )

    parser.add_argument("--config", "-c", help="场景配置文件（JSON格式，多场景模式）")

    parser.add_argument(
        "--no-transitions", action="store_true", help="禁用场景过渡效果"
    )

    parser.add_argument(
        "--transition-duration",
        type=float,
        default=1.5,
        help="过渡效果时长（秒，默认: 1.5）",
    )
    parser.add_argument(
        "--cinematic",
        action="store_true",
        help="启用电影风格（24fps、fadeblack 转场、参考帧连贯）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="目标帧率（cinematic 模式下生效，默认: 24）",
    )
    parser.add_argument(
        "--transition-type",
        default="fadeblack",
        help="xfade 转场类型（cinematic 模式默认: fadeblack）",
    )

    parser.add_argument(
        "--width", type=int, default=1920, help="视频宽度（默认: 1920）"
    )

    parser.add_argument(
        "--height", type=int, default=1080, help="视频高度（默认: 1080）"
    )

    parser.add_argument(
        "--volume", type=float, default=0.8, help="音频音量（0.0-1.0，默认: 0.8）"
    )

    parser.add_argument(
        "--keep-temp", action="store_true", help="保留临时文件（调试用）"
    )

    parser.add_argument(
        "--plan",
        action="store_true",
        help="启用场景规划：将单段描述拆成多场景并生成连贯视频",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=4, help="规划的最大场景数（默认: 4）"
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=20.0,
        help="规划的目标总时长（秒，默认: 20）",
    )
    parser.add_argument(
        "--planner-free",
        action="store_true",
        help="不限制场景数与总时长，交由规划器自由切分（执行前会交互确认）",
    )

    return parser.parse_args()


def load_scenes_from_file(config_path: str):
    """Load scenes from JSON config file"""
    import json

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "scenes" in data:
            return data["scenes"]
        else:
            print(f"❌ 错误: 配置文件格式不正确")
            return None
    except Exception as e:
        print(f"❌ 读取配置文件失败: {str(e)}")
        return None


def main():
    """Main entry point"""
    print_banner()

    # Load environment variables
    load_dotenv()

    # Check API key
    api_key = os.getenv("POE_API_KEY")
    if not api_key:
        print("❌ 错误: 未找到 POE_API_KEY")
        print()
        print("请设置环境变量:")
        print("  export POE_API_KEY=your_key_here")
        print()
        print("或在 .env 文件中配置:")
        print("  POE_API_KEY=your_key_here")
        print()
        return 1

    # Parse arguments
    args = parse_arguments()

    # Validate arguments
    if not args.description and not args.config:
        print("❌ 错误: 请提供 --description 或 --config 参数")
        print()
        print("使用 --help 查看帮助信息")
        return 1

    # Create pipeline config
    # Models can be overridden via environment variables
    # Prefer strong defaults for one-command storytelling
    tti_model = os.getenv("TEXT_TO_IMAGE_MODEL") or "FLUX-pro"
    itv_model = os.getenv("IMAGE_TO_VIDEO_MODEL") or "Runway-Gen-4-Turbo"
    tta_model = os.getenv("TEXT_TO_AUDIO_MODEL") or "hailuo-speech-02"

    # Enable simple mode (planner + cinematic) by default when only description is provided
    simple_mode = bool(args.description and not args.config)

    config = PipelineConfig(
        output_dir=Path("./output"),
        temp_dir=Path("./temp"),
        video_width=args.width,
        video_height=args.height,
        audio_volume=args.volume,
        add_transitions=not args.no_transitions,
        transition_duration=args.transition_duration,
        clean_temp=not args.keep_temp,
        text_to_image_model=tti_model,
        image_to_video_model=itv_model,
        text_to_audio_model=tta_model,
        use_planner=bool(args.plan) or os.getenv("USE_PLANNER") == "1" or simple_mode,
        max_scenes=args.max_scenes if not simple_mode else max(3, min(6, args.max_scenes)),
        target_duration=args.target_duration if not simple_mode else max(20.0, args.target_duration),
        cinematic_mode=bool(args.cinematic) or os.getenv("CINEMATIC") == "1" or simple_mode,
        cinematic_fps=args.fps,
        cinematic_transition=args.transition_type,
        auto_narrate=True,
        planner_free=bool(args.planner_free) or os.getenv("PLANNER_FREE") == "1",
    )

    # Create agent
    print("🤖 初始化 AI Agent...")
    agent = TextToVideoAgentSync(api_key, config)
    print("✓ Agent 就绪")
    print(f"🧠 模型: TTI={config.text_to_image_model}, ITV={config.image_to_video_model}, TTA={config.text_to_audio_model}")
    if config.use_planner:
        print(f"🗺️  规划: max_scenes={config.max_scenes}, target≈{config.target_duration}s, 语言=中文, 自动旁白={config.auto_narrate}")
    print()

    # Generate video
    try:
        if args.config:
            # Multi-scene mode
            print(f"📖 从配置文件加载场景: {args.config}")
            scenes = load_scenes_from_file(args.config)

            if not scenes:
                return 1

            print(f"✓ 加载了 {len(scenes)} 个场景")
            print()

            result = agent.create_video(scenes=scenes, output_name=args.output)
        else:
            # Single scene mode
            print("🎬 单场景模式")
            print(f"描述: {args.description}")
            if args.audio:
                print(f"音频: {args.audio}")
            if args.motion:
                print(f"运动: {args.motion}")
            print()

            if config.use_planner:
                print("🧭 启用场景规划 → 拆分为多场景")
                # 如果是自由规划模式，先打印规划并确认
                if config.planner_free:
                    planned = agent.plan_scenes_sync(args.description)
                    print("\n📋 规划预览：")
                    for i, s in enumerate(planned, 1):
                        print(f"  场景{i}: {s.get('description','')}")
                        if s.get("motion"):
                            print(f"    镜头: {s['motion']}")
                        if s.get("audio_text"):
                            print(f"    旁白: {s['audio_text']}")
                    ans = input("\n是否按以上规划生成视频？(y/N): ").strip().lower()
                    if ans not in ("y", "yes"):  # 用户取消
                        print("已取消。")
                        return 0
                    result = agent.create_video(planned, output_name=args.output)
                else:
                    result = agent.create_from_description(
                        description=args.description,
                        audio_text=args.audio,
                        output_name=args.output,
                    )
            else:
                scene_data = {
                    "description": args.description,
                    "audio_text": args.audio,
                    "motion": args.motion,
                }
                result = agent.create_video(
                    scenes=[scene_data], output_name=args.output
                )

        # Display results
        print()
        print("=" * 70)

        if result.success:
            print("✅ 视频生成成功！")
            print("=" * 70)
            print()
            print(f"📁 输出文件: {result.output_path}")
            print(f"⏱️  处理时间: {result.duration:.1f} 秒")
            print(
                f"🎬 场景数: {result.metadata.get('successful_scenes', 0)}/{result.metadata.get('scenes_count', 0)}"
            )

            if result.metadata.get("failed_scenes", 0) > 0:
                print(f"⚠️  失败场景: {result.metadata['failed_scenes']}")

            print()
            print("场景详情:")
            for i, scene in enumerate(result.scenes, 1):
                status = "✓" if scene.final_path else "✗"
                desc = (
                    scene.description[:50] + "..."
                    if len(scene.description) > 50
                    else scene.description
                )
                print(f"  {status} 场景 {i}: {desc}")

            return 0
        else:
            print("❌ 视频生成失败")
            print("=" * 70)
            print()
            print("错误信息:")
            for error in result.errors:
                print(f"  • {error}")

            print()
            print("部分结果:")
            for i, scene in enumerate(result.scenes, 1):
                status = "✓" if scene.final_path else "✗"
                desc = (
                    scene.description[:50] + "..."
                    if len(scene.description) > 50
                    else scene.description
                )
                print(f"  {status} 场景 {i}: {desc}")

            return 1

    except KeyboardInterrupt:
        print()
        print("⚠️  用户中断")
        return 130
    except Exception as e:
        print()
        print(f"❌ 发生错误: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
