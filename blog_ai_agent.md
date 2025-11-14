# 手把手实战：用 20 分钟做一个文生视频 AI Agent（从零到可用）

这是一篇严格按步骤来的实操教程，带你从 0 到 1 构建并运行一个“文本 → 视频（含配音）”的 AI Agent。你将学会：安装依赖、配置密钥、运行第一个示例、理解核心代码、扩展为多场景视频，以及如何排错和优化。

> 最新实现要点（已内置到命令行）
> - 一键中文故事：仅凭 `--description` 的中文故事，自动规划多场景、自动中文旁白（缺省由场景描述充当旁白）、自动电影化（24fps、fadeblack、镜头连贯）。
> - 中文提示全链路：规划/图生图/图生视频均使用中文提示，输出直链 URL，减少英文与占位。
> - 稳定音频合成：默认 TTS 使用 `hailuo-speech-02`，合并音频时显式映射和必要重编码；音频过短会自动补静音到与视频等长。
> - 占位兜底：若图生视频失败，自动用本地 FFmpeg 对图片生成 Ken Burns 动画片段，避免插入空白。
> - 自由规划模式：`--planner-free` 让模型按故事自由切分场景与时长，生成前会打印规划并询问确认。

## 你将完成什么

- 输入一段文本描述和一段配音文本
- 由 Agent 自动完成：文生图 → 图生视频 → 文生音频 → 合并 →（可选）拼接多场景
- 产出一个可播放的 MP4 视频

示意流程：

```
文本描述 → 生成图片 → 生成视频 → 生成音频 → 合并 →（可选）拼接 → 输出
```

## 一、准备工作（5 分钟）

必备环境：

1) Python 3.8+（建议 3.12）  
2) FFmpeg（命令行工具）  
3) Poe API Key（https://poe.com/api_key）

检查：

```bash
python3 --version
ffmpeg -version
```

## 二、获取与安装（5 分钟）

仓库地址（SSH）：

```
git@github.com:d60-Lab/aigent.git
```

```bash
# 获取源码并进入子模块
git clone git@github.com:d60-Lab/aigent.git
cd aigent/text_to_video_agent

# 安装依赖
python3 -m pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
echo "POE_API_KEY=你的密钥" >> .env
```

目录速览：

```
text_to_video_agent/
├── src/
│   ├── text_to_video_agent.py   # Agent 编排（核心流水线）
│   ├── poe_client.py            # Poe API 封装
│   └── video_processor.py       # FFmpeg 工具
├── examples/
│   ├── simple_example.py        # 单场景示例
│   └── multi_scene_example.py   # 多场景示例
├── main.py                      # 命令行入口
└── output/ temp/               # 输出与临时目录
```

## 三、跑通第一个示例（3 分钟）

方式 A：一键中文故事（推荐）

```bash
python3 main.py -d "塞翁失马焉知非福 的 成语故事" --output se_weng_story
# 可选：交给规划器自由决定场景与时长（生成前会打印规划并确认）
python3 main.py -d "塞翁失马焉知非福 的 成语故事" --planner-free --output se_weng_story
```

方式 B：运行示例脚本

```bash
python3 examples/simple_example.py
```

成功后，在 `output/` 下看到 `sunset_demo_*.mp4`。

小贴士：如果你只想测试流程是否跑通，可先把 `examples/simple_example.py` 里的 `strict_generation=True` 去掉，让占位资源也能通过（便于本地验证）。

## 四、最短可用代码（Python API）

```python
import os
from dotenv import load_dotenv
from pathlib import Path
from src.text_to_video_agent import TextToVideoAgentSync, PipelineConfig

load_dotenv()
agent = TextToVideoAgentSync(os.getenv("POE_API_KEY"), PipelineConfig(output_dir=Path("./output")))

result = agent.create_simple_video(
    description="星空下的雪山，银河清晰可见",
    audio_text="仰望星空，总能感到宁静与渺小",
    output_name="starry"
)

print("OK:" if result.success else "FAIL:", result.output_path)
```

## 五、扩展为多场景（含转场）

方式 A：JSON 配置驱动

```json
{
  "scenes": [
    { "description": "日出时的山峰", "audio_text": "新的一天开始了" },
    { "description": "森林中的溪流", "audio_text": "水声潺潺，生机盎然" }
  ]
}
```

运行：

```bash
python3 main.py --config examples/scenes_example.json --output my_story
```

方式 B：直接用示例脚本

```bash
python3 examples/multi_scene_example.py
```

开启/关闭转场：

```bash
python3 main.py --config examples/scenes_example.json --no-transitions
```

进阶：约 30 秒电影感示例（含中文旁白，保留中间文件便于核对）

```bash
export POE_API_KEY=你的密钥
# 可不设：默认 ITV=Runway-Gen-4-Turbo，TTA=hailuo-speech-02
python3 examples/thirty_seconds_example.py
# 或自定义：
python3 main.py -d "海边旅行短片：开场远景→中景跟拍→近景细节…" \
  --plan --max-scenes 6 --target-duration 30 \
  --cinematic --fps 24 --transition-type fadeblack --transition-duration 0.2 \
  --keep-temp --output travel_30s
```

## 六、核心原理（2 分钟读懂）

Agent 做的事情很朴素：

```python
# 伪代码（src/text_to_video_agent.py）
for scene in scenes:
    image = poe.text_to_image(description)
    video = poe.image_to_video(image, motion)
    audio = poe.text_to_audio(audio_text)
    final = ffmpeg.merge(video, audio)

final_output = ffmpeg.concat(all_scene_finals, transitions=True)
```

代码分层：

- `PoeClient`：把 LLM 的流式响应解析成 URL 或占位资源，必要时下载到本地
- `VideoProcessor`：不同视频片段的处理、合并、转场（FFmpeg）
- `TextToVideoAgent`：编排整个流水线、管理状态与产物

## 七、常见问题与排错

- 提示找不到 POE_API_KEY：在项目根或 `text_to_video_agent/` 下创建 `.env` 并设置 `POE_API_KEY=...`
- FFmpeg 报错：用 `ffmpeg -version` 检查是否安装；或先用较低分辨率测试
- 速度慢：先把分辨率降到 640×360，场景数量减到 1，确认可用再提高参数
- 想看中间结果：加 `--keep-temp`，中间文件会保存在 `temp/`
- 没声音/被截短：合并时已显式映射外部音轨并自动补静音；若仍异常，用 `python3 examples/test_merge_audio_video.py --no-shortest` 指定 `temp/` 中的视频与音频做独立合并验证。
- 占位片段：遇到图生视频失败会自动用图片合成 Ken Burns 片段兜底；也可检查 `temp/video_fb_*.mp4`。

可用的 CLI 选项（常用）：

- `--planner-free`：不限制场景数/时长，交给规划器自由切分；生成前会打印规划并询问确认。
- `--plan/--max-scenes/--target-duration`：常规受控规划（不启用 free 时生效）。
- `--cinematic --fps --transition-type --transition-duration`：电影化参数；默认 24fps、fadeblack。
- `--keep-temp`：保留临时文件便于排错。

## 八、你可以做的改进

- 新的转场/滤镜：在 `video_processor.py` 里加自定义 FFmpeg 滤镜
- 替换模型来源：在 `poe_client.py` 里接入你熟悉的图像/视频/语音服务
- 加字幕/水印：在视频合成阶段插入滤镜或调用额外的工具

## 九、进一步学习

- 架构与原理：`ARCHITECTURE.md`
- 快速操作清单：`QUICK_START.md`
- 代码总览与示例：项目根 `README.md`、子模块 `text_to_video_agent/README.md`

---

到这里，你已经拥有了一个可跑通、可扩展的文生视频 AI Agent。原理并不神秘，关键是把“感知 → 决策 → 执行 → 反馈”的链路用工程方法打磨扎实。动手做几个小任务，你就会更快掌握它。
