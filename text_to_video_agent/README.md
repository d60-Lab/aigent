# Text-to-Video AI Agent

一个完整的文生视频 AI Agent，能够将文本描述转换成带音频的完整视频。

## 功能特性

这个 AI Agent 实现了完整的文生视频流水线：

1. **文生图（Text → Image）** - 将文本描述转换为图像
2. **图生视频（Image → Video）** - 将静态图像转换为动态视频
3. **文生音频（Text → Audio）** - 生成旁白或背景音乐
4. **音视频合并** - 将音频添加到视频中
5. **多视频拼接** - 将多个场景拼接成完整视频，支持过渡效果

## 技术架构

```
┌─────────────┐
│ 用户输入文本 │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  AI Agent 决策   │
│ (任务编排与调度)  │
└──────┬───────────┘
       │
       ├─────────────────────────────────┐
       │                                 │
       ▼                                 ▼
┌─────────────┐                  ┌─────────────┐
│ Poe.com API │                  │   FFmpeg    │
│ ─────────── │                  │ ─────────── │
│ • 文生图    │                  │ • 视频合并  │
│ • 图生视频  │                  │ • 音频混合  │
│ • 文生音频  │                  │ • 场景拼接  │
└─────────────┘                  └─────────────┘
       │                                 │
       └─────────────┬───────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │ 最终视频输出 │
              └──────────────┘
```

## 安装

### 前置要求

1. **Python 3.8+**
2. **FFmpeg** - 用于视频处理
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # Windows
   # 从 https://ffmpeg.org/download.html 下载
   ```

3. **Poe.com API Key**
   - 访问 https://poe.com/api_key 获取 API 密钥

### 安装依赖

```bash
cd text_to_video_agent
pip install -r requirements.txt
```

### 配置

1. 复制环境变量模板：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，添加你的 API 密钥：
   ```bash
   POE_API_KEY=your_api_key_here
   ```

## 使用方法

### 快速开始 - 单场景视频

```python
import os
from dotenv import load_dotenv
from src.text_to_video_agent import TextToVideoAgentSync, PipelineConfig

# 加载配置
load_dotenv()
api_key = os.getenv("POE_API_KEY")

# 创建 Agent
agent = TextToVideoAgentSync(api_key)

# 生成视频
result = agent.create_simple_video(
    description="A beautiful sunset over a calm ocean",
    audio_text="As the day ends, the sun paints the sky in gold.",
    output_name="sunset"
)

if result.success:
    print(f"✅ Video created: {result.output_path}")
```

### 多场景视频（带过渡效果）

```python
from src.text_to_video_agent import TextToVideoAgentSync, PipelineConfig
from pathlib import Path

# 配置
config = PipelineConfig(
    output_dir=Path("./output"),
    add_transitions=True,
    transition_duration=1.5
)

agent = TextToVideoAgentSync(api_key, config)

# 定义多个场景
scenes = [
    {
        "description": "A peaceful mountain at dawn",
        "audio_text": "In the mountains, a new day begins.",
        "motion": "slow pan across the mountain range"
    },
    {
        "description": "A flowing river through forest",
        "audio_text": "The river carries stories downstream.",
        "motion": "follow the river's flow"
    },
    {
        "description": "An eagle soaring in the sky",
        "audio_text": "Above it all, freedom takes flight.",
        "motion": "follow the eagle's glide"
    }
]

# 生成视频
result = agent.create_video(scenes, output_name="nature_story")
```

### 运行示例

```bash
# 简单示例
cd examples
python simple_example.py

# 多场景示例
python multi_scene_example.py
```

## 配置选项

`PipelineConfig` 支持以下配置：

```python
config = PipelineConfig(
    # 模型选择
    text_to_image_model="FLUX-pro",          # 文生图模型
    image_to_video_model="FLUX-pro",         # 图生视频模型
    text_to_audio_model="Claude-3.5-Sonnet", # 文生音频模型
    
    # 输出配置
    output_dir=Path("./output"),  # 输出目录
    temp_dir=Path("./temp"),      # 临时文件目录
    
    # 视频参数
    video_width=1920,             # 视频宽度
    video_height=1080,            # 视频高度
    audio_volume=1.0,             # 音频音量 (0.0-1.0+)
    
    # 过渡效果
    add_transitions=True,         # 是否添加场景过渡
    transition_duration=1.0,      # 过渡时长（秒）
    
    # 清理
    clean_temp=True               # 自动清理临时文件
)
```

## API 参考

### TextToVideoAgent

主要的 AI Agent 类，提供完整的视频生成功能。

#### 方法

**`create_simple_video(description, audio_text, output_name)`**
- 创建单场景视频
- 参数：
  - `description`: 视频场景描述
  - `audio_text`: 旁白文本（可选）
  - `output_name`: 输出文件名
- 返回：`PipelineResult`

**`create_video(scenes, output_name)`**
- 创建多场景视频
- 参数：
  - `scenes`: 场景列表，每个场景包含 `description`, `audio_text`, `motion`
  - `output_name`: 输出文件名
- 返回：`PipelineResult`

### PipelineResult

结果对象包含：
- `success`: 是否成功
- `output_path`: 输出文件路径
- `scenes`: 场景列表
- `errors`: 错误信息列表
- `duration`: 处理时长
- `metadata`: 元数据

## 工作原理

### 流水线步骤

```
1. 用户输入
   ↓
2. Agent 分析并规划任务
   ↓
3. 对每个场景：
   a. 调用 Text-to-Image API 生成图片
   b. 调用 Image-to-Video API 生成视频
   c. 调用 Text-to-Audio API 生成音频（如果有）
   d. 使用 FFmpeg 合并音视频
   ↓
4. 如果有多个场景：
   使用 FFmpeg 拼接所有场景（可选添加过渡效果）
   ↓
5. 输出最终视频
```

### Agent 设计模式

这个 AI Agent 采用了 **Pipeline 模式**：

- **模块化**：每个步骤都是独立的函数
- **可组合**：步骤可以灵活组合
- **容错性**：每个步骤都有错误处理
- **可追踪**：详细的日志记录
- **可扩展**：容易添加新的处理步骤

## 项目结构

```
text_to_video_agent/
├── src/
│   ├── __init__.py
│   ├── text_to_video_agent.py  # 主 Agent 类
│   ├── poe_client.py           # Poe API 客户端
│   └── video_processor.py      # FFmpeg 视频处理
├── examples/
│   ├── simple_example.py       # 简单示例
│   └── multi_scene_example.py  # 多场景示例
├── tests/                      # 单元测试
├── output/                     # 输出目录
├── temp/                       # 临时文件
├── requirements.txt
├── .env.example
└── README.md
```

## 开发指南

### 添加新的 AI 模型

编辑 `poe_client.py`，在相应的方法中添加新模型：

```python
async def text_to_image(self, prompt: str, model: str = "new-model"):
    # 实现新模型调用
    pass
```

### 添加新的视频效果

编辑 `video_processor.py`，添加新的 FFmpeg 操作：

```python
def add_custom_effect(self, video_path: Path, output_path: Path):
    args = [
        "-i", str(video_path),
        "-vf", "your_filter_here",
        str(output_path)
    ]
    return self._run_ffmpeg(args, "Custom effect")
```

### 扩展 Agent 功能

在 `text_to_video_agent.py` 中添加新的步骤：

```python
async def _generate_subtitle(self, text: str) -> Optional[Path]:
    # 生成字幕文件
    pass
```

## 常见问题

### Q: 视频生成失败怎么办？

A: 检查以下几点：
1. API 密钥是否正确
2. FFmpeg 是否安装
3. 查看日志输出的具体错误信息
4. 检查网络连接

### Q: 如何提高视频质量？

A: 
1. 使用更详细的描述文本
2. 选择更强大的 AI 模型
3. 调整视频分辨率配置
4. 为运动添加具体的描述

### Q: 支持哪些视频格式？

A: 默认输出 MP4 格式，但可以通过修改 FFmpeg 参数支持其他格式。

### Q: 如何控制成本？

A: 
1. 减少场景数量
2. 复用生成的图片
3. 使用较小的分辨率进行测试
4. 设置合理的超时时间

## 性能优化建议

1. **并行处理**：对于多场景，可以考虑并行生成图片和音频
2. **缓存策略**：缓存已生成的图片，避免重复生成
3. **资源管理**：及时清理临时文件
4. **批量处理**：一次处理多个视频请求

## 错误处理

Agent 提供多层错误处理：

1. **API 级别**：自动重试失败的 API 调用
2. **Pipeline 级别**：单个场景失败不影响其他场景
3. **文件级别**：自动清理损坏的临时文件
4. **日志记录**：详细记录所有错误信息

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 相关资源

- [Poe API 文档](https://developer.poe.com/)
- [FFmpeg 文档](https://ffmpeg.org/documentation.html)
- [相关 Blog 文章](../blog_ai_agent.md)

## 联系方式

如有问题，请提交 Issue 或联系维护者。

---

**注意**：这是一个演示项目，展示了 AI Agent 的核心概念。实际的 Poe API 接口可能需要根据官方文档进行调整。
