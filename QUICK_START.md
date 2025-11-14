# 快速上手指南

## 5分钟快速体验

### 前置准备

1. **Python 3.8+**
   ```bash
   python --version  # 确认版本
   ```

2. **FFmpeg**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # 验证安装
   ffmpeg -version
   ```

3. **Poe API Key**
   - 访问 https://poe.com/api_key
   - 登录后获取 API 密钥

### 步骤 1：克隆并配置

```bash
# 进入项目目录
cd text_to_video_agent

# 安装依赖
pip install -r requirements.txt

# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，添加你的 API Key
# POE_API_KEY=你的密钥
```

### 步骤 2：运行第一个示例

```bash
# 方式 1：运行简单示例脚本
python examples/simple_example.py

# 方式 2：使用命令行工具
python main.py \
  --description "一座雪山在蓝天下，阳光照耀" \
  --audio "这是大自然的杰作" \
  --output my_first_video
```

### 步骤 3：查看结果

生成的视频将保存在 `output/` 目录下。

## 常用命令

### 生成单场景视频

```bash
python main.py \
  --description "描述你想要的画面" \
  --audio "配音文本（可选）" \
  --output 输出文件名
```

### 生成多场景视频

```bash
# 使用配置文件
python main.py --config examples/scenes_example.json --output story_video
```

### 自定义参数

```bash
python main.py \
  --description "海边日落" \
  --audio "一天即将结束" \
  --width 1280 \
  --height 720 \
  --volume 0.8 \
  --no-transitions \
  --output sunset_720p
```

## Python 代码示例

### 最简单的用法

```python
import os
from dotenv import load_dotenv
from src.text_to_video_agent import TextToVideoAgentSync

# 加载配置
load_dotenv()
agent = TextToVideoAgentSync(os.getenv("POE_API_KEY"))

# 生成视频
result = agent.create_simple_video(
    description="美丽的星空，银河清晰可见",
    audio_text="仰望星空，感受宇宙的浩瀚",
    output_name="starry_night"
)

print(f"视频位置: {result.output_path}")
```

### 多场景视频

```python
from src.text_to_video_agent import TextToVideoAgentSync

agent = TextToVideoAgentSync(api_key)

scenes = [
    {
        "description": "日出时的山峰",
        "audio_text": "新的一天开始了",
        "motion": "从左到右缓慢平移"
    },
    {
        "description": "山间的瀑布",
        "audio_text": "水声潺潺",
        "motion": "跟随水流向下"
    }
]

result = agent.create_video(scenes, output_name="mountain_journey")
```

### 自定义配置

```python
from pathlib import Path
from src.text_to_video_agent import TextToVideoAgentSync, PipelineConfig

config = PipelineConfig(
    output_dir=Path("./my_videos"),
    video_width=1280,
    video_height=720,
    add_transitions=True,
    transition_duration=2.0,
    audio_volume=0.9
)

agent = TextToVideoAgentSync(api_key, config)
result = agent.create_simple_video(...)
```

## 场景配置文件格式

创建 `my_scenes.json`：

```json
{
  "scenes": [
    {
      "description": "场景的视觉描述",
      "audio_text": "配音文本（可选）",
      "motion": "运动描述（可选）"
    },
    {
      "description": "第二个场景...",
      "audio_text": "...",
      "motion": "..."
    }
  ]
}
```

然后运行：
```bash
python main.py --config my_scenes.json --output my_story
```

## 调试技巧

### 保留临时文件

```bash
python main.py --description "测试" --keep-temp
# 临时文件会保存在 temp/ 目录，可以检查中间结果
```

### 查看详细日志

代码中已经包含详细的日志输出，运行时会显示每个步骤的进度。

### 测试单个功能

```python
# 只测试文生图
from src.poe_client import PoeClientSync
from pathlib import Path

client = PoeClientSync(api_key)
result = client.text_to_image(
    prompt="一只可爱的猫",
    output_path=Path("test.png")
)
```

## 常见问题排查

### 问题：API 调用失败

```
检查清单：
□ API Key 是否正确设置
□ 网络连接是否正常
□ Poe API 配额是否用完
□ 描述文本是否符合内容政策
```

### 问题：FFmpeg 错误

```
检查清单：
□ FFmpeg 是否正确安装
□ 临时文件是否存在
□ 磁盘空间是否充足
□ 视频格式是否支持
```

### 问题：生成速度慢

```
原因：
- AI 模型生成需要时间（特别是图生视频）
- 网络延迟
- 视频处理需要计算资源

优化建议：
- 先用低分辨率测试
- 减少场景数量
- 使用更快的 AI 模型
```

## 进阶用法

### 批量生成

```python
descriptions = [
    "春天的樱花",
    "夏日的海滩",
    "秋天的枫叶",
    "冬日的雪景"
]

for desc in descriptions:
    result = agent.create_simple_video(
        description=desc,
        output_name=desc.replace(" ", "_")
    )
```

### 异步并发

```python
import asyncio
from src.text_to_video_agent import TextToVideoAgent

async def generate_multiple():
    agent = TextToVideoAgent(api_key)
    
    tasks = [
        agent.create_simple_video(desc, output_name=f"video_{i}")
        for i, desc in enumerate(descriptions)
    ]
    
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(generate_multiple())
```

### 自定义视频效果

修改 `src/video_processor.py`，添加新的 FFmpeg 滤镜：

```python
def add_blur_effect(self, video_path: Path, output_path: Path):
    args = [
        "-i", str(video_path),
        "-vf", "boxblur=5:1",
        str(output_path)
    ]
    return self._run_ffmpeg(args, "Adding blur effect")
```

## 性能建议

1. **测试时使用低分辨率**
   ```python
   config = PipelineConfig(
       video_width=640,
       video_height=360
   )
   ```

2. **复用生成的图片**
   - 保存 temp 目录的图片
   - 直接调用 `image_to_video` 方法

3. **并行处理多个场景**
   - 使用异步版本的 Agent
   - 利用 `asyncio.gather` 并发执行

## 下一步

- 📖 阅读完整文档：[README.md](./README.md)
- 🎨 查看 Blog 文章：[blog_ai_agent.md](../blog_ai_agent.md)
- 🔧 查看源代码：[src/](./src/)
- 💡 尝试修改和扩展功能

## 获取帮助

- 查看详细文档
- 检查示例代码
- 阅读源代码注释
- 提交 Issue

祝你使用愉快！🚀
