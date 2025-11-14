# 文生视频 AI Agent 架构文档

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户接口层                                │
├─────────────┬─────────────┬─────────────────────────────────────┤
│  CLI 工具   │  Python API │      配置文件 (.json/.env)         │
│  main.py    │  直接导入   │                                     │
└─────────────┴─────────────┴─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI Agent 编排层                              │
│              (text_to_video_agent.py)                           │
├─────────────────────────────────────────────────────────────────┤
│  • 任务规划 (Task Planning)                                     │
│  • 流程编排 (Pipeline Orchestration)                            │
│  • 状态管理 (State Management)                                  │
│  • 错误处理 (Error Handling)                                    │
│  • 结果聚合 (Result Aggregation)                                │
└─────────────────────┬───────────────────┬───────────────────────┘
                      │                   │
          ┌───────────┴──────────┐       │
          ▼                      ▼       ▼
┌───────────────────┐  ┌──────────────────────┐
│   Poe API 客户端  │  │   FFmpeg 视频处理器   │
│  (poe_client.py)  │  │ (video_processor.py)  │
├───────────────────┤  ├──────────────────────┤
│ • Text → Image   │  │ • 音视频合并          │
│ • Image → Video  │  │ • 多视频拼接          │
│ • Text → Audio   │  │ • 转场效果            │
│ • 文件下载       │  │ • 视频缩放            │
│ • 异步/同步接口  │  │ • 音频提取            │
└───────────────────┘  └──────────────────────┘
          │                      │
          ▼                      ▼
┌─────────────────────────────────────────┐
│          外部服务/工具                   │
├─────────────────────────────────────────┤
│  • Poe.com AI Models                   │
│  • FFmpeg (本地安装)                    │
│  • 文件系统 (临时文件/输出)              │
└─────────────────────────────────────────┘
```

## 数据流图

```
用户输入
   │
   ├─ 文本描述 ("美丽的山景")
   ├─ 音频文本 ("欢迎来到大自然")
   └─ 配置参数
   │
   ▼
┌──────────────────────────────────────┐
│  1. Agent 接收并解析任务              │
│     - 验证输入                        │
│     - 规划执行步骤                    │
│     - 初始化资源                      │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│  2. 文生图 (Text → Image)            │
│     Input:  文本描述                  │
│     API:    Poe FLUX-pro             │
│     Output: image_temp_001.png       │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│  3. 图生视频 (Image → Video)          │
│     Input:  image_temp_001.png       │
│     API:    Poe FLUX-pro             │
│     Output: video_temp_001.mp4       │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│  4. 文生音频 (Text → Audio)           │
│     Input:  音频文本                  │
│     API:    Poe Claude-3.5-Sonnet    │
│     Output: audio_temp_001.mp3       │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│  5. 音视频合并                        │
│     Input:  video + audio            │
│     Tool:   FFmpeg                   │
│     Output: scene_final_001.mp4      │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│  6. 多场景拼接 (如果有多个场景)       │
│     Input:  [scene1, scene2, ...]    │
│     Tool:   FFmpeg                   │
│     Output: final_video.mp4          │
└──────────────────────────────────────┘
   │
   ▼
最终输出到 output/
```

## 核心类设计

### 1. TextToVideoAgent

**职责**: 整个流水线的编排者

```python
class TextToVideoAgent:
    # 初始化
    __init__(api_key, config)
    
    # 核心方法
    create_video(scenes, output_name) → PipelineResult
    create_simple_video(description, audio_text) → PipelineResult
    create_scene(description, audio_text, motion) → VideoScene
    
    # 私有方法
    _generate_image(description) → Path
    _generate_video(image_path, motion) → Path
    _generate_audio(text) → Path
    _cleanup_temp_files()
```

**设计模式**: 
- Facade 模式 - 简化复杂子系统的接口
- Pipeline 模式 - 流水线式数据处理

### 2. PoeClient

**职责**: AI 模型调用的抽象层

```python
class PoeClient:
    # API 调用
    text_to_image(prompt, model, output_path) → GenerationResult
    image_to_video(image_path, motion, model) → GenerationResult
    text_to_audio(text, voice, model) → GenerationResult
    
    # 工具方法
    _make_request(bot_name, prompt, attachments) → GenerationResult
    _download_file(url, output_path) → bool
```

**设计模式**:
- Adapter 模式 - 适配 Poe API 接口
- Strategy 模式 - 支持不同的 AI 模型

### 3. VideoProcessor

**职责**: 视频处理的封装

```python
class VideoProcessor:
    # 视频操作
    merge_audio_video(video, audio, output) → bool
    concatenate_videos(videos, output) → bool
    add_fade_transition(videos, output, duration) → bool
    resize_video(video, output, width, height) → bool
    extract_audio(video, output) → bool
    
    # 信息获取
    get_video_info(video_path) → VideoInfo
    
    # 工具方法
    _run_ffmpeg(args, description) → bool
    _check_ffmpeg()
```

**设计模式**:
- Facade 模式 - 简化 FFmpeg 使用
- Command 模式 - 封装 FFmpeg 命令

## 配置管理

### PipelineConfig

```python
@dataclass
class PipelineConfig:
    # AI 模型配置
    text_to_image_model: str = "FLUX-pro"
    image_to_video_model: str = "FLUX-pro"
    text_to_audio_model: str = "Claude-3.5-Sonnet"
    
    # 路径配置
    output_dir: Path = Path("./output")
    temp_dir: Path = Path("./temp")
    
    # 视频参数
    video_width: int = 1920
    video_height: int = 1080
    audio_volume: float = 1.0
    
    # 处理选项
    add_transitions: bool = True
    transition_duration: float = 1.0
    clean_temp: bool = True
```

**优点**:
- 集中管理配置
- 类型安全
- 易于测试不同配置

## 错误处理策略

### 三层错误处理

```
┌─────────────────────────────────────┐
│  Level 1: API 调用层                │
│  - 捕获网络错误                     │
│  - 超时重试                         │
│  - 返回 GenerationResult            │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Level 2: 场景处理层                │
│  - 单个场景失败不影响其他            │
│  - 记录错误信息                     │
│  - 继续处理后续场景                 │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Level 3: Pipeline 层               │
│  - 聚合所有错误                     │
│  - 部分成功也返回结果                │
│  - 提供详细的错误报告                │
└─────────────────────────────────────┘
```

### 错误类型和处理

```python
# API 错误
try:
    result = await api_call()
except asyncio.TimeoutError:
    # 超时 - 返回错误结果
    return GenerationResult(success=False, error="Timeout")
except Exception as e:
    # 其他错误 - 记录并返回
    logger.error(f"API error: {e}")
    return GenerationResult(success=False, error=str(e))

# 文件错误
if not file.exists():
    logger.error(f"File not found: {file}")
    return None

# FFmpeg 错误
if not self._run_ffmpeg(args):
    logger.error("FFmpeg operation failed")
    return False
```

## 状态管理

### VideoScene 状态

```python
@dataclass
class VideoScene:
    description: str                    # 输入
    audio_text: Optional[str] = None   # 输入
    
    # 生成过程的中间状态
    image_path: Optional[Path] = None
    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    final_path: Optional[Path] = None  # 最终结果
```

**状态转换**:
```
[初始] → [有图片] → [有视频] → [有音频] → [最终完成]
   ↓        ↓          ↓          ↓           ↓
  失败     失败       失败       失败      成功/失败
```

## 性能优化

### 1. 异步并发

```python
# 并发处理多个场景的图片生成
tasks = [
    self._generate_image(scene.description)
    for scene in scenes
]
images = await asyncio.gather(*tasks)
```

### 2. 资源管理

```python
# 自动清理临时文件
try:
    # 处理逻辑
    pass
finally:
    if self.config.clean_temp:
        self._cleanup_temp_files()
```

### 3. 缓存策略

```python
# 复用已生成的图片
if cache.has(description):
    return cache.get(description)
else:
    image = generate_image(description)
    cache.set(description, image)
    return image
```

## 可观测性

### 日志级别

```python
logging.INFO    # 正常流程信息
logging.DEBUG   # 调试详细信息
logging.WARNING # 警告但可继续
logging.ERROR   # 错误信息
```

### 日志内容

```
🚀 Starting video generation pipeline with 3 scenes
📍 Processing scene 1/3
📸 Generating image: A beautiful sunset...
✓ Image generated: image_20231111_120001.png
🎬 Generating video from image_20231111_120001.png...
✓ Video generated: video_20231111_120002.mp4
🎵 Generating audio: As the sun sets...
✓ Audio generated: audio_20231111_120003.mp3
🔗 Merging audio into video: scene_final_001.mp4
✓ Merging audio into video: scene_final_001.mp4 completed
...
✅ Pipeline completed successfully in 180.5s
📁 Output: /path/to/output/generated_video_20231111_120000.mp4
```

### 结果元数据

```python
result.metadata = {
    "scenes_count": 3,
    "successful_scenes": 3,
    "failed_scenes": 0,
    "config": {
        "text_to_image_model": "FLUX-pro",
        "image_to_video_model": "FLUX-pro",
        "text_to_audio_model": "Claude-3.5-Sonnet"
    }
}
```

## 扩展点

### 1. 添加新的 AI 服务

```python
class OpenAIClient:
    def text_to_image(self, prompt):
        # 调用 DALL-E API
        pass

# 在 Agent 中使用
agent = TextToVideoAgent(
    ai_client=OpenAIClient(),  # 注入不同的客户端
    video_processor=processor
)
```

### 2. 添加新的视频效果

```python
class VideoProcessor:
    def add_sepia_effect(self, video_path, output_path):
        args = ["-i", str(video_path),
                "-vf", "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
                str(output_path)]
        return self._run_ffmpeg(args, "Adding sepia effect")
```

### 3. 添加中间件

```python
class LoggingMiddleware:
    def before_step(self, step_name):
        logger.info(f"Starting: {step_name}")
    
    def after_step(self, step_name, result):
        logger.info(f"Completed: {step_name}")

agent.add_middleware(LoggingMiddleware())
```

## 测试策略

### 单元测试

```python
# 测试单个组件
def test_ffmpeg_available():
    processor = VideoProcessor()
    assert processor is not None

def test_resize_video():
    success = processor.resize_video(input, output, 640, 480)
    assert success == True
```

### 集成测试

```python
# 测试完整流程
async def test_full_pipeline():
    agent = TextToVideoAgent(api_key, test_config)
    result = await agent.create_simple_video(
        description="test scene",
        output_name="test"
    )
    assert result.success == True
```

### Mock 测试

```python
# Mock API 调用
@mock.patch('poe_client.PoeClient.text_to_image')
def test_with_mock(mock_api):
    mock_api.return_value = GenerationResult(success=True, ...)
    # 测试逻辑
```

## 部署考虑

### 环境要求

```yaml
Python: 3.8+
FFmpeg: 4.0+
依赖: requirements.txt
环境变量: POE_API_KEY
磁盘空间: >= 1GB (用于临时文件)
```

### 资源限制

```python
# 并发限制
MAX_CONCURRENT_SCENES = 5
TIMEOUT_PER_SCENE = 300  # 5分钟

# 文件大小限制
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB
```

### 监控指标

- API 调用次数和成功率
- 平均处理时间
- 错误类型分布
- 磁盘使用情况

## 安全考虑

### API 密钥保护

```python
# 使用环境变量
api_key = os.getenv("POE_API_KEY")

# 不要硬编码
# api_key = "sk-xxx"  # ❌ 错误
```

### 输入验证

```python
# 验证文件路径
if not output_path.parent.exists():
    raise ValueError("Output directory does not exist")

# 验证参数范围
if not 0 <= volume <= 2:
    raise ValueError("Volume must be between 0 and 2")
```

### 临时文件清理

```python
# 确保清理
try:
    process()
finally:
    cleanup_temp_files()
```

## 总结

这个架构设计体现了：

1. **关注点分离** - 每个模块职责单一
2. **可扩展性** - 易于添加新功能
3. **容错性** - 多层错误处理
4. **可测试性** - 模块化便于测试
5. **可观测性** - 详细的日志和元数据

**核心理念**: 简单的概念 + 扎实的工程实践 = 可靠的 AI Agent
