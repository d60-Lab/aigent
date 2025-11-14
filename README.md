# AI Agent Demo

AI Agent 演示项目 - 展示 AI Agent 的核心概念和实际应用

## 项目内容

### 📝 [Blog: AI Agent 其实挺简单](./blog_ai_agent.md)

一篇深入浅出的技术博客，讲解：
- AI Agent 的核心概念（感知→决策→执行）
- 理论简单但实践有挑战的原因
- 从简单到复杂的演进路径
- 实际案例分析

### 🎬 [Text-to-Video AI Agent](./text_to_video_agent/)

一个完整的文生视频 AI Agent 实现，展示了：

**功能流程：**
1. 文生图（Text → Image）
2. 图生视频（Image → Video）
3. 文生音频（Text → Audio）
4. 音视频合并（Video + Audio）
5. 多视频拼接（Multi-scene Concatenation）

**技术栈：**
- Python 3.8+
- Poe.com API（多模型 AI 能力）
- FFmpeg（视频处理）
- 异步编程（高效 I/O）

**设计特点：**
- 🧩 模块化设计 - 每个步骤独立可测
- 🔄 流水线模式 - 清晰的数据流
- 🛡️ 容错机制 - 多层错误处理
- 📊 可观测性 - 详细日志追踪
- ⚡ 异步支持 - 提升处理效率

## 快速开始

### 1. 阅读 Blog

```bash
# 查看 blog
cat blog_ai_agent.md
```

### 2. 运行文生视频 Agent

```bash
# 进入项目目录
cd text_to_video_agent

# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env 文件，添加你的 POE_API_KEY

# 运行简单示例
python examples/simple_example.py

# 或使用命令行
python main.py --description "美丽的日落海景" --audio "夕阳西下，海面波光粼粼"
```

详细使用说明请查看：[text_to_video_agent/README.md](./text_to_video_agent/README.md)

## 项目结构

```
aigent/
├── README.md                          # 项目总览（本文件）
├── blog_ai_agent.md                   # AI Agent 技术博客
└── text_to_video_agent/               # 文生视频 AI Agent
    ├── README.md                      # 详细文档
    ├── requirements.txt               # Python 依赖
    ├── .env.example                   # 环境变量模板
    ├── main.py                        # 命令行入口
    ├── src/                           # 源代码
    │   ├── __init__.py
    │   ├── text_to_video_agent.py    # 主 Agent 类
    │   ├── poe_client.py             # Poe API 客户端
    │   └── video_processor.py        # FFmpeg 视频处理
    ├── examples/                      # 使用示例
    │   ├── simple_example.py         # 单场景示例
    │   ├── multi_scene_example.py    # 多场景示例
    │   └── scenes_example.json       # 场景配置示例
    ├── output/                        # 输出视频
    └── temp/                          # 临时文件
```

## 核心概念演示

### AI Agent 的本质

```python
# 这就是 AI Agent 的核心循环
while not task_completed:
    # 1. 感知：获取当前状态
    state = observe_environment()
    
    # 2. 决策：让 LLM 决定下一步
    action = llm.decide(state, available_tools)
    
    # 3. 执行：调用工具完成任务
    result = execute_tool(action)
    
    # 4. 反馈：更新状态
    update_state(result)
```

### 实际应用示例

本项目的文生视频 Agent 就是这个循环的具体体现：

```python
# Agent 接收任务
task = "创建一段山景视频"

# Agent 规划步骤
steps = [
    "生成山景图片",
    "将图片转为视频",
    "生成配音",
    "合并音视频"
]

# Agent 执行每个步骤
for step in steps:
    result = execute(step)
    if failed(result):
        handle_error()
```

## 学习路径

1. **理解概念** → 阅读 [blog_ai_agent.md](./blog_ai_agent.md)
2. **看懂代码** → 查看 [text_to_video_agent/src/](./text_to_video_agent/src/)
3. **运行示例** → 执行 [examples/](./text_to_video_agent/examples/)
4. **自己实践** → 修改代码，添加新功能

## 扩展方向

基于这个项目，你可以：

### 功能扩展
- 添加字幕生成
- 支持更多视频特效
- 实现批量处理
- 添加视频编辑功能

### 架构改进
- 添加任务队列
- 实现分布式处理
- 增加缓存机制
- 添加监控面板

### 更多 Agent
- 文档生成 Agent
- 代码审查 Agent
- 数据分析 Agent
- 自动化测试 Agent

## 技术亮点

### 1. 模块化设计
每个功能都是独立模块，易于测试和维护：
- `poe_client.py` - 纯 AI 能力调用
- `video_processor.py` - 纯视频处理
- `text_to_video_agent.py` - 编排和协调

### 2. 错误处理
多层次的错误处理策略：
- API 调用失败自动重试
- 单个场景失败不影响其他场景
- 详细的错误日志便于调试

### 3. 配置驱动
通过 `PipelineConfig` 灵活配置：
- 模型选择
- 视频参数
- 处理选项

### 4. 同步/异步支持
同时提供同步和异步接口：
```python
# 异步（高性能）
agent = TextToVideoAgent(api_key)
result = await agent.create_video(scenes)

# 同步（简单易用）
agent = TextToVideoAgentSync(api_key)
result = agent.create_video(scenes)
```

## 相关资源

- [Poe API 文档](https://developer.poe.com/)
- [FFmpeg 文档](https://ffmpeg.org/documentation.html)
- [Python asyncio 文档](https://docs.python.org/3/library/asyncio.html)

## 常见问题

### Q: 需要什么基础？
A: Python 基础 + 对 AI 的兴趣

### Q: 成本如何？
A: 使用 Poe API 会产生费用，建议先小规模测试

### Q: 可以用其他 AI API 吗？
A: 可以！只需实现对应的客户端接口即可

### Q: 生成的视频质量如何？
A: 取决于：
- 使用的 AI 模型
- 文本描述的质量
- 视频参数配置

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

---

**记住：AI Agent 的核心很简单，但要做好需要在细节上下功夫。动手实践是最好的学习方式！**
