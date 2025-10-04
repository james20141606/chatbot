# 🎓 ChatBOT.EDU

智能教育助手 - 集成上下文感知、互联网访问和文档分析功能

## 🌟 主要功能

- **🧠 上下文感知**: 记住对话历史，提供连续的学习体验
- **🌐 互联网访问**: 实时网络搜索获取最新信息
- **📄 文档分析**: 上传PDF文档进行问答和分析
- **🖼️ 图像理解**: 分析和讨论图像内容
- **🎯 GPT集成**: 使用OpenAI最新的GPT模型

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- OpenAI API密钥

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置API密钥

创建 `.streamlit/secrets.toml` 文件：

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

或者设置环境变量：

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 4. 启动应用

```bash
python run_app.py
```

或者直接使用streamlit：

```bash
streamlit run chatbot_edu.py
```

## 📱 使用说明

1. **创建对话**: 点击左侧"➕ 新建对话"按钮创建新的对话
2. **管理对话**: 在左侧对话列表中切换、重命名或删除对话
3. **配置设置**: 为每个对话独立配置模型、温度和功能开关
4. **多模态交互**: 
   - 🌐 互联网访问: 获取实时信息
   - 📄 文档分析: 上传PDF进行分析
   - 🖼️ 图像分析: 上传图片进行讨论
5. **开始对话**: 在输入框中提问或上传文件

## 🔧 功能详解

### 上下文感知
- 记住整个对话历史
- 基于之前的讨论提供相关回答
- 支持多轮对话

### 互联网搜索
- 使用DuckDuckGo或Tavily API进行实时搜索
- 自动将搜索结果整合到回答中
- 获取最新的事件和信息

### 文档分析
- 支持PDF文档上传
- 使用RAG (检索增强生成)技术
- 基于文档内容回答问题

### 图像理解
- 支持PNG、JPG、JPEG、WEBP格式
- 使用GPT-4 Vision模型分析图像
- 可以描述、解释和分析图像内容

## 🎯 使用场景

- **学术研究**: 分析研究论文和文档
- **学习辅导**: 解答学习问题和概念
- **信息查询**: 获取最新的新闻和事件
- **图像分析**: 解释图表、图表和视觉内容
- **文档总结**: 快速理解和总结长文档

## 🔑 API配置

### 必需配置
- `OPENAI_API_KEY`: OpenAI API密钥

### 可选配置
- `TAVILY_API_KEY`: Tavily搜索API密钥 (用于更好的搜索体验)
- `OPENAI_BASE_URL`: 自定义OpenAI API端点

## 🛠️ 技术架构

- **前端**: Streamlit
- **AI模型**: OpenAI GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **文档处理**: LangChain + PyPDF
- **向量存储**: DocArrayInMemorySearch
- **搜索**: DuckDuckGo / Tavily API
- **图像处理**: GPT-4 Vision

## 📝 注意事项

1. 确保有足够的OpenAI API额度
2. 大文件上传可能需要较长时间处理
3. 互联网搜索功能需要网络连接
4. 图像分析功能需要GPT-4 Vision模型支持

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License
