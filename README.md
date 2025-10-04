# ChatBOT.EDU - 教育聊天机器人

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.13-green.svg)](https://langchain.com)

一个基于 LangChain 和 Streamlit 构建的智能教育聊天机器人，支持多种对话模式和文档处理功能。

## 🚀 功能特性

### 核心功能
- **智能对话**: 基于 OpenAI GPT 的自然语言对话
- **上下文记忆**: 记住对话历史，提供连贯的交互体验
- **流式响应**: 实时显示 AI 回复，提升用户体验
- **多模态支持**: 支持文本、图片等多种输入方式

### 高级功能
- **文档问答**: 上传 PDF 文档，基于文档内容回答问题
- **网络搜索**: 集成 Tavily 搜索，获取最新信息
- **SQL 查询**: 支持自然语言查询 SQLite 数据库
- **网站分析**: 分析网页内容并回答问题
- **对话管理**: 保存、加载、删除对话历史

### 技术特性
- **向量嵌入**: 使用 FastEmbed 进行文档向量化
- **RAG 架构**: 检索增强生成，提高回答准确性
- **模块化设计**: 易于扩展和维护
- **Docker 支持**: 容器化部署

## 📦 安装与运行

### 环境要求
- Python 3.9+
- OpenAI API Key
- Tavily API Key (可选，用于网络搜索)

### 快速开始

1. **克隆项目**
```bash
git clone https://github.com/james20141606/chatbot.git
cd chatbot
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境**
```bash
# 复制配置文件
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# 编辑配置文件，添加你的 API 密钥
# OPENAI_API_KEY = "your-openai-api-key"
# TAVILY_API_KEY = "your-tavily-api-key" (可选)
```

4. **启动应用**
```bash
python run_app.py
```

5. **访问应用**
打开浏览器访问: http://localhost:8501

## 🏗️ 项目结构

```
chatbot/
├── app.py                 # 主应用文件
├── chatbot_edu.py         # 教育版聊天机器人
├── example.py             # 示例和测试文件
├── utils.py               # 工具函数
├── streaming.py           # 流式响应处理
├── run_app.py             # 启动脚本
├── requirements.txt       # 依赖包列表
├── Dockerfile            # Docker 配置
├── assets/               # 资源文件
│   └── Chinook.db        # SQLite 示例数据库
├── .streamlit/           # Streamlit 配置
│   ├── secrets.toml.example
│   └── secrets.toml      # 配置文件 (需要创建)
└── README.md             # 项目说明
```

## 🔧 配置说明

### API 密钥配置
在 `.streamlit/secrets.toml` 中配置以下密钥：

```toml
# OpenAI API 密钥 (必需)
OPENAI_API_KEY = "sk-proj-..."

# Tavily 搜索 API 密钥 (可选)
TAVILY_API_KEY = "tvly-..."

# 其他配置
OPENAI_MODEL = "gpt-4"
TEMPERATURE = 0.7
```

### 模型配置
- **默认模型**: GPT-4
- **温度参数**: 0.7 (可调节创造性)
- **最大令牌**: 4000
- **流式响应**: 启用

## 🎯 使用指南

### 基本对话
1. 在输入框中输入问题
2. 点击发送或按 Enter
3. 查看 AI 的实时回复

### 文档问答
1. 点击"上传文档"按钮
2. 选择 PDF 文件
3. 等待文档处理完成
4. 基于文档内容提问

### 网络搜索
1. 启用"网络搜索"功能
2. 输入需要搜索的问题
3. AI 会搜索最新信息并回答

### 数据库查询
1. 选择"SQL 查询"模式
2. 用自然语言描述查询需求
3. 系统会生成并执行 SQL 查询

## 🐳 Docker 部署

```bash
# 构建镜像
docker build -t chatbot-edu .

# 运行容器
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key chatbot-edu
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LangChain](https://langchain.com) - 强大的 LLM 应用框架
- [Streamlit](https://streamlit.io) - 快速构建数据应用
- [OpenAI](https://openai.com) - 提供强大的语言模型
- [Tavily](https://tavily.com) - 智能网络搜索服务

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 项目 Issues: [GitHub Issues](https://github.com/james20141606/chatbot/issues)
- 邮箱: [你的邮箱]

---

⭐ 如果这个项目对你有帮助，请给它一个星标！

