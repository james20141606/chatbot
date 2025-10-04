# ChatBOT.EDU - Educational Chatbot

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.13-green.svg)](https://langchain.com)

An intelligent educational chatbot built with LangChain and Streamlit, supporting multiple conversation modes and document processing capabilities.

## 🚀 Features

### Core Features
- **Intelligent Chat**: Natural language conversations powered by OpenAI GPT
- **Context Memory**: Remembers conversation history for coherent interactions
- **Streaming Response**: Real-time AI responses for enhanced user experience
- **Multimodal Support**: Supports text, images, and other input types

### Advanced Features
- **Document Q&A**: Upload PDF documents and ask questions based on content
- **Web Search**: Integrated Tavily search for up-to-date information
- **SQL Queries**: Natural language queries for SQLite databases
- **Website Analysis**: Analyze web content and answer questions
- **Conversation Management**: Save, load, and delete conversation history

### Technical Features
- **Vector Embeddings**: Document vectorization using FastEmbed
- **RAG Architecture**: Retrieval-Augmented Generation for improved accuracy
- **Modular Design**: Easy to extend and maintain
- **Docker Support**: Containerized deployment

## 📦 Installation & Setup

### Requirements
- Python 3.9+
- OpenAI API Key
- Tavily API Key (optional, for web search)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/james20141606/chatbot.git
cd chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
# Copy configuration file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Edit configuration file and add your API keys
# OPENAI_API_KEY = "your-openai-api-key"
# TAVILY_API_KEY = "your-tavily-api-key" (optional)
```

4. **Launch the application**
```bash
python run_app.py
```

5. **Access the application**
Open your browser and visit: http://localhost:8501

## 🏗️ Project Structure

```
chatbot/
├── app.py                 # Main application file
├── chatbot_edu.py         # Educational chatbot version
├── example.py             # Examples and test files
├── utils.py               # Utility functions
├── streaming.py           # Streaming response handling
├── run_app.py             # Launch script
├── requirements.txt       # Dependencies list
├── Dockerfile            # Docker configuration
├── assets/               # Resource files
│   └── Chinook.db        # SQLite sample database
├── .streamlit/           # Streamlit configuration
│   ├── secrets.toml.example
│   └── secrets.toml      # Configuration file (needs to be created)
└── README.md             # Project documentation
```

## 🔧 Configuration

### API Key Configuration
Configure the following keys in `.streamlit/secrets.toml`:

```toml
# OpenAI API Key (required)
OPENAI_API_KEY = "sk-proj-..."

# Tavily Search API Key (optional)
TAVILY_API_KEY = "tvly-..."

# Other configurations
OPENAI_MODEL = "gpt-4"
TEMPERATURE = 0.7
```

### Model Configuration
- **Default Model**: GPT-4
- **Temperature**: 0.7 (adjustable creativity)
- **Max Tokens**: 4000
- **Streaming Response**: Enabled

## 🎯 Usage Guide

### Basic Chat
1. Type your question in the input box
2. Click send or press Enter
3. View the AI's real-time response

### Document Q&A
1. Click the "Upload Document" button
2. Select a PDF file
3. Wait for document processing to complete
4. Ask questions based on the document content

### Web Search
1. Enable the "Web Search" feature
2. Enter your search query
3. AI will search for the latest information and respond

### Database Queries
1. Select "SQL Query" mode
2. Describe your query requirements in natural language
3. The system will generate and execute SQL queries

## 🐳 Docker Deployment

```bash
# Build image
docker build -t chatbot-edu .

# Run container
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key chatbot-edu
```

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) - Powerful LLM application framework
- [Streamlit](https://streamlit.io) - Rapid data app development
- [OpenAI](https://openai.com) - Advanced language models
- [Tavily](https://tavily.com) - Intelligent web search service

## 📞 Contact

For questions or suggestions, please contact us through:

- Project Issues: [GitHub Issues](https://github.com/james20141606/chatbot/issues)
- Email: [james20141606@gmail.com]
---

⭐ If this project helps you, please give it a star!

