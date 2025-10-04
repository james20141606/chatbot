# ChatBOT.EDU - 重构版智能教育助手

## 🎯 新架构特性

### ✨ 完整用户追踪系统
- **设备指纹识别**: UA + 屏幕 + 时区 + 语言 + IP前三段
- **地理位置解析**: 支持MaxMind GeoIP数据库
- **多设备支持**: 同一用户多设备管理
- **隐私合规**: 透明数据收集，用户知情同意

### 🗄️ 现代化数据存储
- **MongoDB + GridFS**: 会话、消息、图片文件存储
- **本地JSON兜底**: MongoDB不可用时自动降级
- **向量化文档**: PDF文档自动切分和向量化
- **完整索引**: 支持高效查询和检索

### 🎨 现代化UI设计
- **两栏布局**: 左侧聊天，右侧工具面板
- **吸顶工具**: 右侧工具面板固定位置
- **原生输入**: 使用`st.chat_input()`支持回车发送
- **响应式设计**: 适配不同屏幕尺寸

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo>
cd langchain-chatbot-master

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境

```bash
# 复制环境配置
cp env.example .env

# 编辑配置文件
vim .env
```

**必需配置:**
```env
MONGO_URI=mongodb://localhost:27017
MONGO_DB=chatbot
OPENAI_API_KEY=sk-your-key-here
TRACK_API_URL=http://localhost:8666
```

**可选配置:**
```env
# 地理信息解析（需要下载GeoIP数据库）
GEOLITE_DB=./GeoLite2-City.mmdb
```

### 3. 启动服务

#### 方法一：一键启动（推荐）
```bash
./start.sh
```

#### 方法二：手动启动
```bash
# 终端1：启动追踪后端
uvicorn track_api:app --host 0.0.0.0 --port 8666

# 终端2：启动前端
streamlit run app.py
```

### 4. 访问应用

- **前端界面**: http://localhost:8501
- **追踪API**: http://localhost:8666
- **API文档**: http://localhost:8666/docs

## 📁 文件结构

```
langchain-chatbot-master/
├── app.py                 # 主应用（重构版）
├── track_api.py          # 用户追踪后端API
├── utils_new.py          # 工具函数（MongoDB集成）
├── streaming.py          # 流式响应处理
├── start.sh              # 一键启动脚本
├── env.example           # 环境配置示例
├── requirements.txt      # Python依赖
└── README_NEW_ARCHITECTURE.md
```

## 🔧 核心功能

### 用户追踪系统

**设备指纹生成:**
```python
fp_src = f"{user_agent}|{screen}|{timezone}|{language}|{ip_prefix}"
device_id = "dev_" + sha256(fp_src.encode()).hexdigest()[:24]
```

**用户ID管理:**
```python
uid = cookie_uid or "usr_" + sha256((device_id + ip).encode()).hexdigest()[:24]
```

### 数据存储架构

**MongoDB集合设计:**
- `users`: 用户信息和设备管理
- `conversations`: 会话元数据
- `messages`: 消息内容和图片引用
- `fs.files` / `fs.chunks`: GridFS文件存储

### 多模态支持

**图片处理:**
```python
# 图片上传 -> GridFS存储 -> MongoDB引用
gid = _fs.put(raw_data, filename=filename)
```

**文档处理:**
```python
# PDF -> 文本切分 -> 向量化 -> 内存检索
vectorstore = DocArrayInMemorySearch.from_documents(texts, embeddings)
```

## 🛠️ 开发指南

### 添加新功能

1. **后端API**: 在`track_api.py`中添加新端点
2. **前端界面**: 在`app.py`中添加新的UI组件
3. **数据处理**: 在`utils_new.py`中添加工具函数

### 自定义模型

```python
def call_gpt_api(messages, model="gpt-4o-mini", stream=True):
    # 替换为你的模型调用逻辑
    pass
```

### 扩展存储

```python
# 添加新的MongoDB集合
_db.custom_collection.insert_one({...})
```

## 📊 监控和分析

### API统计
```bash
curl http://localhost:8666/stats
```

### 用户行为分析
```python
# MongoDB查询示例
db.users.aggregate([
    {"$group": {"_id": "$geo.country", "count": {"$sum": 1}}}
])
```

## 🔒 隐私和安全

### 数据收集
- **设备信息**: 用于统计和设备管理
- **IP地址**: 用于地理位置分析
- **对话内容**: 用于AI训练和改进

### 合规要求
- 用户同意数据收集
- 透明化数据处理
- 支持数据删除请求

## 🚨 故障排除

### 常见问题

1. **MongoDB连接失败**
   ```bash
   # 检查MongoDB服务
   sudo systemctl status mongod
   ```

2. **追踪API无法访问**
   ```bash
   # 检查端口占用
   netstat -tlnp | grep 8666
   ```

3. **图片上传失败**
   ```bash
   # 检查GridFS权限
   db.fs.files.countDocuments({})
   ```

### 日志查看
```bash
# 查看追踪API日志
uvicorn track_api:app --host 0.0.0.0 --port 8666 --log-level debug
```

## 🔄 迁移指南

### 从旧版本迁移

1. **备份数据**
   ```bash
   cp conversations.json conversations_backup.json
   ```

2. **更新配置**
   ```bash
   # 添加新的环境变量
   echo "MONGO_URI=mongodb://localhost:27017" >> .env
   ```

3. **测试功能**
   ```bash
   # 验证所有功能正常
   python -c "import utils_new; print('✅ Utils loaded')"
   ```

## 📈 性能优化

### 数据库优化
- 添加索引：`db.conversations.createIndex({"user_id": 1})`
- 连接池配置：`MongoClient(maxPoolSize=50)`

### 前端优化
- 图片压缩：自动调整图片大小
- 缓存策略：本地缓存会话数据

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

MIT License - 详见LICENSE文件

## 📞 支持

- **Issues**: GitHub Issues
- **文档**: README文件
- **示例**: 查看代码注释

---

**🎓 ChatBOT.EDU - 让AI教育更智能、更安全、更高效！**
