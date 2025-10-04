#!/bin/bash

# ChatBOT.EDU 启动脚本

echo "🚀 Starting ChatBOT.EDU..."

# 检查环境变量文件
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Please copy env.example to .env and configure it."
    exit 1
fi

# 加载环境变量
export $(cat .env | grep -v '^#' | xargs)

# 检查MongoDB连接
echo "📊 Checking MongoDB connection..."
python -c "
import pymongo
try:
    client = pymongo.MongoClient('$MONGO_URI')
    client.admin.command('ping')
    print('✅ MongoDB connection successful')
except Exception as e:
    print(f'❌ MongoDB connection failed: {e}')
    exit(1)
"

# 启动追踪后端
echo "🔍 Starting tracking API backend..."
uvicorn track_api:app --host 0.0.0.0 --port 8666 &
TRACK_PID=$!

# 等待后端启动
sleep 3

# 启动Streamlit前端
echo "🎓 Starting ChatBOT.EDU frontend..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# 清理：停止追踪后端
echo "🛑 Stopping tracking API..."
kill $TRACK_PID
