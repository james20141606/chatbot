# -*- coding: utf-8 -*-
"""
ChatBOT.EDU 工具函数
包含 MongoDB/GridFS 集成、模型调用、会话管理等
"""

import os
import json
from typing import Optional, List, Tuple, Dict
from datetime import datetime

# MongoDB and GridFS imports
try:
    from pymongo import MongoClient
    from gridfs import GridFS
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("⚠️ MongoDB not available: pymongo not installed")

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI

# OpenAI imports
try:
    from openai import OpenAI as OpenAIClient
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not available: openai not installed")

# ====== MongoDB 连接 ======
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "chatbot")

_client = None
_db = None
_fs = None

if MONGODB_AVAILABLE:
    try:
        _client = MongoClient(MONGO_URI)
        _db = _client[MONGO_DB]
        _fs = GridFS(_db)
        print(f"✅ Connected to MongoDB: {MONGO_DB}")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        _client = None
        _db = None
        _fs = None

# ====== 本地 JSON 兜底 ======
_FILE = "conversations.json"

def load_conversations_from_file() -> dict:
    """从本地JSON文件加载会话"""
    if os.path.exists(_FILE):
        try:
            with open(_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load conversations.json: {e}")
            return {}
    return {}

def save_conversations_to_file(data: dict):
    """保存会话到本地JSON文件"""
    try:
        # 移除不可序列化的对象
        clean_data = {}
        for conv_id, conv_data in data.items():
            clean_conv = {k: v for k, v in conv_data.items() 
                         if k not in ["memory", "vectorstore"]}
            clean_data[conv_id] = clean_conv
        
        with open(_FILE, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save conversations.json: {e}")

# ====== MongoDB：会话/消息 ======
def load_conversations_from_mongodb() -> dict:
    """从MongoDB加载会话数据"""
    if not MONGODB_AVAILABLE or not _db:
        return {}
    
    try:
        data = {}
        for c in _db.conversations.find({}):
            cid = c["_id"]
            # 加载最近的消息（可调整数量）
            msgs = list(_db.messages.find({"conv_id": cid}).sort("_id", 1))
            
            data[cid] = {
                "id": cid,
                "name": c.get("title", cid),
                "created_at": c.get("created_at", datetime.utcnow()).isoformat() if isinstance(c.get("created_at"), datetime) else c.get("created_at", datetime.utcnow().isoformat()),
                "updated_at": c.get("updated_at", datetime.utcnow()).isoformat() if isinstance(c.get("updated_at"), datetime) else c.get("updated_at", datetime.utcnow().isoformat()),
                "messages": [
                    {
                        "role": m["role"],
                        "content": m.get("content", ""),
                        "images": [
                            {"mime": f.get("mime") or "application/octet-stream",
                             "data": _fs.get(f["gridfs_id"]).read()}
                            for f in m.get("images", [])
                        ]
                    } for m in msgs
                ],
                "memory": None,
                "vectorstore": None,
                "model": c.get("model", "gpt-5-mini"),
                "enable_internet": c.get("enable_internet", False),
                "enable_documents": c.get("enable_documents", True),
                "enable_images": c.get("enable_images", True),
                "counter": c.get("counter", 0),
                "user_id": c.get("user_id")
            }
        return data
    except Exception as e:
        print(f"❌ Failed to load conversations from MongoDB: {e}")
        return {}

def save_conversation_to_mongodb(conv: dict) -> Optional[str]:
    """保存会话到MongoDB"""
    if not MONGODB_AVAILABLE or not _db:
        return None
    
    try:
        conv_doc = {
            "_id": conv["id"],
            "user_id": conv.get("user_id"),
            "title": conv["name"],
            "model": conv.get("model", "gpt-5-mini"),
            "enable_internet": conv.get("enable_internet", False),
            "enable_documents": conv.get("enable_documents", True),
            "enable_images": conv.get("enable_images", True),
            "counter": conv.get("counter", 0),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        _db.conversations.insert_one(conv_doc)
        return conv["id"]
    except Exception as e:
        print(f"❌ Failed to save conversation to MongoDB: {e}")
        return None

def update_conversation_in_mongodb(conv_id: str, fields: dict) -> bool:
    """更新会话信息"""
    if not MONGODB_AVAILABLE or not _db:
        return False
    
    try:
        update_fields = dict(fields)
        update_fields["updated_at"] = datetime.utcnow()
        _db.conversations.update_one({"_id": conv_id}, {"$set": update_fields})
        return True
    except Exception as e:
        print(f"❌ Failed to update conversation in MongoDB: {e}")
        return False

def save_message_to_mongodb(conv_id: str, role: str, content: str,
                           images: Optional[List[Tuple[str, bytes]]] = None) -> bool:
    """保存消息到MongoDB"""
    if not MONGODB_AVAILABLE or not _db or not _fs:
        return False
    
    try:
        files = []
        if images:
            for fn, raw in images:
                gid = _fs.put(raw, filename=fn)
                files.append({
                    "filename": fn, 
                    "mime": None, 
                    "gridfs_id": gid
                })
        
        _db.messages.insert_one({
            "conv_id": conv_id,
            "role": role,
            "content": content,
            "images": files,
            "created_at": datetime.utcnow()
        })
        return True
    except Exception as e:
        print(f"❌ Failed to save message to MongoDB: {e}")
        return False

# ====== 嵌入/模型 ======
def configure_embedding_model():
    """配置嵌入模型"""
    try:
        # 使用 HuggingFace 嵌入模型
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        print(f"⚠️ Failed to load embedding model: {e}")
        # 降级到 FastEmbed
        try:
            return FastEmbedEmbeddings()
        except Exception as e2:
            print(f"❌ All embedding models failed: {e2}")
            return None

def get_gpt_client():
    """获取GPT客户端"""
    if not OPENAI_AVAILABLE:
        return None
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY not found")
        return None
    
    try:
        return OpenAIClient(api_key=api_key)
    except Exception as e:
        print(f"❌ Failed to create OpenAI client: {e}")
        return None

def call_gpt_api(messages_payload: List[Dict], model: str = "gpt-4o-mini", stream: bool = True):
    """调用GPT API"""
    if not OPENAI_AVAILABLE:
        return None
    
    try:
        client = get_gpt_client()
        if not client:
            return None
        
        # 让模型自动决定temperature
        response = client.chat.completions.create(
            model=model,
            messages=messages_payload,
            stream=stream
        )
        
        return response
    except Exception as e:
        print(f"❌ GPT API call failed: {e}")
        return None

def generate_conversation_summary(messages: List[Dict]) -> str:
    """生成会话摘要"""
    try:
        # 使用GPT生成摘要
        client = get_gpt_client()
        if client:
            user_messages = [m["content"] for m in messages if m.get("role") == "user"]
            if user_messages:
                summary_prompt = f"请为以下对话生成一个简洁的标题（不超过20字）：\n\n{user_messages[0][:200]}..."
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": summary_prompt}],
                        max_tokens=50,
                        temperature=0.3
                    )
                    return response.choices[0].message.content.strip()
                except Exception:
                    pass
        
        # 降级方案：取第一条用户消息的前20字
        for m in messages:
            if m.get("role") == "user":
                content = (m.get("content") or "").strip()
                return content[:20] if content else "New conversation"
        
        return "New conversation"
    except Exception as e:
        print(f"⚠️ Failed to generate conversation summary: {e}")
        return "New conversation"

def configure_llm():
    """配置LangChain LLM（兜底用）"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return ChatOpenAI(
                openai_api_key=api_key,
                model_name="gpt-4o-mini",
                temperature=0.2
            )
        else:
            return OpenAI(temperature=0.2)
    except Exception as e:
        print(f"⚠️ Failed to configure LLM: {e}")
        return None
