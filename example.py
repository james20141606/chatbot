
'''
MONGODB_URI="mongodb+srv://james20141606_db_user:LKNIFss7ETHjcxf0@cluster0.rqhown7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
'''
from __future__ import annotations
import os, io, base64
from typing import List, Dict, Tuple
from datetime import datetime

import streamlit as st
from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError

from pymongo import MongoClient, ASCENDING, ReturnDocument
from bson import ObjectId, Binary
import uuid
import hashlib

# ============== Config ==============
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_SYSTEM = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, concise assistant. Please answer questions carefully and provide accurate information. Always respond in English unless the user specifically asks for another language."
)
MAX_IMAGES_PER_TURN = 4
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]

MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
DB_NAME = os.getenv("MONGODB_DB_NAME", "chatui")
COLL_CONV = "conversations"
COLL_MSG = "messages"
COLL_IMG = "message_images"
COLL_USERS = "users"
COLL_SESSIONS = "user_sessions"
COLL_DELETED = "user_deleted_conversations"


# ============== Mongo ==============
@st.cache_resource(show_spinner=False)
def get_db():
    if not MONGODB_URI:
        raise RuntimeError(
            "Missing MONGODB_URI environment variable. Set it like: "
            "mongodb+srv://<user>:<password>@cluster0.rqhown7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        )

    client = MongoClient(MONGODB_URI, tls=True)
    db = client[DB_NAME]

    # Ensure indexes once per process
    db[COLL_CONV].create_index([("created_at", ASCENDING)])
    db[COLL_CONV].create_index([("user_hash", ASCENDING), ("created_at", ASCENDING)])
    db[COLL_MSG].create_index([("conversation_id", ASCENDING), ("_id", ASCENDING)])
    db[COLL_MSG].create_index([("conversation_id", ASCENDING), ("role", ASCENDING)])
    db[COLL_IMG].create_index([("message_id", ASCENDING)])
    db[COLL_USERS].create_index([("user_hash", ASCENDING)])
    db[COLL_SESSIONS].create_index([("user_hash", ASCENDING), ("created_at", ASCENDING)])
    db[COLL_DELETED].create_index([("user_hash", ASCENDING), ("conversation_id", ASCENDING)])

    return db

# ============== User Tracking ==============
def get_or_create_user_session(db):
    """Get or create user session with tracking info"""
    # Create a simple session-based user ID
    if "user_session_id" not in st.session_state:
        st.session_state.user_session_id = str(uuid.uuid4())

    session_id = st.session_state.user_session_id

    # Create user hash for privacy (hash of session info)
    if "user_hash" not in st.session_state:
        user_info = f"streamlit_{session_id}"
        st.session_state.user_hash = hashlib.sha256(user_info.encode()).hexdigest()[:16]

    user_hash = st.session_state.user_hash

    # Store session info
    session_data = {
        "user_hash": user_hash,
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "last_activity": datetime.utcnow().isoformat(),
        "platform": "streamlit",
        "ip_address": "unknown",  # Streamlit doesn't expose IP directly
        "location": "unknown"     # Would need additional service for geolocation
    }
    
    # Update or insert session
    db[COLL_SESSIONS].update_one(
        {"session_id": session_id},
        {"$set": session_data},
        upsert=True
    )
    
    return user_hash, session_id


# ============== OpenAI Client ==============
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


# ============== DB Helpers ==============
def new_conversation(db, model="gpt-4o-mini", temperature=0.7, user_hash=None, session_id=None) -> ObjectId:
    # Get user tracking info if not provided
    if user_hash is None or session_id is None:
        user_hash, session_id = get_or_create_user_session(db)

    res = db[COLL_CONV].insert_one({
        "created_at": datetime.utcnow().isoformat(),
        "model": model,
        "temperature": temperature,
        "summary": "New Conversation",
        "user_hash": user_hash,
        "session_id": session_id,
        "message_count": 0,
        "image_count": 0,
        "total_tokens_used": 0,
        "has_user_messages": False,
        "updated_at": datetime.utcnow().isoformat()
    })
    cid = res.inserted_id
    db[COLL_MSG].insert_one({
        "conversation_id": cid,
        "role": "system",
        "content": DEFAULT_SYSTEM,
        "ts": datetime.utcnow().isoformat(),
        "user_hash": user_hash,
        "session_id": session_id
    })
    return cid

def list_conversations(db, user_hash: str, limit=20):
    query = {"user_hash": user_hash}
    return list(db[COLL_CONV].find(query).sort([("updated_at", -1), ("_id", -1)]).limit(limit))

def load_messages(db, cid: ObjectId) -> List[Dict]:
    msgs = list(db[COLL_MSG].find({"conversation_id": cid}).sort([("_id", 1)]))
    # 附带图片
    mids = [m["_id"] for m in msgs]
    imgs_by_mid = {}
    if mids:
        for doc in db[COLL_IMG].find({"message_id": {"$in": mids}}).sort([("_id", 1)]):
            imgs_by_mid.setdefault(doc["message_id"], []).append({
                "mime": doc["mime"],
                "data": bytes(doc["data"])  # Binary -> bytes
            })
    result = []
    for m in msgs:
        result.append({
            "id": m["_id"],
            "role": m["role"],
            "content": m.get("content", ""),
            "images": imgs_by_mid.get(m["_id"], [])
        })
    return result

def append_message(db, cid: ObjectId, role: str, content: str, user_hash=None, session_id=None) -> ObjectId:
    # Get user tracking info if not provided
    if user_hash is None or session_id is None:
        user_hash, session_id = get_or_create_user_session(db)
    
    res = db[COLL_MSG].insert_one({
        "conversation_id": cid,
        "role": role,
        "content": content,
        "ts": datetime.utcnow().isoformat(),
        "user_hash": user_hash,
        "session_id": session_id,
        "content_length": len(content) if content else 0,
        "has_images": False  # Will be updated if images are added
    })

    # Update conversation stats (batch update to reduce DB calls)
    now_iso = datetime.utcnow().isoformat()
    update_operations = {"$inc": {"message_count": 1}, "$set": {"updated_at": now_iso}}
    if role == "user":
        update_operations.setdefault("$set", {})
        update_operations["$set"].update({
            "last_user_message": now_iso,
            "has_user_messages": True
        })

    try:
        db[COLL_CONV].update_one(
            {"_id": cid},
            update_operations
        )
    except Exception as e:
        print(f"Failed to update conversation stats: {e}")  # Don't block the UI
    
    return res.inserted_id

def generate_conversation_summary(messages: List[Dict]) -> str:
    """Generate conversation summary with GPT, fallback to first 10 tokens"""
    # Extract user messages
    user_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
    if not user_messages:
        return "New Conversation"
    
    # Fallback: use first 10 tokens from user input
    first_message = user_messages[0].strip()
    fallback_summary = " ".join(first_message.split()[:10])
    
    # Try GPT summary 3 times
    for attempt in range(3):
        try:
            # Take first few user messages
            summary_text = " ".join(user_messages[:3])
            
            client = get_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Summarize the following conversation content into a short title within 10 characters. Return only the title, no other content:"},
                    {"role": "user", "content": summary_text}
                ],
                max_tokens=20
            )
            
            summary = response.choices[0].message.content.strip()
            if summary:
                return summary[:20]
        except Exception as e:
            print(f"Summary attempt {attempt + 1} failed: {e}")
            if attempt == 2:  # Last attempt failed
                return fallback_summary
    
    return fallback_summary

def append_images(db, conversation_id: ObjectId, message_id: ObjectId, files: List[Tuple[str, bytes]], user_hash=None, session_id=None):
    if not files:
        return
    
    # Get user tracking info if not provided
    if user_hash is None or session_id is None:
        user_hash, session_id = get_or_create_user_session(db)
    
    # Insert images with tracking info
    image_docs = []
    for (mime, raw) in files:
        image_docs.append({
            "message_id": message_id,
            "mime": mime,
            "data": Binary(raw),
            "user_hash": user_hash,
            "session_id": session_id,
            "file_size": len(raw),
            "created_at": datetime.utcnow().isoformat()
        })
    
    db[COLL_IMG].insert_many(image_docs)
    
    try:
        # Update message to indicate it has images
        db[COLL_MSG].update_one(
            {"_id": message_id},
            {"$set": {"has_images": True, "image_count": len(files)}}
        )
        
        # Update conversation stats - use conversation_id instead of message_id
        db[COLL_CONV].update_one(
            {"_id": conversation_id},
            {"$inc": {"image_count": len(files)}}
        )
    except Exception as e:
        print(f"Failed to update image stats: {e}")  # Don't block the UI

def update_first_system(db, cid: ObjectId, content: str):
    db[COLL_MSG].find_one_and_update(
        {"conversation_id": cid, "role": "system"},
        {"$set": {"content": content}},
        sort=[("_id", 1)],
        return_document=ReturnDocument.AFTER
    )

def delete_conversation_for_user(db, conversation_id: str, user_hash: str):
    """Mark conversation as deleted for specific user (soft delete)"""
    # Check if already deleted to avoid duplicates
    existing = db[COLL_DELETED].find_one({
        "conversation_id": ObjectId(conversation_id),
        "user_hash": user_hash
    })
    
    if not existing:
        db[COLL_DELETED].insert_one({
            "conversation_id": ObjectId(conversation_id),
            "user_hash": user_hash,
            "deleted_at": datetime.utcnow().isoformat()
        })

def get_user_deleted_conversations(db, user_hash: str):
    """Get list of conversation IDs deleted by user"""
    deleted_docs = list(db[COLL_DELETED].find({"user_hash": user_hash}))
    return [doc["conversation_id"] for doc in deleted_docs]


# ============== Image/Message Helpers ==============
def file_mime(name: str) -> str:
    n = name.lower()
    if n.endswith(".png"): return "image/png"
    if n.endswith(".jpg") or n.endswith(".jpeg"): return "image/jpeg"
    if n.endswith(".webp"): return "image/webp"
    return "application/octet-stream"

def to_data_url(mime: str, raw: bytes) -> str:
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def messages_to_responses_payload(msgs: List[Dict]) -> List[Dict]:
    """
    转换为 Chat Completions API 的 messages 结构：
    支持文本和图片的多模态消息
    """
    payload = []
    for m in msgs:
        text = (m.get("content") or "").strip()
        images = m.get("images", [])
        
        if images:
            # 如果有图片，使用多模态格式
            content = []
            if text:
                content.append({"type": "text", "text": text})
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": to_data_url(img["mime"], img["data"])}
                })
            payload.append({"role": m["role"], "content": content})
        else:
            # 纯文本消息
            payload.append({"role": m["role"], "content": text or ""})
    return payload


# ============== UI ==============
st.set_page_config(
    page_title="ChatBot EDU", 
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sidebar-header h3 {
        color: white;
        margin: 0;
        font-weight: 600;
    }
    .chat-button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.2rem 0;
        width: 100%;
        transition: all 0.3s ease;
    }
    .chat-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .stChatMessage {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Removed main header to save space

# Shared resources reused across UI redraws
db = get_db()
user_hash, session_id = get_or_create_user_session(db)
deleted_conversations = set(get_user_deleted_conversations(db, user_hash))

with st.sidebar:
    # Sidebar header
    st.markdown("""
    <div class="sidebar-header">
        <h4>🎓 ChatBot EDU</h4>
        <p style="color: rgba(255,255,255,0.8); font-size: 14px; margin: 4px 0 0 0;">Intelligent Education Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    model = st.selectbox(
        "🤖 Select Model", 
        ["gpt-4o-mini","gpt-5-mini", "gpt5"],
        index=0,
        help="Choose the AI model to use"
    )
    
    temperature = st.slider(
        "🌡️ Creativity", 
        0.0, 1.5, 0.7, 0.1,
        help="Control response creativity: 0=conservative, 1.5=highly creative. Note: gpt-5-mini and gpt5 use default temperature (1.0)"
    )
    
    # Use default system prompt (no UI)
    sys_prompt = DEFAULT_SYSTEM

    st.markdown("---")
    
    # New chat button
    if st.button("➕ New Chat", key="new_chat", help="Start a new conversation"):
        cid = new_conversation(db, model, temperature, user_hash=user_hash, session_id=session_id)
        st.session_state.conversation_id = str(cid)
        st.session_state.messages = load_messages(db, cid)
        st.session_state.current_model = model
        st.session_state.current_temperature = temperature
        st.rerun()

    # Recent Chats (only show conversations with actual user messages)
    try:
        all_conversations = list_conversations(db, user_hash, 30)
        conversations_with_content = []
        active_cid = st.session_state.get("conversation_id")

        for conv in all_conversations:
            if conv["_id"] in deleted_conversations:
                continue

            if conv.get("has_user_messages"):
                conversations_with_content.append(conv)
                continue

            if conv.get("last_user_message") or conv.get("message_count", 0) > 1:
                conversations_with_content.append(conv)
                continue

            if active_cid and str(conv["_id"]) == active_cid:
                conversations_with_content.append(conv)

        if conversations_with_content:
            st.markdown("**💬 Recent Chats**")
            for conv in conversations_with_content[:15]:  # Limit to 15
                cid = str(conv["_id"])
                summary = conv.get("summary", "New Conversation")
                model_used = conv.get("model", "gpt-4o-mini")
                
                # Format display
                display_text = f"📋 {summary}"
                # if model_used != model:  # If different model, show model name
                #     display_text += f" ({model_used})"
                
                # Create a row with conversation button and delete button
                col1, col2 = st.columns([6, 1])
                
                with col1:
                    if st.button(display_text, key=f"open_{cid}", help=f"Model: {model_used}"):
                        st.session_state.conversation_id = cid
                        st.session_state.messages = load_messages(db, ObjectId(cid))
                        st.session_state.current_model = model_used
                        st.session_state.current_temperature = conv.get("temperature", 0.7)
                        st.rerun()
                
                with col2:
                    # Simple centered delete button
                    if st.button("🗑️", key=f"delete_{cid}", help="Delete conversation", use_container_width=True):
                        delete_conversation_for_user(db, cid, user_hash)
                        if st.session_state.get("conversation_id") == cid:
                            st.session_state.pop("conversation_id", None)
                            st.session_state.pop("messages", None)
                        st.rerun()
                    
    except Exception as e:
        st.error(f"Database connection failed: {e}")

# Initialize session (db already cached)

# Auto-clear cache if conversation_id exists but conversation doesn't exist in DB
if "conversation_id" in st.session_state:
    try:
        # Check if current conversation still exists in database
        conv_exists = db[COLL_CONV].find_one({"_id": ObjectId(st.session_state.conversation_id)})
        if not conv_exists:
            # Conversation was deleted, clear cache
            st.session_state.pop("messages", None)
            st.session_state.pop("conversation_id", None)
            st.session_state.pop("current_model", None)
            st.session_state.pop("current_temperature", None)
    except:
        # Invalid conversation_id, clear cache
        st.session_state.pop("messages", None)
        st.session_state.pop("conversation_id", None)
        st.session_state.pop("current_model", None)
        st.session_state.pop("current_temperature", None)

# Only create new conversation if none exists in session state
if "conversation_id" not in st.session_state:
    # Try to get the most recent conversation for the current user that isn't soft-deleted
    recent_conversations = list_conversations(db, user_hash, 5)
    recent_conv = next(
        (conv for conv in recent_conversations if conv["_id"] not in deleted_conversations),
        None
    )

    if recent_conv:
        st.session_state.conversation_id = str(recent_conv["_id"])
        st.session_state.current_model = recent_conv.get("model", model)
        st.session_state.current_temperature = recent_conv.get("temperature", temperature)
    else:
        # Create new conversation only if the user truly has none
        cid = new_conversation(db, model, temperature, user_hash=user_hash, session_id=session_id)
        st.session_state.conversation_id = str(cid)
        st.session_state.current_model = model
        st.session_state.current_temperature = temperature

# Load messages if not in session state
if "messages" not in st.session_state:
    st.session_state.messages = load_messages(db, ObjectId(st.session_state.conversation_id))

# Sync system prompt
if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
    if st.session_state.messages[0]["content"] != sys_prompt:
        st.session_state.messages[0]["content"] = sys_prompt
        update_first_system(db, ObjectId(st.session_state.conversation_id), sys_prompt)

# Display history
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    
    with st.chat_message(m["role"]):
        if m.get("images"):
            cols = st.columns(min(4, len(m["images"])))
            for i, img in enumerate(m["images"]):
                with cols[i % len(cols)]:
                    st.image(io.BytesIO(img["data"]), caption=f"image {i+1}", use_column_width=True)
        if m.get("content"):
            st.markdown(m["content"])

# ========= Custom Composer (替代 st.chat_input) =========
st.markdown("""
<style>
#composer-wrapper {
  position: fixed;
  bottom: 20px;
  left: calc(18rem + 32px);
  right: 32px;
  z-index: 1000;
  pointer-events: none;
}
#composer-wrapper .composer-surface {
  pointer-events: auto;
  padding: 18px 22px !important;
  background: rgba(24, 25, 34, 0.95) !important;
  border: 1px solid rgba(115, 130, 255, 0.25) !important;
  border-radius: 18px !important;
  box-shadow: 0 24px 45px rgba(0, 0, 0, 0.4) !important;
  backdrop-filter: blur(12px);
}
#composer-wrapper .composer-surface form {
  gap: 0 !important;
}
#composer-wrapper .composer-surface div[data-testid="column"] {
  padding: 0 !important;
}
#composer-wrapper .composer-surface div[data-testid="column"] > div {
  padding: 0 !important;
}
#composer-wrapper .composer-surface div[data-testid="column"]:first-child {
  max-width: 52px !important;
}
#composer-wrapper .composer-surface div[data-testid="column"]:last-child {
  max-width: 80px !important;
  display: flex;
  justify-content: flex-end;
}
#composer-wrapper .composer-surface div[data-baseweb="input"] {
  background: rgba(15, 16, 24, 0.7) !important;
  border: 1px solid rgba(148, 163, 255, 0.25) !important;
  border-radius: 14px !important;
  transition: all 0.2s ease !important;
}
#composer-wrapper .composer-surface div[data-baseweb="input"]:focus-within {
  border-color: #667eea !important;
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.35) !important;
  background: rgba(18, 20, 30, 0.9) !important;
}
#composer-wrapper .composer-surface input[type="text"] {
  color: #ffffff !important;
  padding: 12px 16px !important;
  font-size: 16px !important;
  background: transparent !important;
}
#composer-wrapper .composer-surface input[type="text"]::placeholder {
  color: rgba(255, 255, 255, 0.55) !important;
}
#composer-wrapper .composer-surface div[data-testid="column"]:last-child button {
  height: 44px !important;
  width: 56px !important;
  border-radius: 14px !important;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  border: none !important;
  color: #ffffff !important;
  font-size: 20px !important;
  transition: transform 0.15s ease, box-shadow 0.2s ease !important;
  box-shadow: 0 16px 30px rgba(102, 126, 234, 0.4) !important;
}
#composer-wrapper .composer-surface div[data-testid="column"]:last-child button:hover {
  transform: translateY(-1px) scale(1.02);
  box-shadow: 0 20px 34px rgba(102, 126, 234, 0.45) !important;
}

#composer-wrapper .composer-surface div[data-testid="stFileUploader"] {
  width: 44px !important;
  min-width: 44px !important;
}
#composer-wrapper .composer-surface div[data-testid="stFileUploader"] > label { display: none !important; }
#composer-wrapper .composer-surface div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
}
#composer-wrapper .composer-surface div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div:first-child,
#composer-wrapper .composer-surface div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div:nth-child(2) {
  display: none !important;
}
#composer-wrapper .composer-surface div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button {
  width: 44px !important;
  height: 44px !important;
  border-radius: 12px !important;
  background: rgba(15, 16, 24, 0.88) !important;
  border: 1px solid rgba(148, 163, 255, 0.25) !important;
  color: transparent !important;
  font-size: 0 !important;
  position: relative;
  transition: border-color 0.2s ease, background 0.2s ease;
  cursor: pointer !important;
}
#composer-wrapper .composer-surface div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button::after {
  content: "+";
  color: rgba(255, 255, 255, 0.75);
  font-size: 24px;
  font-weight: 600;
  display: block;
  line-height: 1;
}
#composer-wrapper .composer-surface div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button:hover {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  border-color: transparent !important;
}
#composer-wrapper .composer-surface div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button:hover::after {
  color: #ffffff !important;
}

div[data-testid="stChatMessage"] {
  margin-bottom: 1rem !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageUserMessage"]) {
  display: flex;
  justify-content: flex-end;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageUserMessage"]) > div {
  max-width: 70% !important;
  margin-left: auto !important;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  color: white !important;
  border-radius: 18px 18px 4px 18px !important;
  padding: 12px 16px !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAssistantMessage"]) {
  display: flex;
  justify-content: flex-start;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAssistantMessage"]) > div {
  max-width: 70% !important;
  margin-right: auto !important;
  background: rgba(255, 255, 255, 0.1) !important;
  border-radius: 18px 18px 18px 4px !important;
  padding: 12px 16px !important;
}
.main .block-container {
  padding-bottom: 220px !important;
}

@media (max-width: 1100px) {
  #composer-wrapper {
    left: 24px;
    right: 24px;
  }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div id="composer-wrapper"><div class="composer-surface">', unsafe_allow_html=True)

with st.form("composer", clear_on_submit=True):
    c1, c2, c3 = st.columns([0.09, 0.83, 0.08])
    with c1:
        uploaded = st.file_uploader(
            " ", type=IMAGE_TYPES, accept_multiple_files=True,
            key="uploader", label_visibility="collapsed"
        )

    with c2:
        user_text_input = st.text_input(
            "Ask Anything", key="composer_text", label_visibility="collapsed",
            placeholder="Ask Anything"
        )

    with c3:
        send = st.form_submit_button("➤", use_container_width=True)

st.markdown('</div></div>', unsafe_allow_html=True)

# 校验选择的图片数量与体积
if uploaded:
    if len(uploaded) > MAX_IMAGES_PER_TURN:
        st.warning(f"⚠️ Maximum {MAX_IMAGES_PER_TURN} images allowed. Selected first {MAX_IMAGES_PER_TURN} images.")
        uploaded = uploaded[:MAX_IMAGES_PER_TURN]

    max_size = 10 * 1024 * 1024
    valid_files = []
    for file in uploaded:
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)
        if size > max_size:
            st.error(f"❌ '{file.name}' is too large ({size/1024/1024:.1f}MB). Max 10MB.")
        else:
            valid_files.append(file)
    uploaded = valid_files

# 🚀 发送：点击按钮 或者 仅上传图片也可以发送
if send and (user_text_input or uploaded):
    user_text = user_text_input or ""
    files_to_save: List[Tuple[str, bytes]] = []

    with st.chat_message("user"):
        if uploaded:
            cols = st.columns(min(4, len(uploaded)))
            for i, f in enumerate(uploaded):
                raw = f.read()
                mime = file_mime(f.name)
                files_to_save.append((mime, raw))
                with cols[i % len(cols)]:
                    st.image(io.BytesIO(raw), caption=f.name, use_column_width=True)
        if user_text:
            st.markdown(user_text)

    cid_obj = ObjectId(st.session_state.conversation_id)
    mid = append_message(db, cid_obj, "user", user_text)
    append_images(db, cid_obj, mid, files_to_save)

    st.session_state.messages.append({
        "id": mid, "role": "user", "content": user_text,
        "images": [{"mime": m, "data": d} for (m, d) in files_to_save]
    })

    # 组装消息与调用 API
    messages_payload = messages_to_responses_payload(st.session_state.messages)

    try:
        client = get_client()
        if model in ["gpt-5-mini", "gpt5"]:
            api_params = {"model": model, "messages": messages_payload, "stream": True}
        else:
            api_params = {"model": model, "temperature": temperature, "messages": messages_payload, "stream": True}

        stream = client.chat.completions.create(**api_params)

        assistant_area = st.chat_message("assistant")
        placeholder = assistant_area.empty()
        chunks: List[str] = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                delta = chunk.choices[0].delta.content
                chunks.append(delta)
                placeholder.markdown("".join(chunks))

        final_text = "".join(chunks).strip()
        if not final_text:
            # fallback 非流式
            if model in ["gpt-5-mini", "gpt5"]:
                fallback_params = {"model": model, "messages": messages_payload}
            else:
                fallback_params = {"model": model, "temperature": temperature, "messages": messages_payload}
            resp = client.chat.completions.create(**fallback_params)
            final_text = (resp.choices[0].message.content or "").strip()

        append_message(db, cid_obj, "assistant", final_text)
        st.session_state.messages.append({"id": ObjectId(), "role": "assistant", "content": final_text, "images": []})

        # 首条用户消息后生成摘要
        if len([m for m in st.session_state.messages if m.get("role") == "user"]) == 1:
            summary = generate_conversation_summary(st.session_state.messages)
            db[COLL_CONV].update_one({"_id": cid_obj}, {"$set": {"summary": summary}})

        placeholder.markdown(final_text)

    except AuthenticationError:
        st.error("🔑 Authentication failed. Please check OPENAI_API_KEY.")
    except RateLimitError:
        st.warning("⏰ Rate limit exceeded. Please try again later.")
    except APIConnectionError as e:
        st.error(f"🌐 Network error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
