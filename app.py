# -*- coding: utf-8 -*-
"""
ChatBOT.EDU - 智能教育助手
重构版本：包含用户追踪、MongoDB存储、现代化UI
"""

import os
import io
import base64
import random
import time
import uuid
import json
from typing import Optional, List, Tuple, Dict
from datetime import datetime

import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

import utils
from streaming import GPTStreamHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="ChatBOT.EDU - Intelligent Education Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Cookies 管理 ----------
cookies = EncryptedCookieManager(prefix="chatbot_edu")
if not cookies.ready:
    st.stop()

def ensure_uid() -> str:
    """确保用户有唯一ID"""
    uid = cookies.get("uid")
    if not uid:
        uid = f"usr_{uuid.uuid4().hex[:24]}"
        cookies["uid"] = uid
        cookies.save()
    return uid

USER_ID = ensure_uid()
TRACK_API_URL = os.getenv("TRACK_API_URL", "http://localhost:8666")

# ---------- 首次加载：上报设备信息到追踪后端 ----------
if "tracked" not in st.session_state:
    st.session_state["tracked"] = True
    st.markdown(f"""
    <script>
    (async () => {{
      const payload = {{
        user_agent: navigator.userAgent,
        screen: {{w: screen.width, h: screen.height, dpr: window.devicePixelRatio}},
        tz: Intl.DateTimeFormat().resolvedOptions().timeZone,
        lang: navigator.language,
        cookie_uid: "{USER_ID}"
      }};
      try {{
        await fetch("{TRACK_API_URL}/track", {{
          method:"POST", 
          headers:{{"Content-Type":"application/json"}},
          body: JSON.stringify(payload)
        }});
        console.log("✅ Device tracking completed");
      }} catch (e) {{ 
        console.log("⚠️ Device tracking failed:", e); 
      }}
    }})();
    </script>
    """, unsafe_allow_html=True)

# ---------- 全局样式 ----------
st.markdown("""
<style>
:root{
  --bg:#0b0d12; 
  --card:#11131a; 
  --border:rgba(148,163,255,.16); 
  --muted:#a9aec7;
  --brand:#8b92ff;
}

.main .block-container{
  max-width:1180px;
  padding-top:1rem;
  padding-bottom:5.5rem;
}

/* 右侧工具吸顶 */
#tools-panel{
  position:sticky; 
  top:12px;
}

/* 卡片样式 */
.card{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:16px;
  padding:14px 16px;
  margin-bottom:14px;
  box-shadow:0 6px 18px rgba(0,0,0,.25);
}

.card h4{
  margin:0 0 8px 0;
  letter-spacing:.2px;
}

.small{
  font-size:13px;
  color:var(--muted);
}

/* 控件圆角统一 */
div[data-baseweb="select"], 
div[data-baseweb="input"]{
  border-radius:12px;
}

/* 消息气泡 */
.bubble{
  border-radius:14px;
  padding:12px 14px;
  margin:6px 0;
  display:inline-block;
  max-width:100%;
}

.bubble.user{
  background:#1f2a44;
}

.bubble.assistant{
  background:#191a22;
  border:1px solid var(--border);
}

/* 图片标题 */
.figcap{
  font-size:12px;
  color:#a9a9b2;
  text-align:center;
  margin-top:4px;
}

/* 隐私提示 */
.privacy-notice{
  background:rgba(255,193,7,0.1);
  border:1px solid rgba(255,193,7,0.3);
  border-radius:8px;
  padding:8px 12px;
  font-size:12px;
  color:rgba(255,193,7,0.9);
  margin:8px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------- 辅助函数 ----------
def format_datetime(dt_str):
    """格式化日期时间"""
    try:
        if isinstance(dt_str, str):
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
        return 'Unknown'
    except:
        return 'Unknown'

def file_mime(name: str) -> str:
    """获取文件MIME类型"""
    n = name.lower()
    if n.endswith(".png"): return "image/png"
    if n.endswith(".jpg") or n.endswith(".jpeg"): return "image/jpeg"
    if n.endswith(".webp"): return "image/webp"
    if n.endswith(".gif"): return "image/gif"
    if n.endswith(".pdf"): return "application/pdf"
    return "application/octet-stream"

def to_data_url(mime: str, raw: bytes) -> str:
    """转换为data URL"""
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def messages_to_responses_payload(msgs: List[Dict]) -> List[Dict]:
    """转换为Chat Completions API格式"""
    payload = []
    for m in msgs:
        text = (m.get("content") or "").strip()
        images = m.get("images", [])
        if images:
            content = []
            if text: 
                content.append({"type":"text","text":text})
            for img in images:
                content.append({
                    "type":"image_url",
                    "image_url":{"url": to_data_url(img["mime"], img["data"])}
                })
            payload.append({"role": m["role"], "content": content})
        else:
            payload.append({"role": m["role"], "content": text or ""})
    return payload

def render_msg(role: str, text: Optional[str] = None, images: Optional[List[Tuple[str, bytes]]] = None):
    """渲染消息"""
    with st.chat_message(role, avatar="👤" if role=="user" else "🤖"):
        if images:
            cols = st.columns(min(4, len(images)))
            for i, (fn, raw) in enumerate(images):
                with cols[i % len(cols)]:
                    st.image(io.BytesIO(raw), use_column_width=True)
                    st.markdown(f"<div class='figcap'>{fn}</div>", unsafe_allow_html=True)
        if text:
            cls = "user" if role=="user" else "assistant"
            st.markdown(f"<div class='bubble {cls}'>{text}</div>", unsafe_allow_html=True)

# ---------- 会话状态管理 ----------
def initialize_session_state():
    """初始化会话状态"""
    if "conversations" not in st.session_state:
        mongo_loaded = utils.load_conversations_from_mongodb()
        if mongo_loaded:
            st.session_state.conversations = mongo_loaded
            print("✅ Loaded conversations from MongoDB")
        else:
            st.session_state.conversations = utils.load_conversations_from_file()
            print("📁 Loaded conversations from JSON")
    
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    
    if "conversation_counter" not in st.session_state:
        st.session_state.conversation_counter = max(
            (v.get("counter", 0) for v in st.session_state.conversations.values()), default=0
        )

def create_new_conversation():
    """创建新会话"""
    st.session_state.conversation_counter += 1
    conv_id = f"conv_{st.session_state.conversation_counter}"
    
    st.session_state.conversations[conv_id] = {
        "id": conv_id,
        "name": f"Conversation {st.session_state.conversation_counter}",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages": [],
        "memory": ConversationBufferMemory(return_messages=True),
        "vectorstore": None,
        "model": "gpt-5-mini",
        "enable_internet": False,
        "enable_documents": True,
        "enable_images": True,
        "counter": st.session_state.conversation_counter,
        "user_id": USER_ID
    }
    
    st.session_state.current_conversation_id = conv_id
    
    # 保存到MongoDB
    mongodb_id = utils.save_conversation_to_mongodb(st.session_state.conversations[conv_id])
    if mongodb_id:
        print(f"✅ Saved conversation to MongoDB: {mongodb_id}")
    else:
        utils.save_conversations_to_file(st.session_state.conversations)
        print("📁 Saved conversation to JSON")
    
    return conv_id

def delete_conversation(conv_id):
    """删除会话"""
    if conv_id in st.session_state.conversations:
        del st.session_state.conversations[conv_id]
        if st.session_state.current_conversation_id == conv_id:
            st.session_state.current_conversation_id = next(iter(st.session_state.conversations), None)
        utils.save_conversations_to_file(st.session_state.conversations)

def restore_conversation_memory(conversation):
    """恢复会话记忆"""
    if "memory" not in conversation or conversation["memory"] is None:
        conversation["memory"] = ConversationBufferMemory(return_messages=True)
        for msg in conversation.get("messages", []):
            if msg["role"] == "user":
                conversation["memory"].save_context({"input": msg["content"]}, {"output": ""})
            elif msg["role"] == "assistant":
                # 绑定到上一条user消息
                pass

def get_current_conversation():
    """获取当前会话"""
    cid = st.session_state.get("current_conversation_id")
    if cid and cid in st.session_state.conversations:
        conv = st.session_state.conversations[cid]
        restore_conversation_memory(conv)
        return conv
    return None

# ---------- 文档处理 ----------
def setup_document_retrieval(uploaded_files, conversation):
    """设置文档检索"""
    if not uploaded_files:
        return conversation.get("vectorstore")
    
    try:
        documents = []
        for f in uploaded_files:
            if f.name.lower().endswith(".pdf"):
                tmp = f"temp_{f.name}"
                with open(tmp, "wb") as temp_file:
                    temp_file.write(f.getvalue())
                documents += PyPDFLoader(tmp).load()
                os.remove(tmp)
        
        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = splitter.split_documents(documents)
            embeddings = utils.configure_embedding_model()
            if embeddings:
                vs = DocArrayInMemorySearch.from_documents(texts, embeddings)
                conversation["vectorstore"] = vs
                return vs
    except Exception as e:
        st.error(f"Document processing failed: {e}")
    
    return conversation.get("vectorstore")

# ---------- 互联网搜索 ----------
def setup_internet_search():
    """设置互联网搜索"""
    try:
        return DuckDuckGoSearchRun()
    except Exception as e:
        st.warning(f"Internet search setup failed: {e}")
        return None

def search_internet(q, tool):
    """执行互联网搜索"""
    try:
        # 随机延迟，降低被限制的概率
        time.sleep(random.uniform(1.2, 2.4))
        return tool.run(q)
    except Exception as e:
        st.warning(f"Search failed: {e}")
        return None

# ---------- 初始化 ----------
initialize_session_state()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%); padding:12px; border-radius:10px; text-align:center;">
      <h4 style="color:white; margin:0;">🎓 ChatBOT.EDU</h4>
      <p style="color:rgba(255,255,255,.85); font-size:14px; margin:4px 0 0 0;">Smart Education Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # 隐私提示
    st.markdown("""
    <div class="privacy-notice">
      📊 本应用会收集设备信息用于统计分析，不会泄露个人隐私
    </div>
    """, unsafe_allow_html=True)

    if st.button("➕ New Conversation", key="new_conversation"):
        create_new_conversation()
        st.rerun()

    st.markdown("---")
    
    if st.session_state.conversations:
        st.markdown("**💬 My Conversations**")
        for cid, conv in st.session_state.conversations.items():
            c1, c2 = st.columns([4,1])
            with c1:
                is_active = st.session_state.current_conversation_id == cid
                if st.button(conv["name"], key=f"conv_{cid}", 
                           help=f"Created: {format_datetime(conv.get('created_at'))}",
                           type="primary" if is_active else "secondary"):
                    st.session_state.current_conversation_id = cid
                    st.rerun()
            with c2:
                if st.button("🗑️", key=f"del_{cid}", help="Delete conversation"):
                    delete_conversation(cid)
                    st.rerun()
    else:
        st.info("No conversations yet. Create one.")

    st.markdown("---")
    
    current_conv = get_current_conversation()
    if current_conv:
        st.markdown("**⚙️ Settings**")
        new_name = st.text_input("Conversation Name", value=current_conv["name"], key="conv_name")
        if new_name != current_conv["name"]:
            current_conv["name"] = new_name
        
        current_conv["model"] = st.selectbox(
            "🤖 Model", 
            ["gpt-5-mini","gpt-5","gpt-4o-mini"],
            index=["gpt-5-mini","gpt-5","gpt-4o-mini"].index(current_conv["model"]) 
            if current_conv["model"] in ["gpt-5-mini","gpt-5","gpt-4o-mini"] else 0,
            key="model_select"
        )

# ---------- 主界面 ----------
current_conv = get_current_conversation()

if not current_conv:
    # 欢迎页面
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.markdown("### 🧠 Context Awareness\n- Remember conversation history\n- Continuous learning experience\n- Smart conversation management")
    with c2: 
        st.markdown("### 🌐 Internet Access\n- Real-time information search\n- Latest events query\n- Knowledge updates")
    with c3: 
        st.markdown("### 📄 Document Analysis\n- PDF document processing\n- Image content understanding\n- Multimodal interaction")
else:
    # 两栏布局：左侧聊天，右侧工具
    col_chat, col_tools = st.columns([9,3], gap="large")

    # 左侧：聊天历史
    with col_chat:
        for m in current_conv["messages"]:
            imgs = None
            if m.get("images"):
                # 兼容两种图片格式：[(name,bytes)] 或 [{'data','mime'}]
                if isinstance(m["images"][0], tuple):
                    imgs = m["images"]
                else:
                    imgs = [(f"image {i+1}", im["data"]) for i, im in enumerate(m["images"])]
            render_msg(m["role"], m.get("content"), imgs)

    # 右侧：工具面板（吸顶）
    with col_tools:
        st.markdown("<div id='tools-panel'>", unsafe_allow_html=True)

        # PDF文档处理
        with st.container(border=True):
            st.markdown("<div class='card'><h4>📎 Documents</h4>", unsafe_allow_html=True)
            uploaded_docs = st.file_uploader(
                "Upload PDFs", 
                type=["pdf"], 
                accept_multiple_files=True, 
                key="pdf_uploader"
            )
            st.caption("PDF上传后会被切分并建立向量库，用于检索增强。")
            st.markdown("</div>", unsafe_allow_html=True)

        # 图片附件
        with st.container(border=True):
            st.markdown("<div class='card'><h4>🖼️ Attachments</h4>", unsafe_allow_html=True)
            imgs = st.file_uploader(
                "Attach images for next message", 
                type=["png","jpg","jpeg","webp","gif"],
                accept_multiple_files=True, 
                key="img_uploader_right"
            )
            if imgs:
                st.session_state["pending_images"] = [(f.name, f.read()) for f in imgs]
                st.success(f"Will send {len(imgs)} image(s) with your next message.")
            st.caption("选择后会绑定到**下一条**消息。")
            st.markdown("</div>", unsafe_allow_html=True)

        # 互联网搜索开关
        with st.container(border=True):
            st.markdown("<div class='card'><h4>🌐 Search</h4>", unsafe_allow_html=True)
            current_conv["enable_internet"] = st.toggle(
                "Enable internet search", 
                value=current_conv.get("enable_internet", False)
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # 处理PDF文档
        if uploaded_docs:
            st.session_state["_pending_docs"] = uploaded_docs

    # 处理待处理的PDF文档
    if st.session_state.get("_pending_docs"):
        with st.spinner("Processing documents..."):
            vs = setup_document_retrieval(st.session_state["_pending_docs"], current_conv)
            if vs: 
                st.success("✅ PDFs processed successfully.")
            else:  
                st.error("❌ Document processing failed")
        st.session_state["_pending_docs"] = None

    # 聊天输入（原生chat_input，支持回车发送）
    prompt = st.chat_input("Ask anything…")
    if prompt:
        # 获取待发送的图片
        pending_imgs = st.session_state.pop("pending_images", [])
        
        # 渲染用户消息
        render_msg("user", prompt, pending_imgs)
        
        # 准备图片数据（转换为{'mime','data'}格式）
        img_struct = [{"mime": file_mime(fn), "data": raw} for fn, raw in pending_imgs]
        
        # 保存用户消息
        current_conv["messages"].append({
            "role":"user",
            "content":prompt,
            "images":img_struct
        })
        
        # 保存到MongoDB
        utils.save_message_to_mongodb(current_conv["id"], "user", prompt, pending_imgs)

        # 准备消息payload
        msgs = [
            {"role":m["role"], "content":m["content"], "images":m.get("images", [])}
            for m in current_conv["messages"] if m["role"] in ["user","assistant"]
        ]
        
        if img_struct and current_conv.get("enable_images", True):
            messages_payload = messages_to_responses_payload(msgs)
        else:
            messages_payload = [{"role":m["role"],"content":m["content"]} for m in msgs]

        # 互联网搜索增强
        if current_conv.get("enable_internet", False):
            tool = setup_internet_search()
            if tool:
                with st.spinner("🔍 Searching..."):
                    sr = search_internet(prompt, tool)
                if sr:
                    messages_payload.append({
                        "role":"user",
                        "content":f"Internet search results for '{prompt}':\n{sr}"
                    })

        # 调用AI模型
        with st.chat_message("assistant", avatar="🤖"):
            try:
                stream = utils.call_gpt_api(messages_payload, model=current_conv["model"], stream=True)
                if stream:
                    resp = GPTStreamHandler(st.empty()).handle_stream(stream)
                else:
                    resp = "Sorry, I'm having trouble connecting to the AI service. Please try again."
            except Exception as e:
                st.error(str(e))
                resp = "Sorry, something went wrong."
            
            st.markdown(f"<div class='bubble assistant'>{resp}</div>", unsafe_allow_html=True)
            current_conv["messages"].append({"role":"assistant","content":resp,"images":[]})

        # 第一条消息后自动生成摘要
        if sum(1 for m in current_conv["messages"] if m["role"]=="user") == 1:
            try:
                summary = utils.generate_conversation_summary(current_conv["messages"])
                current_conv["name"] = summary
                current_conv["updated_at"] = datetime.now().isoformat()
                
                # 更新MongoDB
                ok = utils.update_conversation_in_mongodb(
                    current_conv["id"], 
                    {"title": summary, "updated_at": current_conv["updated_at"]}
                )
                if not ok:
                    utils.save_conversations_to_file(st.session_state.conversations)
            except Exception:
                pass