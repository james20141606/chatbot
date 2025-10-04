# -*- coding: utf-8 -*-
import os, io, base64, random, time
import streamlit as st
from typing import List, Dict, Tuple
from datetime import datetime

import utils
from streaming import StreamHandler, GPTStreamHandler

from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun


# ===========================
# Page config
# ===========================
st.set_page_config(
    page_title="ChatBOT.EDU - Intelligent Education Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Global CSS (统一观感)
# ===========================
st.markdown("""
<style>
:root{
  --bg:#0b0d12; --card:#11131a; --border:rgba(148,163,255,.16);
  --brand:#8b92ff; --muted:#a9aec7;
}
.main .block-container{max-width:1180px;padding-top:1rem;padding-bottom:5.5rem}

/* 右侧工具吸顶 */
#tools-panel{position:sticky; top:12px}

/* 卡片 */
.card{background:var(--card); border:1px solid var(--border); border-radius:16px;
      padding:14px 16px; margin-bottom:14px; box-shadow:0 6px 18px rgba(0,0,0,.25)}
.card h4{margin:0 0 8px 0; letter-spacing:.2px}
.small{font-size:13px; color:var(--muted)}
hr.sep {border:none; border-top:1px solid var(--border); margin:10px 0}

/* 控件圆角统一 */
div[data-baseweb="select"], div[data-baseweb="input"]{border-radius:12px}

/* 气泡 */
.bubble{border-radius:14px;padding:12px 14px;margin:6px 0;display:inline-block;max-width:100%}
.bubble.user{background:#1f2a44}
.bubble.assistant{background:#191a22;border:1px solid var(--border)}
.figcap{font-size:12px; color:#a9a9b2; text-align:center; margin-top:4px}
</style>
""", unsafe_allow_html=True)

# ===========================
# Helper UI
# ===========================
def render_msg(role: str, text: str | None = None, images: List[Tuple[str, bytes]] | None = None):
    with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
        if images:
            cols = st.columns(min(4, len(images)))
            for i, (fn, raw) in enumerate(images):
                with cols[i % len(cols)]:
                    st.image(io.BytesIO(raw), use_column_width=True)
                    st.markdown(f"<div class='figcap'>{fn}</div>", unsafe_allow_html=True)
        if text:
            cls = "user" if role == "user" else "assistant"
            st.markdown(f"<div class='bubble {cls}'>{text}</div>", unsafe_allow_html=True)

def format_datetime(dt_str):
    try:
        if isinstance(dt_str, str):
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
        return 'Unknown'
    except:
        return 'Unknown'

# ===========================
# Session & conversation
# ===========================
def initialize_session_state():
    if "conversations" not in st.session_state:
        mongo_loaded = utils.load_conversations_from_mongodb()
        if mongo_loaded:
            st.session_state.conversations = mongo_loaded
            print("✅ Loaded from MongoDB")
        else:
            st.session_state.conversations = utils.load_conversations_from_file()
            print("📁 Loaded from JSON")

    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None

    if "conversation_counter" not in st.session_state:
        max_counter = 0
        for conv_data in st.session_state.conversations.values():
            if "counter" in conv_data:
                max_counter = max(max_counter, conv_data["counter"])
        st.session_state.conversation_counter = max_counter

def create_new_conversation():
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
        "counter": st.session_state.conversation_counter
    }
    st.session_state.current_conversation_id = conv_id

    mongodb_id = utils.save_conversation_to_mongodb(st.session_state.conversations[conv_id])
    if mongodb_id:
        print(f"✅ Saved conversation to MongoDB: {mongodb_id}")
    else:
        utils.save_conversations_to_file(st.session_state.conversations)
        print("📁 Saved conversation to JSON")
    return conv_id

def delete_conversation(conv_id):
    if conv_id in st.session_state.conversations:
        del st.session_state.conversations[conv_id]
        if st.session_state.current_conversation_id == conv_id:
            st.session_state.current_conversation_id = next(iter(st.session_state.conversations), None)
        utils.save_conversations_to_file(st.session_state.conversations)

def restore_conversation_memory(conversation):
    if "memory" not in conversation or conversation["memory"] is None:
        conversation["memory"] = ConversationBufferMemory(return_messages=True)
        for msg in conversation.get("messages", []):
            if msg["role"] == "user":
                conversation["memory"].save_context({"input": msg["content"]}, {"output": ""})
            elif msg["role"] == "assistant":
                # 将上一条 user 与当前 assistant 绑定
                messages = conversation["messages"]
                for i, prev in enumerate(messages):
                    if prev == msg:
                        for j in range(i-1, -1, -1):
                            if messages[j]["role"] == "user":
                                conversation["memory"].save_context({"input": messages[j]["content"]},
                                                                     {"output": msg["content"]})
                                break
                        break

def get_current_conversation():
    cid = st.session_state.get("current_conversation_id")
    if cid and cid in st.session_state.conversations:
        conv = st.session_state.conversations[cid]
        restore_conversation_memory(conv)
        return conv
    return None

# ===========================
# Files & multimodal helpers
# ===========================
def file_mime(name: str) -> str:
    n = name.lower()
    if n.endswith(".png"): return "image/png"
    if n.endswith(".jpg") or n.endswith(".jpeg"): return "image/jpeg"
    if n.endswith(".webp"): return "image/webp"
    if n.endswith(".gif"): return "image/gif"
    if n.endswith(".pdf"): return "application/pdf"
    return "application/octet-stream"

def to_data_url(mime: str, raw: bytes) -> str:
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def messages_to_responses_payload(msgs: List[Dict]) -> List[Dict]:
    """将 messages 转为 Chat Completions 多模态结构。要求 images 为 [{'mime','data'}] 列表。"""
    payload = []
    for m in msgs:
        text = (m.get("content") or "").strip()
        images = m.get("images", [])
        if images:
            content = []
            if text: content.append({"type":"text","text":text})
            for img in images:
                content.append({"type":"image_url","image_url":{"url": to_data_url(img["mime"], img["data"])}})
            payload.append({"role": m["role"], "content": content})
        else:
            payload.append({"role": m["role"], "content": text or ""})
    return payload

# ===========================
# Internet search (DDG)
# ===========================
def setup_internet_search():
    try:
        return DuckDuckGoSearchRun()
    except Exception as e:
        st.warning(f"Internet search setup failed: {e}")
        return None

def search_internet(query: str, search_tool):
    try:
        # 简单退避，降低被限的概率
        time.sleep(random.uniform(1.2, 2.4))
        return search_tool.run(query)
    except Exception as e:
        st.warning(f"Search failed: {e}")
        return None

# ===========================
# Document -> vector store
# ===========================
def setup_document_retrieval(uploaded_files, conversation):
    if not uploaded_files:
        return conversation.get("vectorstore", None)
    try:
        documents = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith('.pdf'):
                tmp = f"temp_{uploaded_file.name}"
                with open(tmp, "wb") as f:
                    f.write(uploaded_file.getvalue())
                docs = PyPDFLoader(tmp).load()
                documents.extend(docs)
                os.remove(tmp)
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            embeddings = utils.configure_embedding_model()
            vectorstore = DocArrayInMemorySearch.from_documents(texts, embeddings)
            conversation["vectorstore"] = vectorstore
            return vectorstore
    except Exception as e:
        st.error(f"Document processing failed: {e}")
    return conversation.get("vectorstore", None)

# ===========================
# Init
# ===========================
initialize_session_state()

# ===========================
# Sidebar: conversation mgmt
# ===========================
with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%); padding:12px; border-radius:10px; text-align:center;">
      <h4 style="color:white; margin:0;">🎓 ChatBOT.EDU</h4>
      <p style="color:rgba(255,255,255,.85); font-size:14px; margin:4px 0 0 0;">Smart Education Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("➕ New Conversation", key="new_conversation", help="Create a new conversation"):
        create_new_conversation()
        st.rerun()

    st.markdown("---")

    if st.session_state.conversations:
        st.markdown("**💬 My Conversations**")
        for conv_id, conv in st.session_state.conversations.items():
            c1, c2 = st.columns([4,1])
            with c1:
                is_active = st.session_state.current_conversation_id == conv_id
                if st.button(conv["name"], key=f"conv_{conv_id}",
                             help=f"Created: {format_datetime(conv.get('created_at'))}",
                             type="primary" if is_active else "secondary"):
                    st.session_state.current_conversation_id = conv_id
                    st.rerun()
            with c2:
                if st.button("🗑️", key=f"delete_{conv_id}", help="Delete conversation"):
                    delete_conversation(conv_id)
                    st.rerun()
    else:
        st.info("No conversations yet, click above to create one.")

    st.markdown("---")

    current_conv = get_current_conversation()
    if current_conv:
        st.markdown("**⚙️ Current Conversation Settings**")
        new_name = st.text_input("Conversation Name", value=current_conv["name"], key="conv_name")
        if new_name != current_conv["name"]:
            current_conv["name"] = new_name
        current_conv["model"] = st.selectbox(
            "🤖 Model",
            ["gpt-5-mini", "gpt-5", "gpt-4o-mini"],
            index=["gpt-5-mini","gpt-5","gpt-4o-mini"].index(current_conv["model"])
                  if current_conv["model"] in ["gpt-5-mini","gpt-5","gpt-4o-mini"] else 0,
            key="model_select"
        )

# ===========================
# Main
# ===========================
current_conv = get_current_conversation()

if not current_conv:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🧠 Context Awareness\n- Remember history\n- Continuous learning\n- Smart management")
    with c2:
        st.markdown("### 🌐 Internet Access\n- Real-time search\n- Latest events\n- Knowledge updates")
    with c3:
        st.markdown("### 📄 Document Analysis\n- PDF processing\n- Image understanding\n- Multimodal")
else:
    # ===== 两栏：左 9 / 右 3 =====
    col_chat, col_tools = st.columns([9, 3], gap="large")

    # ------ 左：历史消息 ------
    with col_chat:
        for msg in current_conv["messages"]:
            imgs = None
            if msg.get("images"):
                # 兼容两种结构：[(name,bytes)] 或 [{'data','mime'}] -> 渲染统一用 (name,bytes)
                if isinstance(msg["images"][0], tuple):
                    imgs = msg["images"]
                else:
                    imgs = [(f"image {i+1}", im["data"]) for i, im in enumerate(msg["images"])]
            render_msg(msg["role"], msg.get("content"), imgs)

    # ------ 右：工具面板（吸顶） ------
    with col_tools:
        st.markdown("<div id='tools-panel'>", unsafe_allow_html=True)

        # PDF 入库
        with st.container(border=True):
            st.markdown("<div class='card'><h4>📎 Documents</h4>", unsafe_allow_html=True)
            uploaded_docs = st.file_uploader(
                "Upload PDFs", type=["pdf"], accept_multiple_files=True, key="pdf_uploader"
            )
            st.caption("PDF 上传后会被切分并建立向量库，用于检索增强。")
            st.markdown("</div>", unsafe_allow_html=True)

        # 图片附件（随下一条消息发送）
        with st.container(border=True):
            st.markdown("<div class='card'><h4>🖼️ Attachments (optional)</h4>", unsafe_allow_html=True)
            imgs = st.file_uploader(
                "Attach images for next message",
                type=["png","jpg","jpeg","webp","gif"],
                accept_multiple_files=True,
                key="img_uploader_right"
            )
            if imgs:
                st.session_state["pending_images"] = [(f.name, f.read()) for f in imgs]
                st.success(f"Will send {len(imgs)} image(s) with your next message.")
            st.caption("选择后会绑定到你**下一条**消息。")
            st.markdown("</div>", unsafe_allow_html=True)

        # 搜索开关
        with st.container(border=True):
            st.markdown("<div class='card'><h4>🌐 Search</h4>", unsafe_allow_html=True)
            current_conv["enable_internet"] = st.toggle(
                "Enable internet search",
                value=current_conv.get("enable_internet", False)
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # /tools-panel

        # 处理 PDF -> 向量库
        if uploaded_docs:
            st.session_state["_pending_docs"] = uploaded_docs

    # 右栏选择的 PDF 文档 -> 建立向量库
    if st.session_state.get("_pending_docs"):
        with st.spinner("Processing documents..."):
            vectorstore = setup_document_retrieval(st.session_state["_pending_docs"], current_conv)
            if vectorstore:
                current_conv["vectorstore"] = vectorstore
                st.success(f"✅ Processed {len([f for f in st.session_state['_pending_docs'] if f.name.lower().endswith('.pdf')])} PDF(s)")
            else:
                st.error("❌ Document processing failed")
        st.session_state["_pending_docs"] = None

    # ------ 输入：原生 chat_input ------
    prompt = st.chat_input("Ask anything…")
    if prompt:
        # 取出待发送的图片（来自右栏附件卡片）
        pending_imgs = st.session_state.pop("pending_images", [])
        # 渲染 & 保存 user
        render_msg("user", prompt, pending_imgs)
        # 存储时将图片转换为 {'mime','data'} 结构，便于多模态 payload
        img_struct = [{"mime": file_mime(fn), "data": raw} for fn, raw in pending_imgs]
        current_conv["messages"].append({"role":"user","content":prompt,"images":img_struct})
        utils.save_message_to_mongodb(current_conv["id"], "user", prompt, pending_imgs)

        # 组 payload
        msgs = [{"role":m["role"], "content":m["content"], "images":m.get("images", [])}
                for m in current_conv["messages"] if m["role"] in ["user","assistant"]]
        if img_struct and current_conv.get("enable_images", True):
            messages_payload = messages_to_responses_payload(msgs)
        else:
            messages_payload = [{"role":m["role"], "content":m["content"]} for m in msgs]

        # 联网检索增强
        if current_conv.get("enable_internet", False):
            tool = setup_internet_search()
            if tool:
                with st.spinner("🔍 Searching..."):
                    sr = search_internet(prompt, tool)
                if sr:
                    messages_payload.append({"role":"user","content":f"Internet search results for '{prompt}':\n{sr}"})

        # 调用模型
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

        # 首条后摘要改名
        if sum(1 for m in current_conv["messages"] if m["role"]=="user") == 1:
            try:
                summary = utils.generate_conversation_summary(current_conv["messages"])
                current_conv["name"] = summary
                current_conv["updated_at"] = datetime.now().isoformat()
                ok = utils.update_conversation_in_mongodb(current_conv["id"],
                                                          {"name": summary, "updated_at": current_conv["updated_at"]})
                if not ok:
                    utils.save_conversations_to_file(st.session_state.conversations)
            except Exception:
                pass