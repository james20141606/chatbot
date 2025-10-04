import os, io, base64
import utils
import streamlit as st
from streaming import StreamHandler, GPTStreamHandler
from typing import List, Dict, Tuple
from datetime import datetime
import uuid

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.retrievers import TavilySearchAPIRetrieval  # Temporarily commented out due to import path issues
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(
    page_title="ChatBOT.EDU - Intelligent Education Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Minimal, chic CSS for bubbles & cards ----
st.markdown("""
<style>
/* 两栏间距更自然 */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
/* 卡片外观 */
.card {border:1px solid rgba(148,163,255,.2); border-radius:14px; padding:14px 14px; margin-bottom:12px;
       background:rgba(18,20,30,.6)}
.card h4 {margin:0 0 8px 0}
.small {font-size:13px; color:#bfc3ff}
hr.sep {border:none; border-top:1px solid rgba(148,163,255,.15); margin:10px 0}

/* 气泡 */
.bubble{border-radius:14px; padding:12px 14px; margin:6px 0; display:inline-block; max-width:100%}
.bubble.user{background:#1f2a44}
.bubble.assistant{background:#22212b; border:1px solid rgba(148,163,255,.2)}
/* 图片网格标题 */
.figcap{font-size:12px; color:#a9a9b2; text-align:center; margin-top:4px}
</style>
""", unsafe_allow_html=True)

def render_msg(role, text=None, images=None):
    with st.chat_message(role, avatar="👤" if role=="user" else "🤖"):
        if images:
            cols = st.columns(min(4, len(images)))
            for i,(fn,raw) in enumerate(images):
                with cols[i % len(cols)]:
                    st.image(io.BytesIO(raw), use_column_width=True)
                    st.markdown(f"<div class='figcap'>{fn}</div>", unsafe_allow_html=True)
        if text:
            cls = "user" if role=="user" else "assistant"
            st.markdown(f"<div class='bubble {cls}'>{text}</div>", unsafe_allow_html=True)

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sidebar-header h4 {
        color: white;
        margin: 0;
        font-weight: 600;
    }
    .conversation-item {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.2rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .conversation-item:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-1px);
    }
    .conversation-item.active {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-color: #667eea;
    }
    .new-conversation-btn {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        width: 100%;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    .new-conversation-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
    }
    .delete-btn {
        background: #ff4444;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        font-size: 0.8rem;
        cursor: pointer;
        float: right;
        margin-top: -0.2rem;
    }
    .delete-btn:hover {
        background: #cc0000;
    }
</style>
""", unsafe_allow_html=True)

# ============== Helper Functions ==============
def format_datetime(dt_str):
    """Format datetime string for display"""
    try:
        if isinstance(dt_str, str):
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
        else:
            return 'Unknown'
    except:
        return 'Unknown'

# ============== Conversation Management ==============
def initialize_session_state():
    """Initialize session state for conversations with persistence"""
    if "conversations" not in st.session_state:
        # Try to load from MongoDB first, then fallback to JSON file
        mongodb_conversations = utils.load_conversations_from_mongodb()
        if mongodb_conversations:
            st.session_state.conversations = mongodb_conversations
            print("✅ Loaded conversations from MongoDB")
        else:
            st.session_state.conversations = utils.load_conversations_from_file()
            print("📁 Loaded conversations from JSON file")
    
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    if "conversation_counter" not in st.session_state:
        # Find the highest conversation counter from existing conversations
        max_counter = 0
        for conv_data in st.session_state.conversations.values():
            if "counter" in conv_data:
                max_counter = max(max_counter, conv_data["counter"])
        st.session_state.conversation_counter = max_counter

def create_new_conversation():
    """Create a new conversation"""
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
    
    # Save to MongoDB first, then fallback to JSON file
    mongodb_id = utils.save_conversation_to_mongodb(st.session_state.conversations[conv_id])
    if mongodb_id:
        print(f"✅ Saved conversation to MongoDB with ID: {mongodb_id}")
    else:
        utils.save_conversations_to_file(st.session_state.conversations)
        print("📁 Saved conversation to JSON file")
    
    return conv_id

def delete_conversation(conv_id):
    """Delete a conversation"""
    if conv_id in st.session_state.conversations:
        del st.session_state.conversations[conv_id]
        if st.session_state.current_conversation_id == conv_id:
            if st.session_state.conversations:
                st.session_state.current_conversation_id = list(st.session_state.conversations.keys())[0]
            else:
                st.session_state.current_conversation_id = None
        
        # Save to persistent storage
        utils.save_conversations_to_file(st.session_state.conversations)

def restore_conversation_memory(conversation):
    """Restore conversation memory from messages if not present"""
    if "memory" not in conversation or conversation["memory"] is None:
        conversation["memory"] = ConversationBufferMemory(return_messages=True)
        # Restore messages to memory
        for msg in conversation.get("messages", []):
            if msg["role"] == "user":
                conversation["memory"].save_context(
                    {"input": msg["content"]}, 
                    {"output": ""}
                )
            elif msg["role"] == "assistant":
                # Find the previous user message to pair with this assistant response
                messages = conversation["messages"]
                for i, prev_msg in enumerate(messages):
                    if prev_msg == msg:
                        # Find the most recent user message before this assistant message
                        for j in range(i-1, -1, -1):
                            if messages[j]["role"] == "user":
                                conversation["memory"].save_context(
                                    {"input": messages[j]["content"]}, 
                                    {"output": msg["content"]}
                                )
                                break
                        break

def get_current_conversation():
    """Get current conversation"""
    if st.session_state.current_conversation_id and st.session_state.current_conversation_id in st.session_state.conversations:
        conversation = st.session_state.conversations[st.session_state.current_conversation_id]
        # Ensure memory is restored for loaded conversations
        restore_conversation_memory(conversation)
        return conversation
    return None

# ============== Image Processing Functions ==============
MAX_IMAGES_PER_TURN = 4
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "pdf"]

def file_mime(name: str) -> str:
    n = name.lower()
    if n.endswith(".png"): return "image/png"
    if n.endswith(".jpg") or n.endswith(".jpeg"): return "image/jpeg"
    if n.endswith(".webp"): return "image/webp"
    if n.endswith(".pdf"): return "application/pdf"
    return "application/octet-stream"

def to_data_url(mime: str, raw: bytes) -> str:
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def messages_to_responses_payload(msgs: List[Dict]) -> List[Dict]:
    """Convert to Chat Completions API messages structure: supports text and images multimodal messages"""
    payload = []
    for m in msgs:
        text = (m.get("content") or "").strip()
        images = m.get("images", [])
        
        if images:
            # If there are images, use multimodal format
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
            # Text-only message
            payload.append({"role": m["role"], "content": text or ""})
    return payload

# ============== Internet Search Functions ==============
def setup_internet_search():
    """Setup internet search functionality"""
    try:
        # Use DuckDuckGo as the main search tool
        search = DuckDuckGoSearchRun()
        return search
    except Exception as e:
        st.warning(f"Internet search setup failed: {e}")
        return None

def search_internet(query: str, search_tool):
    """Execute internet search"""
    try:
        # Use DuckDuckGo search
        result = search_tool.run(query)
        return result
    except Exception as e:
        st.warning(f"Search failed: {e}")
        return None

# ============== Document Processing Functions ==============
def setup_document_retrieval(uploaded_files, conversation):
    """Setup document retrieval functionality"""
    if not uploaded_files:
        return conversation.get("vectorstore", None)
    
    try:
        # Process PDF files
        documents = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith('.pdf'):
                # Save PDF file
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Load PDF
                loader = PyPDFLoader(f"temp_{uploaded_file.name}")
                docs = loader.load()
                documents.extend(docs)
                
                # Clean up temporary file
                os.remove(f"temp_{uploaded_file.name}")
        
        if documents:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            
            # Create vector store
            embeddings = utils.configure_embedding_model()
            vectorstore = DocArrayInMemorySearch.from_documents(texts, embeddings)
            
            # Save to conversation
            conversation["vectorstore"] = vectorstore
            return vectorstore
    except Exception as e:
        st.error(f"Document processing failed: {e}")
    
    return conversation.get("vectorstore", None)

# ============== Initialization ==============
initialize_session_state()

# Main title
# st.markdown("""
# <div class="main-header">
#     <h1>🎓 ChatBOT.EDU</h1>
#     <p>Intelligent Education Assistant with Multi-Conversation Support</p>
# </div>
# """, unsafe_allow_html=True)

# ============== Sidebar - Conversation Management ==============
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h4>🎓 ChatBOT.EDU</h4>
        <p style="color: rgba(255,255,255,0.8); font-size: 14px; margin: 4px 0 0 0;">Smart Education Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # New conversation button
    if st.button("➕ New Conversation", key="new_conversation", help="Create a new conversation"):
        conv_id = create_new_conversation()
        st.rerun()
    
    st.markdown("---")
    
    # Conversation list
    if st.session_state.conversations:
        st.markdown("**💬 My Conversations**")
        
        for conv_id, conv in st.session_state.conversations.items():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Conversation name button
                is_active = st.session_state.current_conversation_id == conv_id
                if st.button(
                    conv["name"], 
                    key=f"conv_{conv_id}",
                    help=f"Created: {format_datetime(conv.get('created_at'))}",
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.current_conversation_id = conv_id
                    st.rerun()
            
            with col2:
                # Delete button
                if st.button("🗑️", key=f"delete_{conv_id}", help="Delete conversation"):
                    delete_conversation(conv_id)
                    st.rerun()
    
    else:
        st.info("No conversations yet, click the button above to create a new conversation")
    
    st.markdown("---")
    
    # Current conversation settings
    current_conv = get_current_conversation()
    if current_conv:
        st.markdown("**⚙️ Current Conversation Settings**")
        
        # Conversation name editing
        new_name = st.text_input("Conversation Name", value=current_conv["name"], key="conv_name")
        if new_name != current_conv["name"]:
            current_conv["name"] = new_name
        
        # Model selection
        current_conv["model"] = st.selectbox(
            "🤖 Model", 
            ["gpt-5-mini", "gpt-5", "gpt-4o-mini"],
            index=["gpt-5-mini", "gpt-5", "gpt-4o-mini"].index(current_conv["model"]) if current_conv["model"] in ["gpt-5-mini", "gpt-5", "gpt-4o-mini"] else 0,
            key="model_select"
        )
        
        

# ============== Main Chat Interface ==============
current_conv = get_current_conversation()

if not current_conv:
    # Welcome interface when no conversations exist
    # st.markdown("""
    # <div style="text-align: center; padding: 4rem 2rem; color: #666;">
    #     <h2>🎓 Welcome to ChatBOT.EDU</h2>
    #     <p style="font-size: 1.2rem; margin: 1rem 0;">Intelligent Education Assistant with Multi-Conversation Support</p>
    #     <p>Click the "➕ New Conversation" button on the left to start your first conversation</p>
    # </div>
    # """, unsafe_allow_html=True)
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🧠 Context Awareness
        - Remember conversation history
        - Continuous learning experience
        - Smart conversation management
        """)
    
    with col2:
        st.markdown("""
        ### 🌐 Internet Access
        - Real-time information search
        - Latest events query
        - Knowledge updates
        """)
    
    with col3:
        st.markdown("""
        ### 📄 Document Analysis
        - PDF document processing
        - Image content understanding
        - Multimodal interaction
        """)

else:
    # ===== 布局：左 7 / 右 5，右侧做工具卡片 =====
    col_chat, col_tools = st.columns([7, 5], gap="large")

    # ---------- 左：历史消息 ----------
    with col_chat:
        for msg in current_conv["messages"]:
            imgs = None
            if msg.get("images"):
                # 兼容你现有的两种结构：[(name,bytes)] 或 [{"data":..}]
                if isinstance(msg["images"][0], tuple):
                    imgs = msg["images"]
                else:
                    imgs = [(f"image {i+1}", im["data"]) for i,im in enumerate(msg["images"])]
            render_msg(msg["role"], msg.get("content"), imgs)

    # ---------- 右：工具卡片（设置、上传、搜索等） ----------
    with col_tools:
        with st.container(border=True):
            st.markdown("<div class='card'><h4>⚙️ Conversation</h4>", unsafe_allow_html=True)
            new_name = st.text_input("Name", value=current_conv["name"], label_visibility="collapsed")
            if new_name != current_conv["name"]:
                current_conv["name"] = new_name
            current_conv["model"] = st.selectbox("Model", ["gpt-5-mini","gpt-5","gpt-4o-mini"],
                                                 index=["gpt-5-mini","gpt-5","gpt-4o-mini"].index(current_conv["model"]) if current_conv["model"] in ["gpt-5-mini","gpt-5","gpt-4o-mini"] else 0)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div class='card'><h4>📎 Documents & Images</h4>", unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Upload files", type=["pdf","png","jpg","jpeg","webp"],
                accept_multiple_files=True
            )
            st.caption("PDFs will进入向量库；图片直接随消息发送。")
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div class='card'><h4>🌐 Search</h4>", unsafe_allow_html=True)
            current_conv["enable_internet"] = st.toggle("Enable internet search", value=current_conv.get("enable_internet", False))
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- 底部输入区：一行并排 ----------
    with st.container():
        st.markdown("<hr class='sep'/>", unsafe_allow_html=True)
        i1, i2, i3, i4 = st.columns([8, 2, 2, 1])
        with i1:
            user_text_input = st.text_input("Ask Anything", key="composer_text", label_visibility="collapsed",
                                            placeholder="Ask Anything")
        with i2:
            # 一个轻量"只图片"上传（与右栏文档上传互不冲突）
            img_files = st.file_uploader("Add images", type=["png","jpg","jpeg","webp"],
                                         accept_multiple_files=True, key="img_uploader", label_visibility="collapsed")
        with i3:
            internet_search_active = current_conv.get("enable_internet", False)
            st.button(("Search ON" if internet_search_active else "Search OFF"), key="noop", disabled=True)
        with i4:
            send = st.button("➤", use_container_width=True, type="primary")

    # ---------- 处理上传（右栏 PDFs -> 向量库；底部图片 -> 消息） ----------
    files_to_save = []
    if img_files and current_conv.get("enable_images", True):
        files_to_save = [(f.name, f.read()) for f in img_files]  # 仅图片跟随消息

    # 右栏的 uploaded_files 里可能有 PDF，需要入库（见 C 段）
    if uploaded_files:
        st.session_state["_pending_docs"] = uploaded_files  # 暂存，发送时或立刻处理

    # 右栏选择的 PDF 文档 -> 建立向量库
    if st.session_state.get("_pending_docs"):
        with st.spinner("Processing documents..."):
            vectorstore = setup_document_retrieval(st.session_state["_pending_docs"], current_conv)  # ← 修复传参
            if vectorstore:
                current_conv["vectorstore"] = vectorstore
                st.success(f"✅ Processed {len([f for f in st.session_state['_pending_docs'] if f.name.lower().endswith('.pdf')])} PDF(s)")
            else:
                st.error("❌ Document processing failed")
        st.session_state["_pending_docs"] = None

    # Process input
    if send and user_text_input:
        user_text = user_text_input
        files_to_save: List[Tuple[str, bytes]] = []
        
        # Check if there are uploaded files to process as images
        if uploaded_files and current_conv["enable_images"]:
            # Filter for image files
            image_files = [f for f in uploaded_files if f.type.startswith('image/')]
            if image_files:
                files_to_save = [(f.name, f.read()) for f in image_files]

        # 先把图片/文本渲染一版
        render_msg("user", user_text, files_to_save)

        # Add user message to conversation history
        current_conv["messages"].append({
            "role": "user",
            "content": user_text,
            "images": files_to_save
        })
        
        # Save user message to MongoDB
        utils.save_message_to_mongodb(
            current_conv["id"], 
            "user", 
            user_text, 
            files_to_save
        )
        
        # Prepare messages for GPT API
        messages = []
        for msg in current_conv["messages"]:
            if msg["role"] in ["user", "assistant"]:
                msg_content = {
                    "role": msg["role"], 
                    "content": msg["content"],
                    "images": msg.get("images", [])
                }
                messages.append(msg_content)
        
        # If there are images, convert to multimodal format
        if files_to_save and current_conv["enable_images"]:
            messages_payload = messages_to_responses_payload(messages)
        else:
            messages_payload = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        
        # Internet search enhancement
        if current_conv["enable_internet"] and user_text:
            search_tool = setup_internet_search()
            if search_tool:
                with st.spinner("🔍 Searching internet..."):
                    search_results = search_internet(user_text, search_tool)
                    if search_results:
                        # Add search results to messages
                        search_context = f"Internet search results for '{user_text}':\n{search_results}\n\nPlease use this information to provide a comprehensive answer."
                        messages_payload.append({"role": "user", "content": search_context})

        # 调用模型
        with st.chat_message("assistant", avatar="🤖"):
            try:
                stream = utils.call_gpt_api(messages_payload, model=current_conv["model"], stream=True)
                if stream:
                    gpt_handler = GPTStreamHandler(st.empty())
                    resp = gpt_handler.handle_stream(stream)
                else:
                    resp = "Sorry, I'm having trouble connecting to the AI service. Please try again."
            except Exception as e:
                st.error(f"{e}")
                resp = "Sorry, I encountered an error while processing your request. Please try again."

            st.markdown(f"<div class='bubble assistant'>{resp}</div>", unsafe_allow_html=True)
            current_conv["messages"].append({"role":"assistant","content":resp,"images":[]})

        # 首条后自动摘要改名
        if sum(1 for m in current_conv["messages"] if m["role"]=="user") == 1:
            try:
                summary = utils.generate_conversation_summary(current_conv["messages"])
                current_conv["name"] = summary
                current_conv["updated_at"] = datetime.now().isoformat()
                
                # Save to MongoDB first, then fallback to JSON file
                mongodb_success = utils.update_conversation_in_mongodb(
                    current_conv["id"], 
                    {"name": summary, "updated_at": current_conv["updated_at"]}
                )
                if not mongodb_success:
                    utils.save_conversations_to_file(st.session_state.conversations)
            except Exception:
                pass
