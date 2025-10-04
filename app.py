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
from langchain_community.retrievers import TavilySearchAPIRetrieval
from langchain.tools import DuckDuckGoSearchRun

st.set_page_config(
    page_title="ChatBOT.EDU - 智能教育助手",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
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

# ============== 会话管理 ==============
def initialize_session_state():
    """初始化会话状态"""
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    if "conversation_counter" not in st.session_state:
        st.session_state.conversation_counter = 0

def create_new_conversation():
    """创建新的对话"""
    st.session_state.conversation_counter += 1
    conv_id = f"conv_{st.session_state.conversation_counter}"
    
    st.session_state.conversations[conv_id] = {
        "id": conv_id,
        "name": f"对话 {st.session_state.conversation_counter}",
        "created_at": datetime.now(),
        "messages": [],
        "memory": ConversationBufferMemory(return_messages=True),
        "vectorstore": None,
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "enable_internet": True,
        "enable_documents": True,
        "enable_images": True
    }
    
    st.session_state.current_conversation_id = conv_id
    return conv_id

def delete_conversation(conv_id):
    """删除对话"""
    if conv_id in st.session_state.conversations:
        del st.session_state.conversations[conv_id]
        if st.session_state.current_conversation_id == conv_id:
            if st.session_state.conversations:
                st.session_state.current_conversation_id = list(st.session_state.conversations.keys())[0]
            else:
                st.session_state.current_conversation_id = None

def get_current_conversation():
    """获取当前对话"""
    if st.session_state.current_conversation_id and st.session_state.current_conversation_id in st.session_state.conversations:
        return st.session_state.conversations[st.session_state.current_conversation_id]
    return None

# ============== 图像处理功能 ==============
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
    """转换为 Chat Completions API 的 messages 结构：支持文本和图片的多模态消息"""
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

# ============== 互联网搜索功能 ==============
def setup_internet_search():
    """设置互联网搜索功能"""
    try:
        # 尝试使用Tavily API
        if os.getenv("TAVILY_API_KEY"):
            retriever = TavilySearchAPIRetrieval(api_key=os.getenv("TAVILY_API_KEY"))
            return retriever
        else:
            # 使用DuckDuckGo作为备选
            search = DuckDuckGoSearchRun()
            return search
    except Exception as e:
        st.warning(f"Internet search setup failed: {e}")
        return None

def search_internet(query: str, search_tool):
    """执行互联网搜索"""
    try:
        if hasattr(search_tool, 'get_relevant_documents'):
            # Tavily API
            docs = search_tool.get_relevant_documents(query)
            return "\n".join([doc.page_content for doc in docs[:3]])
        else:
            # DuckDuckGo
            result = search_tool.run(query)
            return result
    except Exception as e:
        st.warning(f"Search failed: {e}")
        return None

# ============== 文档处理功能 ==============
def setup_document_retrieval(uploaded_files, conversation):
    """设置文档检索功能"""
    if not uploaded_files:
        return conversation.get("vectorstore", None)
    
    try:
        # 处理PDF文件
        documents = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith('.pdf'):
                # 保存PDF文件
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # 加载PDF
                loader = PyPDFLoader(f"temp_{uploaded_file.name}")
                docs = loader.load()
                documents.extend(docs)
                
                # 清理临时文件
                os.remove(f"temp_{uploaded_file.name}")
        
        if documents:
            # 分割文档
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            
            # 创建向量存储
            embeddings = utils.configure_embedding_model()
            vectorstore = DocArrayInMemorySearch.from_documents(texts, embeddings)
            
            # 保存到对话中
            conversation["vectorstore"] = vectorstore
            return vectorstore
    except Exception as e:
        st.error(f"Document processing failed: {e}")
    
    return conversation.get("vectorstore", None)

# ============== 初始化 ==============
initialize_session_state()

# 主标题
# st.markdown("""
# <div class="main-header">
#     <h1>🎓 ChatBOT.EDU</h1>
#     <p>Intelligent Education Assistant with Multi-Conversation Support</p>
# </div>
# """, unsafe_allow_html=True)

# ============== 侧边栏 - 对话管理 ==============
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h4>🎓 ChatBOT.EDU</h4>
        <p style="color: rgba(255,255,255,0.8); font-size: 14px; margin: 4px 0 0 0;">Smart Education Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 新建对话按钮
    if st.button("➕ 新建对话", key="new_conversation", help="创建新的对话"):
        conv_id = create_new_conversation()
        st.rerun()
    
    st.markdown("---")
    
    # 对话列表
    if st.session_state.conversations:
        st.markdown("**💬 我的对话**")
        
        for conv_id, conv in st.session_state.conversations.items():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # 对话名称按钮
                is_active = st.session_state.current_conversation_id == conv_id
                if st.button(
                    conv["name"], 
                    key=f"conv_{conv_id}",
                    help=f"创建时间: {conv['created_at'].strftime('%Y-%m-%d %H:%M')}",
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.current_conversation_id = conv_id
                    st.rerun()
            
            with col2:
                # 删除按钮
                if st.button("🗑️", key=f"delete_{conv_id}", help="删除对话"):
                    delete_conversation(conv_id)
                    st.rerun()
    
    else:
        st.info("暂无对话，点击上方按钮创建新对话")
    
    st.markdown("---")
    
    # 当前对话设置
    current_conv = get_current_conversation()
    if current_conv:
        st.markdown("**⚙️ 当前对话设置**")
        
        # 对话名称编辑
        new_name = st.text_input("对话名称", value=current_conv["name"], key="conv_name")
        if new_name != current_conv["name"]:
            current_conv["name"] = new_name
        
        # 模型选择
        current_conv["model"] = st.selectbox(
            "🤖 模型", 
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"].index(current_conv["model"]),
            key="model_select"
        )
        
        # 温度设置
        current_conv["temperature"] = st.slider(
            "🌡️ 创造力", 
            0.0, 1.5, current_conv["temperature"], 0.1,
            key="temp_select"
        )
        
        # 功能开关
        st.markdown("**🔧 功能开关**")
        current_conv["enable_internet"] = st.checkbox("🌐 互联网搜索", value=current_conv["enable_internet"], key="internet_toggle")
        current_conv["enable_documents"] = st.checkbox("📄 文档分析", value=current_conv["enable_documents"], key="docs_toggle")
        current_conv["enable_images"] = st.checkbox("🖼️ 图像分析", value=current_conv["enable_images"], key="images_toggle")

# ============== 主聊天界面 ==============
current_conv = get_current_conversation()

if not current_conv:
    # 没有对话时显示欢迎界面
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #666;">
        <h2>🎓 欢迎使用 ChatBOT.EDU</h2>
        <p style="font-size: 1.2rem; margin: 1rem 0;">智能教育助手，支持多对话管理</p>
        <p>点击左侧 "➕ 新建对话" 按钮开始您的第一个对话</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 功能展示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🧠 上下文感知
        - 记住对话历史
        - 连续学习体验
        - 智能对话管理
        """)
    
    with col2:
        st.markdown("""
        ### 🌐 互联网访问
        - 实时信息搜索
        - 最新事件查询
        - 知识更新
        """)
    
    with col3:
        st.markdown("""
        ### 📄 文档分析
        - PDF文档处理
        - 图像内容理解
        - 多模态交互
        """)

else:
    # 显示当前对话
    st.markdown(f"### 💬 {current_conv['name']}")
    
    # 显示对话历史
    for msg in current_conv["messages"]:
        with st.chat_message(msg["role"]):
            if msg.get("images"):
                cols = st.columns(min(4, len(msg["images"])))
                for i, img in enumerate(msg["images"]):
                    with cols[i % len(cols)]:
                        st.image(io.BytesIO(img["data"]), caption=f"image {i+1}", use_column_width=True)
            if msg.get("content"):
                st.markdown(msg["content"])
    
    # 文档上传区域
    if current_conv["enable_documents"]:
        with st.expander("📁 上传文档", expanded=False):
            uploaded_files = st.file_uploader(
                "选择PDF文件", 
                type=["pdf"], 
                accept_multiple_files=True,
                key="doc_uploader"
            )
            
            if uploaded_files:
                with st.spinner("处理文档中..."):
                    vectorstore = setup_document_retrieval(uploaded_files, current_conv)
                    if vectorstore:
                        st.success(f"✅ 成功处理 {len(uploaded_files)} 个文档")
                    else:
                        st.error("❌ 文档处理失败")
    
    # 自定义输入框
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
            uploaded_images = st.file_uploader(
                " ", type=IMAGE_TYPES, accept_multiple_files=True,
                key="uploader", label_visibility="collapsed"
            )

        with c2:
            user_text_input = st.text_input(
                "Ask Anything", key="composer_text", label_visibility="collapsed",
                placeholder="Ask Anything - I can analyze documents, search the web, and understand images!"
            )

        with c3:
            send = st.form_submit_button("➤", use_container_width=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    # 处理输入
    if send and (user_text_input or uploaded_images):
        user_text = user_text_input or ""
        files_to_save: List[Tuple[str, bytes]] = []

        # 显示用户消息
        with st.chat_message("user"):
            if uploaded_images and current_conv["enable_images"]:
                cols = st.columns(min(4, len(uploaded_images)))
                for i, f in enumerate(uploaded_images):
                    raw = f.read()
                    mime = file_mime(f.name)
                    files_to_save.append((mime, raw))
                    with cols[i % len(cols)]:
                        st.image(io.BytesIO(raw), caption=f.name, use_column_width=True)
            if user_text:
                st.markdown(user_text)

        # 添加用户消息到对话历史
        current_conv["messages"].append({
            "role": "user",
            "content": user_text,
            "images": files_to_save
        })
        
        # 准备消息用于GPT API
        messages = []
        for msg in current_conv["messages"]:
            if msg["role"] in ["user", "assistant"]:
                msg_content = {
                    "role": msg["role"], 
                    "content": msg["content"],
                    "images": msg.get("images", [])
                }
                messages.append(msg_content)
        
        # 如果有图片，转换为多模态格式
        if files_to_save and current_conv["enable_images"]:
            messages_payload = messages_to_responses_payload(messages)
        else:
            messages_payload = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        
        # 互联网搜索增强
        if current_conv["enable_internet"] and user_text:
            search_tool = setup_internet_search()
            if search_tool:
                with st.spinner("🔍 搜索互联网..."):
                    search_results = search_internet(user_text, search_tool)
                    if search_results:
                        # 将搜索结果添加到消息中
                        search_context = f"Internet search results for '{user_text}':\n{search_results}\n\nPlease use this information to provide a comprehensive answer."
                        messages_payload.append({"role": "user", "content": search_context})

        # 生成回复
        with st.chat_message("assistant"):
            try:
                # 使用GPT API直接调用
                stream = utils.call_gpt_api(
                    messages_payload, 
                    model=current_conv["model"], 
                    temperature=current_conv["temperature"], 
                    stream=True
                )
                if stream:
                    gpt_handler = GPTStreamHandler(st.empty())
                    response = gpt_handler.handle_stream(stream)
                    current_conv["messages"].append({
                        "role": "assistant",
                        "content": response,
                        "images": []
                    })
                else:
                    # 回退到LangChain
                    from langchain.chains import ConversationChain
                    llm = utils.configure_llm()
                    chain = ConversationChain(llm=llm, memory=current_conv["memory"], verbose=False)
                    st_cb = StreamHandler(st.empty())
                    result = chain.invoke(
                        {"input": user_text},
                        {"callbacks": [st_cb]}
                    )
                    response = result["response"]
                    current_conv["messages"].append({
                        "role": "assistant",
                        "content": response,
                        "images": []
                    })
            except Exception as e:
                st.error(f"Error: {e}")
                # 最终回退
                from langchain.chains import ConversationChain
                llm = utils.configure_llm()
                chain = ConversationChain(llm=llm, memory=current_conv["memory"], verbose=False)
                st_cb = StreamHandler(st.empty())
                result = chain.invoke(
                    {"input": user_text},
                    {"callbacks": [st_cb]}
                )
                response = result["response"]
                current_conv["messages"].append({
                    "role": "assistant",
                    "content": response,
                    "images": []
                })
