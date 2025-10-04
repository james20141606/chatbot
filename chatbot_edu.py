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
    # Display current conversation
    # st.markdown(f"### 💬 {current_conv['name']}")
    
    # Display conversation history
    for msg in current_conv["messages"]:
        with st.chat_message(msg["role"]):
            if msg.get("images"):
                cols = st.columns(min(4, len(msg["images"])))
                for i, img in enumerate(msg["images"]):
                    with cols[i % len(cols)]:
                        st.image(io.BytesIO(img["data"]), caption=f"image {i+1}", use_column_width=True)
            if msg.get("content"):
                st.markdown(msg["content"])
    
    
    # Custom input box
    st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Wait for elements to be ready
        setTimeout(function() {
            // Find the text input by placeholder or other attributes
            const textInputs = document.querySelectorAll('input[type="text"]');
            let targetInput = null;
            
            for (let input of textInputs) {
                if (input.placeholder && input.placeholder.includes('Ask Anything')) {
                    targetInput = input;
                    break;
                }
            }
            
            if (targetInput) {
                targetInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        
                        // Try multiple ways to find and click the send button
                        let sendButton = document.querySelector('button[kind="secondary"]');
                        if (!sendButton) {
                            sendButton = document.querySelector('button:has(span:contains("➤"))');
                        }
                        if (!sendButton) {
                            // Look for button with arrow character
                            const buttons = document.querySelectorAll('button');
                            for (let btn of buttons) {
                                if (btn.textContent.includes('➤') || btn.innerHTML.includes('➤')) {
                                    sendButton = btn;
                                    break;
                                }
                            }
                        }
                        
                        if (sendButton) {
                            sendButton.click();
                        }
                    }
                });
            }
        }, 1000);
    });
    </script>
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

    # Create custom input layout with buttons
    col1, col2, col3, col4, col5 = st.columns([0.01, 0.75, 0.08, 0.08, 0.08])
    
    with col2:
        user_text_input = st.text_input(
            "Ask Anything", key="composer_text", label_visibility="collapsed",
            placeholder="Ask Anything"
        )
    
    with col3:
        # File upload button
        uploaded_files = st.file_uploader(
            "📁", type=["pdf", "png", "jpg", "jpeg", "gif", "bmp", "webp"], 
            accept_multiple_files=True, key="file_uploader", label_visibility="collapsed",
            help="Upload PDF documents or images"
        )
    
    with col4:
        # Internet search toggle button
        internet_search_active = current_conv.get("enable_internet", False)
        
        # Custom styling for the buttons
        button_style = """
        <style>
        /* Internet search toggle button styling */
        .search-button-active {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
            color: white !important;
            border: 2px solid #0f3460 !important;
            box-shadow: 0 4px 12px rgba(15, 52, 96, 0.4) !important;
            transform: translateY(-1px) !important;
        }
        .search-button-inactive {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: 2px solid transparent !important;
        }
        
        /* File uploader button styling */
        .stFileUploader > div > div > button {
            height: 38px !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: 2px solid transparent !important;
            border-radius: 8px !important;
            font-size: 16px !important;
        }
        
        .stFileUploader > div > div > button:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
            border-color: #667eea !important;
        }
        
        /* Send button styling */
        div[data-testid="column"] button[kind="secondary"] {
            height: 38px !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: 2px solid transparent !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            font-weight: bold !important;
        }
        
        div[data-testid="column"] button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
            border-color: #667eea !important;
        }
        
        /* Internet search button styling */
        div[data-testid="column"] button[kind="secondary"][title*="Toggle internet search"] {
            height: 38px !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            font-weight: bold !important;
        }
        </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)
        
        # Apply active styling based on state
        button_text = "🌐"
        button_help = f"Toggle internet search ({'ON' if internet_search_active else 'OFF'})"
        
        # Use different button styling based on active state
        if internet_search_active:
            # Active state - dark styling
            active_button_style = """
            <style>
            div[data-testid="column"] button[kind="secondary"][title*="Toggle internet search"] {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
                color: white !important;
                border: 2px solid #0f3460 !important;
                box-shadow: 0 4px 12px rgba(15, 52, 96, 0.4) !important;
                transform: translateY(-1px) !important;
            }
            </style>
            """
            st.markdown(active_button_style, unsafe_allow_html=True)
        
        if st.button(button_text, use_container_width=True, key="search_toggle", 
                    help=button_help):
            current_conv["enable_internet"] = not current_conv.get("enable_internet", False)
            st.rerun()
    
    with col5:
        # Send button
        send = st.button("➤", use_container_width=True, key="send_button",
                        help="Send message")
    
    # Handle file uploads separately
    if uploaded_files:
        if current_conv["enable_documents"]:
            # Process documents
            with st.spinner("Processing documents..."):
                vectorstore = setup_document_retrieval(uploaded_files)
                if vectorstore:
                    current_conv["vectorstore"] = vectorstore
                    st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)")
                else:
                    st.error("❌ Document processing failed")
        else:
            st.warning("⚠️ Document analysis is disabled. Enable it in settings to process documents.")

    st.markdown('</div></div>', unsafe_allow_html=True)

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

        # Display user message
        with st.chat_message("user"):
            if files_to_save and current_conv["enable_images"]:
                cols = st.columns(min(4, len(files_to_save)))
                for i, (filename, raw_data) in enumerate(files_to_save):
                    with cols[i % len(cols)]:
                        st.image(io.BytesIO(raw_data), caption=filename, use_column_width=True)
            if user_text:
                st.markdown(user_text)

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

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Use GPT API directly
                stream = utils.call_gpt_api(
                    messages_payload, 
                    model=current_conv["model"], 
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
                    
                    # Save assistant message to MongoDB
                    utils.save_message_to_mongodb(
                        current_conv["id"], 
                        "assistant", 
                        response
                    )
                else:
                    # Fallback message if GPT API fails
                    response = "Sorry, I'm having trouble connecting to the AI service. Please try again."
                    current_conv["messages"].append({
                        "role": "assistant",
                        "content": response,
                        "images": []
                    })
                    
                    # Save assistant message to MongoDB
                    utils.save_message_to_mongodb(
                        current_conv["id"], 
                        "assistant", 
                        response
                    )
                
                # Generate conversation summary after first user message
                user_message_count = len([m for m in current_conv["messages"] if m.get("role") == "user"])
                if user_message_count == 1:
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
                    except Exception as summary_error:
                        print(f"Failed to generate conversation summary: {summary_error}")
                        
            except Exception as e:
                st.error(f"Error: {e}")
                # Final fallback message
                response = "Sorry, I encountered an error while processing your request. Please try again."
                current_conv["messages"].append({
                    "role": "assistant",
                    "content": response,
                    "images": []
                })
                
                # Generate conversation summary after first user message
                user_message_count = len([m for m in current_conv["messages"] if m.get("role") == "user"])
                if user_message_count == 1:
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
                    except Exception as summary_error:
                        print(f"Failed to generate conversation summary: {summary_error}")
