import os
import json
import openai
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
try:
    from pymongo import MongoClient, ASCENDING
    from bson import ObjectId
    MONGODB_AVAILABLE = True
except Exception as e:
    print(f"MongoDB not available: {e}")
    MongoClient = None
    ASCENDING = None
    ObjectId = None
    MONGODB_AVAILABLE = False

import hashlib
import uuid

logger = get_logger('Langchain-Chatbot')

# ============== MongoDB Configuration ==============
MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
DB_NAME = os.getenv("MONGODB_DB_NAME", "chatbot_edu")
COLL_CONV = "conversations"
COLL_MSG = "messages"
COLL_IMG = "message_images"
COLL_USERS = "users"
COLL_SESSIONS = "user_sessions"

# ============== MongoDB Functions ==============
@st.cache_resource(show_spinner=False)
def get_mongodb_client():
    """Get MongoDB client with connection caching"""
    if not MONGODB_AVAILABLE or not MONGODB_URI:
        return None
    
    try:
        client = MongoClient(MONGODB_URI, tls=True)
        # Test connection
        client.admin.command('ping')
        
        db = client[DB_NAME]
        # Create indexes
        db[COLL_CONV].create_index([("created_at", ASCENDING)])
        db[COLL_MSG].create_index([("conversation_id", ASCENDING), ("ts", ASCENDING)])
        db[COLL_MSG].create_index([("user_hash", ASCENDING)])
        db[COLL_IMG].create_index([("message_id", ASCENDING)])
        db[COLL_SESSIONS].create_index([("session_id", ASCENDING)])
        
        return db
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return None

def get_or_create_user_session(db):
    """Get or create user session for tracking"""
    if not db:
        return "anonymous", "session_" + str(uuid.uuid4())
    
    # Use session state for user tracking
    if "user_hash" not in st.session_state:
        st.session_state.user_hash = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:16]
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = "session_" + str(uuid.uuid4())
    
    user_hash = st.session_state.user_hash
    session_id = st.session_state.session_id
    
    # Update session info in database
    session_data = {
        "session_id": session_id,
        "user_hash": user_hash,
        "last_activity": datetime.utcnow().isoformat(),
        "created_at": datetime.utcnow().isoformat()
    }
    
    try:
        db[COLL_SESSIONS].update_one(
            {"session_id": session_id},
            {"$set": session_data},
            upsert=True
        )
    except Exception as e:
        print(f"Failed to update session: {e}")
    
    return user_hash, session_id

def save_conversation_to_mongodb(conversation_data):
    """Save conversation to MongoDB"""
    db = get_mongodb_client()
    if not db:
        return None
    
    try:
        user_hash, session_id = get_or_create_user_session(db)
        
        # Save conversation
        conv_doc = {
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "model": conversation_data.get("model", "gpt-5-mini"),
            "name": conversation_data.get("name", "New Conversation"),
            "user_hash": user_hash,
            "session_id": session_id,
            "message_count": len([m for m in conversation_data.get("messages", []) if m.get("role") == "user"]),
            "total_tokens_used": 0,
            "enable_internet": conversation_data.get("enable_internet", False),
            "enable_documents": conversation_data.get("enable_documents", True),
            "enable_images": conversation_data.get("enable_images", True)
        }
        
        result = db[COLL_CONV].insert_one(conv_doc)
        conversation_id = result.inserted_id
        
        # Save messages
        for msg in conversation_data.get("messages", []):
            msg_doc = {
                "conversation_id": conversation_id,
                "role": msg.get("role"),
                "content": msg.get("content", ""),
                "ts": datetime.utcnow().isoformat(),
                "user_hash": user_hash,
                "session_id": session_id,
                "has_images": len(msg.get("images", [])) > 0
            }
            
            msg_result = db[COLL_MSG].insert_one(msg_doc)
            
            # Save images if any
            for img_data in msg.get("images", []):
                img_doc = {
                    "message_id": msg_result.inserted_id,
                    "conversation_id": conversation_id,
                    "mime": img_data.get("mime", "image/jpeg"),
                    "data": img_data.get("data"),
                    "ts": datetime.utcnow().isoformat(),
                    "user_hash": user_hash
                }
                db[COLL_IMG].insert_one(img_doc)
        
        return conversation_id
    except Exception as e:
        print(f"Failed to save conversation to MongoDB: {e}")
        return None

def load_conversations_from_mongodb():
    """Load conversations from MongoDB"""
    db = get_mongodb_client()
    if not db:
        return {}
    
    try:
        user_hash, _ = get_or_create_user_session(db)
        
        # Get conversations for this user
        conversations = list(db[COLL_CONV].find(
            {"user_hash": user_hash}
        ).sort([("updated_at", -1), ("_id", -1)]).limit(50))
        
        conversations_dict = {}
        for conv in conversations:
            conv_id = str(conv["_id"])
            
            # Load messages for this conversation
            messages = list(db[COLL_MSG].find(
                {"conversation_id": conv["_id"]}
            ).sort([("_id", 1)]))
            
            # Load images for messages
            message_ids = [msg["_id"] for msg in messages]
            images = {}
            if message_ids:
                img_docs = list(db[COLL_IMG].find({"message_id": {"$in": message_ids}}))
                for img in img_docs:
                    if img["message_id"] not in images:
                        images[img["message_id"]] = []
                    images[img["message_id"]].append({
                        "mime": img["mime"],
                        "data": img["data"]
                    })
            
            # Reconstruct message objects
            message_objects = []
            for msg in messages:
                msg_obj = {
                    "role": msg["role"],
                    "content": msg["content"],
                    "images": images.get(msg["_id"], [])
                }
                message_objects.append(msg_obj)
            
            conversations_dict[conv_id] = {
                "id": conv_id,
                "name": conv["name"],
                "created_at": conv["created_at"],
                "updated_at": conv["updated_at"],
                "messages": message_objects,
                "model": conv["model"],
                "enable_internet": conv.get("enable_internet", False),
                "enable_documents": conv.get("enable_documents", True),
                "enable_images": conv.get("enable_images", True),
                "memory": None,  # Will be rebuilt
                "vectorstore": None  # Will be rebuilt
            }
        
        return conversations_dict
    except Exception as e:
        print(f"Failed to load conversations from MongoDB: {e}")
        return {}

def update_conversation_in_mongodb(conversation_id, updates):
    """Update conversation in MongoDB"""
    db = get_mongodb_client()
    if not db:
        return False
    
    try:
        user_hash, _ = get_or_create_user_session(db)
        
        # Update conversation
        update_doc = {
            "updated_at": datetime.utcnow().isoformat(),
            **updates
        }
        
        if ObjectId:
            db[COLL_CONV].update_one(
                {"_id": ObjectId(conversation_id), "user_hash": user_hash},
                {"$set": update_doc}
            )
        else:
            db[COLL_CONV].update_one(
                {"_id": conversation_id, "user_hash": user_hash},
                {"$set": update_doc}
            )
        
        return True
    except Exception as e:
        print(f"Failed to update conversation in MongoDB: {e}")
        return False

def save_message_to_mongodb(conversation_id, role, content, images=None):
    """Save a single message to MongoDB"""
    db = get_mongodb_client()
    if not db:
        return None
    
    try:
        user_hash, session_id = get_or_create_user_session(db)
        
        # Save message
        msg_doc = {
            "conversation_id": ObjectId(conversation_id) if ObjectId else conversation_id,
            "role": role,
            "content": content,
            "ts": datetime.utcnow().isoformat(),
            "user_hash": user_hash,
            "session_id": session_id,
            "has_images": len(images or []) > 0
        }
        
        msg_result = db[COLL_MSG].insert_one(msg_doc)
        
        # Save images if any
        if images:
            for img_data in images:
                img_doc = {
                    "message_id": msg_result.inserted_id,
                    "conversation_id": ObjectId(conversation_id) if ObjectId else conversation_id,
                    "mime": img_data.get("mime", "image/jpeg"),
                    "data": img_data.get("data"),
                    "ts": datetime.utcnow().isoformat(),
                    "user_hash": user_hash
                }
                db[COLL_IMG].insert_one(img_doc)
        
        # Update conversation stats
        conv_id = ObjectId(conversation_id) if ObjectId else conversation_id
        db[COLL_CONV].update_one(
            {"_id": conv_id},
            {
                "$set": {"updated_at": datetime.utcnow().isoformat()},
                "$inc": {"message_count": 1}
            }
        )
        
        return msg_result.inserted_id
    except Exception as e:
        print(f"Failed to save message to MongoDB: {e}")
        return None

#decorator
def enable_chat_history(func):
    if os.environ.get("OPENAI_API_KEY"):

        # to clear chat history after swtching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # to show chat history on ui
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def choose_custom_openai_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="SELECTED_OPENAI_API_KEY"
        )
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()

    model = "gpt-4.1-mini"
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        available_models = [{"id": i.id, "created":datetime.fromtimestamp(i.created)} for i in client.models.list() if str(i.id).startswith("gpt")]
        available_models = sorted(available_models, key=lambda x: x["created"])
        available_models = [i["id"] for i in available_models]

        model = st.sidebar.selectbox(
            label="Model",
            options=available_models,
            key="SELECTED_OPENAI_MODEL"
        )
    except openai.AuthenticationError as e:
        st.error(e.body["message"])
        st.stop()
    except Exception as e:
        print(e)
        st.error("Something went wrong. Please try again later.")
        st.stop()
    return model, openai_api_key


def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))

@st.cache_resource
def configure_embedding_model():
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embedding_model

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v

def get_gpt_client():
    """Get OpenAI client with API key from secrets"""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in secrets")
        return openai.OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Failed to create OpenAI client: {e}")
        return None

def call_gpt_api(messages, model="gpt-4o-mini", stream=False):
    """Call GPT API with error handling - let model decide temperature automatically"""
    try:
        client = get_gpt_client()
        if not client:
            return None
            
        # Let the model decide temperature automatically (no temperature parameter)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream
        )
        
        return response
    except Exception as e:
        print(f"GPT API call failed: {e}")
        return None

def generate_conversation_summary(messages: list) -> str:
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
            
            client = get_gpt_client()
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

def save_conversations_to_file(conversations: dict, file_path: str = "conversations.json"):
    """Save conversations to JSON file for persistence"""
    try:
        # Create a copy and remove non-serializable objects
        conversations_copy = {}
        for conv_id, conv_data in conversations.items():
            conv_copy = conv_data.copy()
            # Remove memory object as it can't be serialized
            if 'memory' in conv_copy:
                del conv_copy['memory']
            # Remove vectorstore as it can't be serialized
            if 'vectorstore' in conv_copy:
                del conv_copy['vectorstore']
            conversations_copy[conv_id] = conv_copy
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversations_copy, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save conversations: {e}")

def load_conversations_from_file(file_path: str = "conversations.json") -> dict:
    """Load conversations from JSON file"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
            
            # Restore non-serializable objects for each conversation
            for conv_id, conv_data in conversations.items():
                # Add memory object
                from langchain.memory import ConversationBufferMemory
                conv_data['memory'] = ConversationBufferMemory(return_messages=True)
                
                # Add vectorstore (will be None initially)
                conv_data['vectorstore'] = None
                
                # Restore memory from messages if they exist
                if conv_data.get('messages'):
                    for msg in conv_data['messages']:
                        if msg['role'] == 'user':
                            conv_data['memory'].save_context(
                                {"input": msg["content"]}, 
                                {"output": ""}
                            )
                        elif msg['role'] == 'assistant':
                            # Find the most recent user message before this assistant message
                            messages = conv_data['messages']
                            for i, prev_msg in enumerate(messages):
                                if prev_msg == msg:
                                    for j in range(i-1, -1, -1):
                                        if messages[j]['role'] == 'user':
                                            conv_data['memory'].save_context(
                                                {"input": messages[j]["content"]}, 
                                                {"output": msg["content"]}
                                            )
                                            break
                                    break
            
            return conversations
        return {}
    except Exception as e:
        print(f"Failed to load conversations: {e}")
        return {}

def update_conversation_summary(conversation_id: str, summary: str, file_path: str = "conversations.json"):
    """Update conversation summary in the persistent storage"""
    try:
        conversations = load_conversations_from_file(file_path)
        if conversation_id in conversations:
            conversations[conversation_id]["name"] = summary
            conversations[conversation_id]["updated_at"] = datetime.now().isoformat()
            save_conversations_to_file(conversations, file_path)
    except Exception as e:
        print(f"Failed to update conversation summary: {e}")
