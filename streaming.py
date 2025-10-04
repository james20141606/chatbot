from langchain_core.callbacks import BaseCallbackHandler
import streamlit as st

class StreamHandler(BaseCallbackHandler):
    
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

class GPTStreamHandler:
    """Handler for GPT streaming responses"""
    
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def handle_stream(self, stream):
        """Handle GPT streaming response"""
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                delta = chunk.choices[0].delta.content
                chunks.append(delta)
                self.text = "".join(chunks)
                self.container.markdown(self.text)
        
        return self.text