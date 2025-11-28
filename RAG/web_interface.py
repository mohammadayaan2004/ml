import streamlit as st
import pandas as pd
import logging
from rag_chatbot import RAGChatbot

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state with preloaded documents
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = RAGChatbot()
    # Preload sample documents
    try:
        with open("sample_documents.txt", "r", encoding="utf-8") as f:
            content = f.read()
            # Split by double newlines to separate documents
            documents = [doc.strip() for doc in content.split("\n\n") if doc.strip()]
            st.session_state.chatbot.add_documents(documents)
            logger.info(f"Preloaded {len(documents)} documents")
    except FileNotFoundError:
        logger.warning("sample_documents.txt not found")
        # Use fallback documents
        fallback_docs = [
            "The Earth revolves around the Sun in approximately 365.25 days.",
            "Water boils at 100 degrees Celsius at sea level.",
            "The human body has 206 bones.",
            "Photosynthesis is the process by which plants convert sunlight into energy.",
            "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            "Albert Einstein developed the theory of relativity.",
            "The Pacific Ocean is the largest ocean on Earth.",
            "Python is a high-level programming language created by Guido van Rossum.",
            "The capital of France is Paris.",
            "Leonardo da Vinci painted the Mona Lisa."
        ]
        st.session_state.chatbot.add_documents(fallback_docs)
    
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("🤖 RAG Chatbot")
st.markdown("""
This chatbot provides document-grounded responses based on the pre-loaded knowledge base.
""")

# Main chat interface
st.subheader("Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.chat(prompt)
        st.markdown(response)
        logger.info(f"User: {prompt}")
        logger.info(f"Bot: {response}")
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display document count information
st.info(f"Knowledge base contains {len(st.session_state.chatbot.documents)} documents")