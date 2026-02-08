import os
import warnings
import streamlit as st

# Setup optimal environment before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from typing import List

from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env.example")

# Fix for Windows Event Loop RuntimeError
import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@st.cache_resource(show_spinner=False)
def initialize_rag_system():
    """Initializes the RAG system components with visible progress."""
    
    import sys
    import os
    from io import StringIO
    
    # Create a placeholder for status updates
    status_container = st.empty()
    
    # Use the placeholder for the status container
    with status_container.status("Initializing System...", expanded=True) as status:
        
        # 1. Suppress noise but allow our own logs
        st.write("🔧 Setting up environment...")
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            # Lazy imports
            st.write("📚 Loading core modules...")
            from src.document_loader import DocumentLoader
            from src.text_chunker import TextChunker
            from src.vector_store import VectorStore
            from src.rag_pipeline import RagPipeline
            
            # 2. Vector Store Setup
            st.write("💾 Connecting to Vector Database (ChromaDB)...")
            vector_store = VectorStore(collection_name="policies")
            
            # Check documents
            count = vector_store.collection.count()
            st.write(f"📊 Found {count} existing documents in current collection.")
            
            if count == 0:
                st.write("🚀 Empty DB detected. Starting ingestion process...")
                
                st.write("   - Loading PDFs from /data...")
                loader = DocumentLoader()
                raw_docs = loader.load_documents()
                
                st.write(f"   - Splitting {len(raw_docs)} pages into chunks...")
                chunker = TextChunker()
                chunks = chunker.split_documents(raw_docs)
                
                st.write(f"   - Indexing {len(chunks)} chunks into ChromaDB (this may take a moment)...")
                vector_store.add_documents(chunks)
                
                st.success("✅ Ingestion complete!")
            
            # 3. RAG Pipeline
            st.write("🤖 Initializing RAG Pipeline (Llama 3)...")
            pipeline = RagPipeline(vector_store)
            
            status.update(label="System Ready!", state="complete", expanded=False)
            
        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            st.error(f"❌ Initialization Error: {e}")
            status.update(label="Initialization Failed", state="error")
            return None
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
    # Clear the status container after successful initialization
    status_container.empty()
    return pipeline

def main():
    st.set_page_config(page_title="Policy Assistant", page_icon="", layout="wide")
    
    st.markdown("""
        <style>
        /* Shimmer Effect Keyframes */
        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }
        
        /* Shimmer Loader Class */
        .shimmer-loader {
            animation: shimmer 2s infinite linear;
            background: linear-gradient(to right, #2b313e 4%, #3b4252 25%, #2b313e 36%);
            background-size: 1000px 100%;
            height: 20px;
            width: 100%;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        
        .shimmer-container {
            padding: 20px;
            border-radius: 8px;
            background-color: #1a1c24;
            margin: 10px 0;
            border: 1px solid #30323a;
        }
        
        /* Chat Input Styling - Move Up */
        .stChatInput {
            bottom: 40px !important;
        }
        
        /* Remove Avatars */
        .stChatMessage .stAvatar {
            display: none !important;
        }
        
        /* Message Container - Reset defaults */
        [data-testid="stChatMessage"] {
            background-color: transparent !important;
            padding: 10px 0 !important;
            gap: 0 !important;
        }
        
        /* Bubble Styling Common */
        [data-testid="stChatMessageContent"] {
            padding: 12px 16px !important;
            border-radius: 20px !important;
            max-width: 75% !important;
            width: fit-content !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        /* USER MESSAGE (Right - Subtle Blue) */
        [data-testid="stChatMessage"]:has(div[aria-label="user"]) {
            flex-direction: row-reverse !important;
        }
        
        [data-testid="stChatMessage"]:has(div[aria-label="user"]) [data-testid="stChatMessageContent"] {
            background-color: #2b3b55 !important; /* Subtle Navy Blue */
            color: #e0e0e0 !important;
            margin-left: auto !important;
            border-bottom-right-radius: 4px !important;
        }
        
        /* ASSISTANT MESSAGE (Left - Subtle Grey) */
        [data-testid="stChatMessage"]:has(div[aria-label="assistant"]) [data-testid="stChatMessageContent"] {
            background-color: #2c2e35 !important; /* Subtle Dark Grey */
            color: #e0e0e0 !important;
            margin-right: auto !important;
            border-bottom-left-radius: 4px !important;
            border: 1px solid #3c3f4a;
        }
        
        /* Ensure text in bubbles is readable and inherits correct color */
        [data-testid="stChatMessageContent"] p {
            color: inherit !important;
            margin: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("RAG Policy Assistant")
    
    # Initialize implementation
    pipeline = initialize_rag_system()
    
    if not pipeline:
        st.error("Failed to initialize RAG system. Check logs.")
        st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Send a message..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        # Add to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response with Shimmer Loading
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Show shimmer animation while processing
            message_placeholder.markdown("""
                <div class="shimmer-container">
                    <div class="shimmer-loader" style="width: 70%;"></div>
                    <div class="shimmer-loader" style="width: 90%;"></div>
                    <div class="shimmer-loader" style="width: 60%;"></div>
                </div>
            """, unsafe_allow_html=True)
            
            try:
                # Get response
                response = pipeline.run(prompt)
                
                # Replace shimmer with actual response
                message_placeholder.markdown(response)
                
                # Optional details
                with st.expander("🔍 View Source Details"):
                    docs = pipeline.retrieve(prompt)
                    if docs:
                        for i, content in enumerate(docs):
                            st.markdown(f"**Source {i+1}:**")
                            st.caption(content)
                    else:
                        st.write("No specific source context found.")
                        
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                message_placeholder.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
