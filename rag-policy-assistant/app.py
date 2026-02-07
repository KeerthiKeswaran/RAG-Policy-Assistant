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

from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker
from src.vector_store import VectorStore
from src.rag_pipeline import RagPipeline

# Load environment variables
load_dotenv(dotenv_path=".env.example")

st.set_page_config(page_title="Policy Assistant", layout="wide")

@st.cache_resource(show_spinner=False)
def initialize_rag_system():
    """Initializes the RAG system components and loads data if needed."""
    
    import sys
    from io import StringIO
    
    # Suppress print statements during initialization
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # 1. Vector Store Setup (Native ChromaDB)
        vector_store = VectorStore(persist_directory="chroma_db", collection_name="policies")
        
        # Check if vector store has documents
        count = vector_store.collection.count()
        
        if count == 0:
            # Load
            loader = DocumentLoader()
            raw_docs = loader.load_documents()
            
            # Chunk
            chunker = TextChunker()
            chunks = chunker.split_documents(raw_docs)
            
            # Store
            vector_store.add_documents(chunks)
            sys.stdout = old_stdout
            st.success(f"Successfully indexed {len(chunks)} chunks from {len(raw_docs)} documents.")
        else:
            sys.stdout = old_stdout
            
        # 2. RAG Pipeline
        pipeline = RagPipeline(vector_store)
        
        return pipeline
        
    except Exception as e:
        sys.stdout = old_stdout
        st.error(f"Error initializing RAG system: {e}")
        return None
    finally:
        sys.stdout = old_stdout
    return pipeline

def main():
    st.title("Policy Assistant (RAG)")
    st.markdown("""
    Ask questions about the company's **Refund, Cancellation, and Shipping policies**.
    The system will answer strictly based on the provided documents.
    """)
    
    # Initialize implementation
    pipeline = initialize_rag_system()
    
    if not pipeline:
        st.error("Failed to initialize RAG system. Check logs.")
        st.stop()
    
    # User Input
    query = st.text_input("Enter your question:", placeholder="e.g., What is the refund policy for sale items?")
    
    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Analyzing policy documents..."):
                try:
                    # Retrieve context first to show transparency (optional but good for debugging)
                    # For UI simplicity, we just run the pipeline
                    # Note: run calls retrieve internally now
                    
                    response = pipeline.run(query)
                    
                    st.subheader("Answer:")
                    st.write(response)
                    
                    # Optional: Expand to show source context
                    with st.expander("View Retrieved Context (Source)"):
                        docs = pipeline.retrieve(query) # Re-run retrieval just for display
                        if docs:
                            for i, content in enumerate(docs):
                                st.markdown(f"**Chunk {i+1}:**")
                                st.text(content)
                        else:
                            st.write("No relevant context found.")
                            
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
