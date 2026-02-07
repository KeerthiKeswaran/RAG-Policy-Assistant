import chromadb
import os
import shutil
import uuid
from typing import List, Dict, Any

# Force CPU mode for Chroma embeddings to silence PyTorch logs/warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from langchain_core.documents import Document

class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "policy_documents"):
        """
        Initializes the vector store using the native ChromaDB client.
        
        Args:
            persist_directory (str): Local directory to persist vector DB.
            collection_name (str): Name of the Chroma collection.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Check for ChromaDB Cloud credentials
        chroma_api_key = os.getenv("CHROMA_API_KEY")
        chroma_tenant = os.getenv("CHROMA_TENANT")
        chroma_database = os.getenv("CHROMA_DATABASE")
        
        if chroma_api_key and chroma_tenant and chroma_database:
            # Use ChromaDB Cloud
            self.client = chromadb.CloudClient(
                api_key=chroma_api_key,
                tenant=chroma_tenant,
                database=chroma_database
            )
            print(f"Connected to ChromaDB Cloud - Database: {chroma_database}")
        else:
            # Fallback to local persistent client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            print(f"Using local ChromaDB at {self.persist_directory}")
        
        # Use ChromaDB's default embedding function (onnx/all-MiniLM-L6-v2) implicitly
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, documents: List[Document]):
        """Adds documents to the vector store."""
        if not documents:
            print("No documents provided to add.")
            return

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents] if documents else None
        # Generate unique IDs based on content or UUID. UUID is standard.
        ids = [str(uuid.uuid4()) for _ in documents]

        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(documents)} documents to vector store.")
        except Exception as e:
            print(f"Error adding documents: {e}")

    def query(self, query_text: str, k: int = 3) -> Dict[str, Any]:
        """
        Queries the vector store for similar documents.
        Returns the raw ChromaDB result dictionary.
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k
            )
            return results
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return {}

    def clear_store(self):
        """Clears the local vector store (for testing/reset)."""
        # Delete collection first
        try:
           self.client.delete_collection(self.collection_name)
        except:
           pass
           
        if os.path.exists(self.persist_directory):
            try:
                # Wait for file handles to close potentially, but shutil usually works
                shutil.rmtree(self.persist_directory)
                print(f"Cleared vector store at {self.persist_directory}")
            except Exception as e:
                print(f"Could not delete directory: {e}")

if __name__ == "__main__":
    # Test stub
    vs = VectorStore()
    # vs.add_documents([Document(page_content="Test document", metadata={"source": "test"})])
    # print(vs.query("Test"))
    pass
