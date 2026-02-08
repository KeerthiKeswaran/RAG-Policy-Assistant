import chromadb
import os
from typing import List, Dict, Any

# Force CPU mode for Chroma embeddings to silence PyTorch logs/warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class VectorStore:
    def __init__(self, collection_name: str = "policy_documents"):
        """
        Initializes the vector store using an in-memory ChromaDB client.
        This avoids permission errors on Streamlit Cloud.
        """
        self.client = chromadb.Client()  # in-memory (Ephemeral)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents):
        """Adds documents to the vector store."""
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        # Using simple range IDs as requested for the fix
        ids = [str(i) for i in range(len(texts))]

        try:
            self.collection.add(
                documents=texts,
                ids=ids
            )
            print(f"Added {len(documents)} documents to in-memory vector store.")
        except Exception as e:
            print(f"Error adding documents: {e}")

    def query(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Queries the vector store for similar documents.
        Returns the raw ChromaDB result dictionary.
        """
        try:
            return self.collection.query(
                query_texts=[query],
                n_results=k
            )
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return {}
