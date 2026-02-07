import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def load_documents(self) -> List[Document]:
        """Loads all PDF files from the data directory."""
        documents = []
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory '{self.data_dir}' not found.")
            
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".pdf"):
                filepath = os.path.join(self.data_dir, filename)
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {filename}")
        
        return documents

if __name__ == "__main__":
    loader = DocumentLoader()
    docs = loader.load_documents()
    print(f"Total documents loaded: {len(docs)}")
