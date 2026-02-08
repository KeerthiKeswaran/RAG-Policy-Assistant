import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self):
        # Resolve 'data' directory relative to this file (src/document_loader.py -> src -> project root)
        base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = base_dir / "data"

    def load_documents(self) -> List[Document]:
        """Loads all PDF files from the data directory."""
        documents = []
        
        # Robust check using pathlib
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at {self.data_dir}")
            
        # Iterate using pathlib's glob (cleaner)
        for file_path in self.data_dir.glob("*.pdf"):
            try:
                # PyPDFLoader expects a string path
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
        
        return documents

if __name__ == "__main__":
    loader = DocumentLoader()
    try:
        docs = loader.load_documents()
        print(f"Total documents loaded: {len(docs)}")
    except Exception as e:
        print(e)
