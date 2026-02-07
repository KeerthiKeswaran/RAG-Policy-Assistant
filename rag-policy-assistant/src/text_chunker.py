from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 75, separators: List[str] = ["\n\n", "\n", " ", ""]):
        """
        Initializes the text chunker.
        Args:
            chunk_size (int): Target size of each chunk in tokens.
            chunk_overlap (int): Overlap between chunks in tokens as text context.
            separators (List[str]): Separators used for splitting text.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        # Using standard RecursiveCharacterTextSplitter for simple character-based splitting
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits the provided documents into chunks."""
        chunks = self.splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} original documents.")
        return chunks

if __name__ == "__main__":
    # Test stub
    loader = PyPDFLoader("data/refund_policy.pdf")
    docs = loader.load()
    chunker = TextChunker()
    chunks = chunker.split_documents(docs)
    if chunks:
        print(f"Sample chunk: {chunks[0].page_content[:100]}...")
