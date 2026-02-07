import os
from typing import List, Dict, Any, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.vector_store import VectorStore
from src.model import get_groq_model

# Prompt definitions
PROMPT_V1 = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context.

Context:
{context}

Question:
{question}
"""
)

PROMPT_V2 = ChatPromptTemplate.from_template(
    """You are a strict policy assistant. Your sole purpose is to answer user questions based ONLY on the provided context.

**Instructions:**
1. Analyze the Context below carefully.
2. Answer the Question using **only** the information present in the Context.
3. If the answer is not explicitly stated in the specific Context, you MUST respond with: "I'm sorry, but this information is not available in the company policy documents."
4. Do NOT make up, guess, or infer information that is not there.
5. Do NOT use outside knowledge.
6. If the context contains conflicting information, state the conflict clearly.
7. Structure your answer clearly with bullet points if applicable.

**Context:**
{context}

**Question:**
{question}

**Answer:**
"""
)

def get_prompt(version: str = "v2") -> ChatPromptTemplate:
    if version == "v1":
        return PROMPT_V1
    elif version == "v2":
        return PROMPT_V2
    else:
        raise ValueError(f"Unknown prompt version: {version}")

class RagPipeline:
    def __init__(self, vector_store: VectorStore, llm_model: str = "llama-3.3-70b-versatile"):
        """
        Initializes the RAG Pipeline.
        
        Args:
            vector_store (VectorStore): The initialized custom VectorStore instance.
            llm_model (str): The name of the Groq model to use.
        """
        self.vector_store = vector_store
        # Use centralized model initialization
        self.llm = get_groq_model(model_name=llm_model)
        
        self.prompt = get_prompt("v2") # Default to strict prompt

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieves relevant document contents based on the query.
        
        Args:
            query (str): User query.
            k (int): Number of documents to retrieve. 
        
        Returns:
            List[str]: Content of relevant documents.
        """
        # Query the vector store
        results = self.vector_store.query(query, k=k)
        
        relevant_docs = []
        
        # Process ChromaDB results dict
        # structure: {'ids': [['id1']], 'embeddings': None, 'documents': [['text1']], 'uris': None, 'data': None, 'metadatas': [[{'source': 'x'}]], 'distances': [[0.5]]}
        if results and 'documents' in results and results['documents']:
            # results['documents'] is a list of lists (one per query)
            # We only sent one query, so take the first list
            docs_list = results['documents'][0]
            
            # Simple check: if list is empty, no docs found
            if not docs_list:
                return []
                
            # We can also check distances if we want strict filtering, but for now just return top k
            relevant_docs = docs_list
                
        return relevant_docs

    def run(self, query: str) -> str:
        """
        Runs the RAG pipeline end-to-end.
        
        Args:
            query (str): User query.
            
        Returns:
            str: Generated answer.
        """
        docs = self.retrieve(query)
        
        if not docs:
            return "I'm sorry, but I couldn't find any information in the policy documents related to your query."
        
        context_str = "\n\n".join(docs)
        
        chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Invoke chain
        return chain.invoke({"context": context_str, "question": query})

if __name__ == "__main__":
    pass
