# RAG Policy Assistant

## 1. Project Overview
This project is a Retrieval-Augmented Generation (RAG) based question-answering assistant designed to query company policy documents (Refund, Cancellation, and Shipping policies). The system is engineered to provide grounded answers strictly based on the provided source text, minimizing hallucinations and ensuring that users receive accurate information derived solely from the official documentation.

## 2. Architecture Overview
The system follows a standard RAG architecture:

1.  **Ingestion**: PDF documents are loaded and processed.
2.  **Chunking**: Text is split into manageable segments.
3.  **Embedding & Storage**: Chunks are embedded and stored in a vector database (ChromaDB).
4.  **Retrieval**: User queries are converted to embeddings to find semantically similar chunks.
5.  **Generation**: The retrieved context and the user query are passed to a Large Language Model (LLM) via a strict prompt to generate an answer.

```
[PDF Docs] -> [Loader] -> [Chunker] -> [Vector DB]
                                           ^
                                           | (Retrieval)
[User Query] -> [Embedding] ---------------+
                                           |
                                     [LLM Context] -> [Answer]
```

## 3. Data Preparation
-   **Loading**: `PyPDFLoader` is used to extract text from PDF files located in the `data/` directory.
-   **Cleaning**: Basic whitespace normalization is applied during loading.
-   **Chunking**: The text is split using `RecursiveCharacterTextSplitter`.
    -   **Chunk Size**: 500 tokens. This size was chosen to capture sufficient context for a policy clause without including too much irrelevant information.
    -   **Overlap**: 75 tokens. This ensures continuity between chunks, preventing sentences or logical clauses from being cut off at boundaries.

## 4. RAG Pipeline
-   **Vector Database**: ChromaDB is used as the local vector store. It handles the storage of embeddings and performs similarity searches.
-   **Retrieval**: The system queries the top `k` (default 3) most relevant chunks based on cosine similarity to the user's query.
-   **Generation**: The `llama-3.3-70b-versatile` model (via Groq) is used for generation. It receives the retrieved chunks as "Context" and the user's question, producing a natural language response.

## 5. Prompt Engineering
The system uses a strictly engineered prompt to enforce grounding. Key aspects include:
-   **Role Definition**: The model is defined as a "strict policy assistant."
-   **Negative Constraints**: Explicit instructions to "Do NOT make up information" and "Do NOT use outside knowledge."
-   **Fallback Mechanism**: Instructions to state "I'm sorry, but this information is not available..." if the context is insufficient.
-   **Conflict Resolution**: Instructions to clearly state if conflicting information is found in the context.

## 6. Edge Case Handling
-   **No Relevant Documents**: If the vector search returns results that are valid but contain no relevant answers (or if the LLM determines the context doesn't answer the question), the model is instructed to refuse the request politely.
-   **Out of Domain Questions**: Questions unrelated to the provided policies will result in a standard refusal message, as the grounded context will not contain the answer.

## 7. Deployment & Setup
The application is built with Streamlit and designed to run on the Streamlit Community Cloud or locally.

### Prerequisites
-   Python 3.10+
-   A Groq Cloud API Key (`GROQ_API_KEY`)

### Local Setup
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Create a `.env` file and add your API keys:
    ```env
    # Required for LLM
    GROQ_API_KEY=your_groq_api_key_here

    # Optional: For ChromaDB Cloud (defaults to local if not set)
    CHROMA_API_KEY=your_chroma_api_key
    CHROMA_TENANT=your_tenant_id
    CHROMA_DATABASE=your_database_name
    ```
4.  Run the application:
    ```bash
    streamlit run app.py
    ```

## 8. Trade-offs and Design Decisions
1.  **Strict Refusal vs. Helpfulness**: The system leans heavily towards strict refusal. If the answer isn't explicitly in the text, it will not attempt to answer using general knowledge. This trades off "chatty" helpfulness for factual accuracy and safety.
2.  **Simple Chunking vs. Semantic Splitting**: Standard recursive character splitting was chosen over semantic or agentic splitting. While semantic splitting might yield cleaner boundaries, recursive splitting is deterministic, faster to implement, and sufficiently effective for structured policy documents.

## 9. Future Improvements
-   **Hybrid Search**: Implementing a hybrid search approach (combining keyword/BM25 with semantic vector search) could improve retrieval accuracy for specific terminology (e.g., specific policy codes or exact phrases).
-   **Citing Sources**: Modifying the pipeline to return and display the specific page numbers or filenames alongside the answer for better verification.
-   **Metadata Filtering**: Adding metadata tags (e.g., "shipping", "refund") to chunks to allow for pre-filtering based on the query intent.

## 10. Evaluation

The following Q&A pairs demonstrate the system's performance on key policy questions:

**Q1. What is the time limit for requesting a refund after delivery?**
> **Answer:**
> The time limit for requesting a refund after delivery is 30 calendar days from the confirmed delivery date.
>
> **Score:** ✅

**Q2. How can a customer cancel an order before it is shipped?**
> **Answer:**
> A customer can cancel an order before shipment based on its status:
>
> Orders in Pending/Awaiting Payment status can be cancelled through the account dashboard or customer support.
>
> Orders in Processing status require contacting customer support, and cancellation is not guaranteed.
>
> Customized or personalized goods can be cancelled within two hours of order placement.
>
> High-value orders may require additional confirmation before cancellation.
>
> **Score:** ✅

**Q3. Can a customer return a discounted item purchased during a sale?**
> **Answer:**
> The policy documents do not explicitly mention return eligibility for discounted or sale items.
> Therefore, this information is not available in the company policy documents.
>
> **Score:** (Correct refusal with clear limitation)

**Q4. What happens if a package is delayed due to courier issues?**
> **Answer:**
> The shipping policy does not specify how courier-related delays are handled or whether compensation is provided.
> As a result, this information is not available in the company policy documents.
>
> **Score:** (Correct refusal with clear limitation)

**Q5. Does the company offer refunds for digital products like software subscriptions?**
> **Answer:**
> The policy references digital products but does not clearly state whether refunds are offered for software subscriptions.
> Therefore, this information is not available in the company policy documents.
>
> **Score:** (Correct unanswerable handling)

**Q6. What is the customer support phone number for urgent issues?**
> **Answer:**
> The policy documents do not provide any customer support phone number.
> This information is not available in the company policy documents.
>
> **Score:** (Correct unanswerable handling)

**Q7. Do employees receive special internal discounts?**
> **Answer:**
> Employee benefits or internal discounts are not mentioned in the policy documents.
> This information is not available in the company policy documents.
>
> **Score:** (Correct unanswerable handling)
