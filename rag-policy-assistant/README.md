# Policy Assistant RAG

A Retrieval-Augmented Generation (RAG) system designed to answer questions about company policies (Refund, Cancellation, Shipping) using strict grounding to prevent hallucinations.

## Project Overview

This tool allows users to query a set of PDF policy documents. It uses a semantic search backend to retrieve relevant sections and an LLM (via Groq) to generate answers *only* from that context.

**Key Features:**
*   **Strict Grounding:** The system explicitly refuses to answer if information is not found in the documents.
*   **Deterministic Chunking:** Uses token-based splitting for consistent context windows.
*   **Dual Prompting:** Implements both a baseline (v1) and a strict, anti-hallucination (v2) prompt.
*   **Evaluation Suite:** Includes a script to test the system against a set of known questions.

## Architecture

The system follows a standard RAG pipeline:

1.  **Ingestion:**
    *   **Loader:** `PyPDFLoader` extracts text from PDFs in `data/`.
    *   **Chunker:** `RecursiveCharacterTextSplitter` (Token-based) splits text into ~500 token chunks with 75 token overlap. This ensures semantic continuity.
    *   **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` encodes chunks into vectors.
    *   **Storage:** `ChromaDB` (Local/Cloud supported) stores vectors and metadata.

2.  **Retrieval:**
    *   **Query Embedding:** User question is embedded using the same model.
    *   **Similarity Search:** System retrieves top-k (default 3) chunks based on cosine similarity.
    *   **Relevance Check:** A threshold mechanism filters out irrelevant chunks to trigger fallback responses.

3.  **Generation:**
    *   **LLM:** Groq (e.g., `llama3-70b-8192` or `mixtral-8x7b-32768`) generates the answer.
    *   **Prompting:** A strict prompt (Prompt V2) enforces "No Hallucination" rules.

## Setup & Run

### Prerequisites
*   Python 3.10+
*   Groq API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd rag-policy-assistant
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment:**
    *   Rename `.env.example` to `.env`.
    *   Add your Groq API Key: `GROQ_CLOUD_API_KEY=your_key_here`.

### Running the Application

Start the Streamlit UI:
```bash
streamlit run app.py
```
The first run will automatically index the documents found in `data/`.

### Running Evaluation

Execute the evaluation script to test the system:
```bash
python -m src.evaluator
```
Results specific to your documents will be saved to `evaluation_results.csv`.

## Chunking Strategy

We use **Recursive Character Splitting** with a target size of **500 tokens** and **75 tokens overlap**.

*   **Rationale:** 
    *   **Size (500 tokens):** Large enough to capture full policy clauses (context) but small enough to retrieve precise segments.
    *   **Overlap (75 tokens):** Prevents cutting off sentences or context at the edges of chunks, ensuring the LLM has complete statements.
    *   **Tokenizer:** Using `tiktoken` (cl100k_base) ensures we respect the LLM's context window accurately compared to simple character counting.

## Prompt Engineering

### Prompt V1 (Baseline)
A simple "Answer the question based on context" prompt. 
*   *Weakness:* Prone to using outside knowledge or hallucinating if context is vague.

### Prompt V2 (Strict - Implemented)
Explicitly designed to prevent hallucinations.
*   **Constraint 1:** "Answer ONLY from retrieved context."
*   **Constraint 2:** "If answer is not found, state: 'I'm sorry, but this information is not available...'"
*   **Structure:** Forces structured/bullet-point output for readability.
*   **Negative Constraint:** "Do NOT use outside knowledge."

## Evaluation

The system is evaluated on 5-8 questions covering fully answerable, partially answerable, and unanswerable scenarios.

| Question Type | Status | Goal |
| :--- | :--- | :--- |
| **Fully Answerable** | ✅ | Retrieve correct clause (e.g., "30 days") and answer accurately. |
| **Partially Answerable** | ⚠️ | Retrieve relevant section but state ambiguity if details are missing. |
| **Unanswerable** | ❌ | Retrieve NO relevant chunks or low similarity -> Return Refusal. |

*Sample Results Table (Run `src.evaluator` to generate actuals):*

| Question | Expected | Actual Result (Model) | Evaluation |
| :--- | :--- | :--- | :--- |
| "What is the return window?" | "30 days" | "You can return items within 30 days of receipt." | ✅ Correct |
| "Who is the CEO?" | Refusal | "I'm sorry, but this information is not available in the company policy documents." | ✅ Correct Refusal |
| "Can I pay with Bitcoin?"| Refusal | "I'm sorry, but this information is not available..." | ✅ Correct Refusal |

## Key Trade-offs
*   **Local vs Cloud Vector Store:** We default to local ChromaDB for simplicity and reproducibility. Cloud integration is scaffolded but not active without valid credentials.
*   **Retrieval Granularity:** Smaller chunks retrieval might miss broader context; larger chunks might dilute specific answers. 500 tokens is a balanced choice for policy docs.
*   **Strictness vs Helpfulness:** The system is biased towards *precision* (refusing to answer) rather than *recall* (guessing), which is critical for legal/policy bots.

## Future Improvement
**Source Attribution in UI:** Currently, the UI shows the answer. Ideally, it should highlight the specific PDF page number and paragraph used to generate the answer for verification.
