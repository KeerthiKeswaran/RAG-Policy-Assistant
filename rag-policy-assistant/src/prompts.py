from langchain_core.prompts import ChatPromptTemplate

# Baseline Prompt (V1)
PROMPT_V1 = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context.

Context:
{context}

Question:
{question}
"""
)

# Improved Prompt (V2) - Strict, No Hallucinations, Structured
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
8. If information is partially present but inconclusive, state this in one sentence and refuse.

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

if __name__ == "__main__":
    print(PROMPT_V2.format(context="[Example Context]", question="What is the refund policy?"))
