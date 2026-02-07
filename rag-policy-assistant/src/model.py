from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

def get_groq_model(model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.2):
    """
    Initializes and returns the Groq LLM instance.
    
    Args:
        model_name (str): The name of the Groq model.
        temperature (float): Generation temperature.
    
    Returns:
        ChatGroq: The initialized LangChain ChatGroq model.
    """
    load_dotenv(dotenv_path=".env.example")
    
    api_key = os.getenv("GROQ_CLOUD_API_KEY") # Or check GROQ_API_KEY
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
        
    if not api_key:
        raise ValueError("GROQ_CLOUD_API_KEY environment variable is not set. Please add it to .env.")

    # Using the exact configuration from user request
    llm = ChatGroq(
        temperature=temperature,
        groq_api_key=api_key,
        model_name=model_name
    )
    
    return llm

if __name__ == "__main__":
    try:
        model = get_groq_model()
        print(f"Model initialized: {model.model_name}")
    except Exception as e:
        print(f"Error initializing model: {e}")
