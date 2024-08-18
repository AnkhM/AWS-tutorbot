from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Access environment variables from .env
load_dotenv()

# Access OpenAI API Key 
api_key = os.getenv("OPENAI_API_KEY")

def get_embeddings():
    embeddings = OpenAIEmbeddings(api_key = api_key, model = "text-embedding-3-large")
    return embeddings
