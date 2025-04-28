from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY
import logging

def setup_llm():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=OPENAI_API_KEY)
    logging.info(f"Using LLM: {llm.model_name}")
    return llm
