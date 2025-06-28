import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class LLM:
    def __init__(self) -> None:
        self.groq_api_key = os.getenv("WTE_GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("WTE_GROQ_API_KEY is not set. Please check your .env file.")
        self.model = os.getenv("MODEL")
        if not self.model:
            raise ValueError("MODEL is not set. Please check your .env file.")

    def get_groq_llm(self):
        llm = ChatGroq(
            temperature=0,
            groq_api_key=self.groq_api_key,
            model=self.model,
        )
        return llm

