"""
summary
"""

import os

from dotenv import load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_KEY")

# Create OpenAI embedding and llm
oepnai_embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
openai_llm = OpenAI(openai_api_key=OPENAI_API_KEY)
