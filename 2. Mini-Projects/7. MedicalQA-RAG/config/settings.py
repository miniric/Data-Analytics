import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = "gpt-5-nano"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DB_PATH = ":memory:"  # 如果有 Qdrant 位置也可以視情況更改, 這邊先使用記憶體暫存就好