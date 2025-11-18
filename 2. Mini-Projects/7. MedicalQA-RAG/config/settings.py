import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = "gpt-5-nano"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
