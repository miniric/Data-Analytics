from langchain_openai import ChatOpenAI
from core.prompt_templates import report_prompt

from config.settings import OPENAI_MODEL, OPENAI_API_KEY
import os


def generate_report(question: str, answer: str) -> str:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # 溫度降低，讓回答盡量更精確一點
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
    prompt = report_prompt(question, answer)
    return llm.invoke(prompt).content