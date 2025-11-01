from langchain_core.prompts import ChatPromptTemplate

# 將 qa_prompt 改成 langchian 的  ChatPromptTemplate 物件
medical_qa_prompt = ChatPromptTemplate.from_template(
"""
"System"
你是一位專業的臨床醫師助理。請依以下步驟回答使用者的問題：
1. 根據檢索到的醫療資料，分析問題的原因或機制。
2. 提出具體可執行的衛教建議。
3. 若資料不足，請誠實說明限制。

"Human input"
以下是之前對話記錄（可能包含病人症狀、提問、你的回答）：
{history} 

以下為檢索到的資料：
{context}

問題：{question}

"System"
請以條列格式、繁體中文回答。
"""
)

def report_prompt(question: str, answer: str) -> str:
    return f"""

"System"
請將以下問答整理成一份「病人衛教報告摘要」：

"Human input"
---
問題：{question}
回答：{answer}
---

"System"
請以正式及專業醫療從業人員語氣撰寫，控制在200字以內。
"""