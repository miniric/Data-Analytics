from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 從我們的 config 載入 api_key, LLM_model, 和 向量資料庫的位址
from config.settings import OPENAI_API_KEY, OPENAI_MODEL, VECTOR_DB_PATH
import os


def build_rag_pipeline():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # 透過 loader 載入預先存好的正確醫療資訊    
    loader = TextLoader("data/medical_facts.txt")
    docs = loader.load()

    # 資訊向量化，使用openAI的模型將文字向量化
    embeddings = OpenAIEmbeddings()

    # 使用 Qdrant 建立向量資料庫
    # 參數:
    # docs = 預先載好的正確醫療資訊
    # embediings = 使用 OpenAI
    # 最終產出一 個medical_docs 的向量集合
    db = Qdrant.from_documents(
        docs,
        embeddings,
        location=VECTOR_DB_PATH,
        collection_name="medical_docs"
    )

    # RAG 裡面的 retriever，使用輸入者的關鍵字檢索向量資料庫裡面資訊
    # 參數
    # search_kwargs, 找出最接近的 x 筆資料
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # llm 指定 settings 裡面的 OpenAI model, 溫度控制回答的準確性 (t越高, 越容易即興發揮)
    # (t這面可以想像在最後一層的 softmax 裡面，增加一個參數t 去調整他機率分布的值, 當t越高時
    # 會將機率低的 tokens 進行機率補償)
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)

    # prompt engineering ，建立 prompt 模版
    template = """
    你是一位專業的臨床醫師助理，請根據以下資料回答問題。
    若資料不足，請誠實說明限制。

    相關資料：
    {context}

    問題：{question}
    """

    # 使用 langchain prompt
    prompt = ChatPromptTemplate.from_template(template)

    # 最後的
    def rag_chain(question: str):
        docs = retriever.invoke(question)
        # 組合檢索結果內容成一個上下文字串
        context = "\n\n".join([d.page_content for d in docs])
        filled_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(filled_prompt)
        return response.content
    
    return rag_chain