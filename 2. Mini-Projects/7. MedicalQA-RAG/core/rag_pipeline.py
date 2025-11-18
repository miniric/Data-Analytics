import os

from langchain_community.document_loaders import TextLoader, WebBaseLoader # type: ignore
from langchain_community.vectorstores import InMemoryVectorStore # type: ignore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # type: ignore

from config.settings import OPENAI_API_KEY, OPENAI_MODEL
from core.prompt_templates import medical_qa_prompt

# 從我們的 config 載入 api_key, LLM_model, 和 向量資料庫的位址,
from config.settings import OPENAI_API_KEY, OPENAI_MODEL # VECTOR_DB_PATH

from core.prompt_templates import medical_qa_prompt
import os

# 增加一個簡易的對話紀錄
conversation_history = []


def build_rag_pipeline():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # 透過 loader 載入預先存好的正確醫療資訊
    local_loader = TextLoader("data/medical_facts.txt", encoding="utf-8")
    local_docs = local_loader.load() 
    
    # 暫時不用網路爬蟲
    # bs4_strainer = bs4.SoupStrainer(
    #     class_=("post-title", "post-header", "post-content")
    # )
    # web_loader = WebBaseLoader(
    #     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    #     bs_kwargs={"parse_only": bs4_strainer},
    # )
    # web_docs = web_loader.load()  # 若抓不到內容，這裡會是空 list

    all_docs = local_docs  #+ web_docs

    # 資訊向量化，使用openAI的模型將文字向量化
    embeddings = OpenAIEmbeddings()
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(all_docs)

    # RAG 裡面的 retriever，使用輸入者的關鍵字檢索向量資料庫裡面資訊
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    # LLM
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)



    # 最後透過 RAG 架構將 retreiver llm 串起來
    def rag_chain(question: str):
        docs = retriever.invoke(question)
        
        # 增加歷史對話記憶, 使用 slice 避免使用過多歷史對話紀錄導致 token 爆炸
        history_text = "\n".join(conversation_history[-10:])
        # 組合檢索結果內容成一個上下文字串
        context = "\n\n".join([d.page_content for d in docs])

        # prompt 載入歷史對話
        filled_prompt = medical_qa_prompt.format(
            history=history_text, 
            context=context, 
            question=question,  
            )
        
        response = llm.invoke(filled_prompt)
        answer = response.content
        
        # 新增對話去對話歷史紀錄
        conversation_history.append(f"病人輸入：{question}")
        conversation_history.append(f"虛擬助理回答：{answer}")

        return answer
    
    return rag_chain