# MedicalQA-RAG

這是一個以 **大語言模型 (LLM)** 為核心的簡易 RAG 醫療應用 Demo，展示：
- RAG (Retrieval-Augmented Generation)
- LangChain + Qdrant 整合
- 自動衛教報告摘要生成

## 主要特色
1. 支援知識檢索與語意強化生成
2. 可自由替換 LLM 模型（GPT-4o / Claude / LLaMA）
3. Data 資料夾內，可使用自己的文件資料輔助進行衛教報告生成


## 執行步驟
```bash
cd MedicalQA-RAG
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-xxxx" > .env # 請輸入自己的 API
python app.py