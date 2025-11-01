from core.rag_pipeline import build_rag_pipeline
from core.report_generator import generate_report

def main():
    print("醫療問答 RAG Demo")
    print("以下回覆為 Demo 展示用，請勿視為正式醫療建議")
    print("若有身體健康上的疑問，請務必諮詢專業醫療人員")

    qa_chain = build_rag_pipeline()

    while True:
        question = input("請輸入您的問題： 或輸入 exit / quit 離開: ")
        if question.lower() in ["exit", "quit"]:
            break

        answer = qa_chain(question)
        
        print("\n 回答：")
        print(answer)
        print("\n")


        report = generate_report(question, answer)
        print("\n")
        print("\n")
        print("\n 衛教摘要 demo：")
        print(report)
        print("\n" + "-"*50 + "\n")
        


if __name__ == "__main__":
    main()