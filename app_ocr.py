import os
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# 環境変数の読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# PDFを画像に変換
pdf_path = "data/Document_240323_143426.pdf"
print("🔄 PDFを画像に変換中...")
images = convert_from_path(pdf_path)

# OCRでテキスト抽出
print("🧠 OCRでテキスト抽出中...")
full_text = ""
for page_num, image in enumerate(images):
    text = pytesseract.image_to_string(image)
    full_text += f"\n--- Page {page_num + 1} ---\n{text}"

# テキストを分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents([Document(page_content=full_text)])

# ベクトルデータベース作成
print("🔍 ベクトルDB作成中...")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

# RAG QA チェーンの構築
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
    retriever=retriever,
    chain_type="stuff"
)

print("✅ 準備完了！日本語で質問してください（終了するには 'exit'）")

# ユーザーとの対話ループ
while True:
    query = input("🗨 質問: ")
    if query.lower() in ["exit", "quit", "終了"]:
        print("👋 終了します。")
        break
    answer = qa.run(f"日本語で回答してください。質問: {query}")
    print(f"💬 回答: {answer}\n")
