# app_txt_loader.py

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# .envからAPIキーを読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ファイル読み込み
with open("Genetic_Deepreserch.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

print("📄 テキスト読み込み完了")

# チャンクに分割
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([full_text])

# 埋め込み生成
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(docs, embeddings)

# 保存
db.save_local("faiss_index")
print("✅ FAISSベクトルDBに保存完了")