import os
import pathlib
import streamlit as st
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# 環境変数読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# フォルダ指定
pdf_path = "data/your_file.pdf"  # ←あなたのPDFファイル名に変更
index_dir = "faiss_index"

# ベクトルDBの読み込み or 初期構築
@st.cache_resource(show_spinner="ベクトルDBをロード中...")
def load_vector_db():
    if pathlib.Path(index_dir).exists():
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        return db
    else:
        st.info("初回起動：PDFをOCRし、ベクトルDBを構築中...")
        images = convert_from_path(pdf_path)
        full_text = ""
        for page_num, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents([Document(page_content=full_text)])
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(index_dir)
        return db

# アプリ構成
st.set_page_config(page_title="RAGチャットボット", page_icon="🤖")
st.title("🧬 OCR対応RAGチャットボット")

# ユーザー入力
user_input = st.text_input("💬 質問を入力してください", placeholder="この遺伝子検査で異常のある項目は？")

# チャット処理
if user_input:
    with st.spinner("回答を生成中..."):
        db = load_vector_db()
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
            retriever=retriever,
            chain_type="stuff"
        )
        response = qa.run(f"日本語で回答してください。質問: {user_input}")
        st.markdown(f"**🧠 回答：**\n\n{response}")
