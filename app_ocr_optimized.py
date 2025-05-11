import os
import pathlib
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
from langchain_openai import OpenAIEmbeddings  # 新しい構文
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# 環境変数読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# パス定義
pdf_path = "data/Document_240323_143426.pdf"  # ←ここをあなたのPDFファイル名に合わせて変更！
index_dir = "faiss_index"

# ベクトルDBが既にあるかどうかで処理分岐
if pathlib.Path(index_dir).exists():
    print("🚀 保存済みのベクトルDBを読み込みます...")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
else:
    print("🔄 PDFを画像に変換中...")
    images = convert_from_path(pdf_path)

    print("🧠 OCRでテキスト抽出中...")
    full_text = ""
    print("✅ 画像ごとのOCRテキスト出力：")
    for image in images:
        text = pytesseract.image_to_string(image, lang='eng')
        print("----- 1ページ分のテキスト -----")
        print(text)
        full_text += text

    print("📄 OCR全文（先頭1000文字）:\n", full_text[:1000])
    for page_num, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    print("📄 OCRテキスト結果の一部:\n", full_text[:1000])

    print("📄 テキストを分割中...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents([Document(page_content=full_text)])

    print("🔍 ベクトルDBを作成しています...")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)

    print("💾 ベクトルDBを保存中...")
    db.save_local(index_dir)

# QAチェーン構築
print("⚙️ QAチェーンを構築中...")
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
    retriever=retriever,
    chain_type="stuff"
)
print("✅ チェーン構築完了。日本語で質問してください（終了するには 'exit'）")

# 質問ループ
while True:
    query = input("🗨 質問: ")
    if query.lower() in ["exit", "quit", "終了"]:
        print("👋 終了します。")
        break
    answer = qa.run(f"日本語で回答してください。質問: {query}")
    print(f"💬 回答: {answer}\n")
