from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import os
from dotenv import load_dotenv

# LangChain RAG用
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# .env 読み込み
load_dotenv()

app = Flask(__name__)

# 環境変数からLINE & OpenAI設定を取得
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# FAISSベクトルDBを読み込む
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Webhookエンドポイント
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

# メッセージ受信時の挙動（RAGで応答）
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    print(f"🔍 質問を受信: {user_message}")

    try:
        answer = qa.run(f"日本語で回答してください。質問: {user_message}")
        print(f"💬 返答: {answer}")
    except Exception as e:
        answer = "申し訳ありません。情報を取得できませんでした。"
        print(f"❌ エラー: {e}")

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=answer)
    )

# Flaskサーバー起動
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
    