from flask import Flask

# Flask アプリケーションを作成
app = Flask(__name__)

# ルートエンドポイント（トップページ）
@app.route("/")
def home():
    return "Hello, this is Flask running!"

# Flask アプリケーションの起動（外部アクセス可能に）
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
