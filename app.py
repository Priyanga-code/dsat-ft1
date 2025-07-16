from flask import Flask, render_template, request
import joblib
from groq import Groq
import os
import requests

# ✅ Load environment variables
os.environ['GROQ_API_KEY'] = os.getenv("groq")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/main", methods=["GET", "POST"])
def main():
    q = request.form.get("q")
    return render_template("main.html")

# =========================
# LLAMA Routes
# =========================
@app.route("/llama", methods=["GET", "POST"])
def llama():
    return render_template("llama.html")

@app.route("/llama_reply", methods=["GET", "POST"])
def llama_reply():
    q = request.form.get("q")
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": q}]
    )
    return render_template("llama_reply.html", r=completion.choices[0].message.content)

# =========================
# DeepSeek Routes
# =========================
@app.route("/deepseek", methods=["GET", "POST"])
def deepseek():
    return render_template("deepseek.html")

@app.route("/deepseek_reply", methods=["GET", "POST"])
def deepseek_reply():
    user_prompt = request.form.get("prompt")
    client = Groq()
    completion_ds = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "user", "content": user_prompt}]
    )
    return render_template("deepseek_reply.html", result=completion_ds.choices[0].message.content)

# =========================
# DBS Prediction Routes
# =========================
@app.route("/dbs", methods=["GET", "POST"])
def dbs():
    return render_template("dbs.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    q = float(request.form.get("q"))
    model = joblib.load("dbs.jl")
    pred = model.predict([[q]])
    return render_template("prediction.html", r=pred)

# =========================
# Telegram Info Page
# =========================
@app.route("/telegram", methods=["GET"])
def telegram_info():
    bot_link = "https://t.me/dsai_trial_bot"
    return render_template("telegram.html", bot_link=bot_link)

# =========================
# Telegram Webhook Setup
# =========================
@app.route("/setup_webhook", methods=["GET"])
def setup_webhook():
    domain_url = 'https://dsat-ft1-z7w5.onrender.com'
    delete_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
    set_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook?url={domain_url}/webhook"

    # Delete previous webhook
    requests.post(delete_url, json={"drop_pending_updates": True})

    # Set new webhook
    response = requests.get(set_url)
    if response.status_code == 200:
        status = "✅ Telegram bot is connected. Try messaging the bot."
    else:
        status = "❌ Failed to connect the Telegram bot."

    return render_template("telegram.html", status=status)

# =========================
# Telegram Webhook Handler
# =========================
@app.route("/webhook", methods=["GET", "POST"])
def telegram_webhook():
    if request.method == "GET":
        return "Telegram webhook endpoint. Use POST to send updates.", 200

    data = request.get_json()

    if not data or "message" not in data:
        return "No valid data", 400

    chat_id = data["message"]["chat"]["id"]
    user_text = data["message"].get("text", "")

    client = Groq()
    response = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "user", "content": user_text}]
    )
    reply_text = response.choices[0].message.content

    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(telegram_url, json={
        "chat_id": chat_id,
        "text": reply_text
    })

    return "OK", 200


# =========================
# Run the app
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
