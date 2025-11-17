from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from datetime import datetime
import json
import os

app = Flask(__name__)

MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
    return model.config.id2label[int(torch.argmax(probs))]

def save_mood(text, emotion):
    entry = {
        "text": text,
        "emotion": emotion,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    file = os.path.join(os.path.dirname(__file__), "..", "mood_history.json")

    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(entry)
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

# 🎯 NEW: Greetings and Friendly Chat Rules
GREETINGS = {
    "hi": "Hi there! I'm really glad you're here. How are you feeling today?",
    "hello": "Hello! I'm here with you. What’s on your mind?",
    "hey": "Hey! How are you doing right now?",
    "good morning": "Good morning ☀️ I hope today is gentle for you.",
    "good night": "Good night 🌙 Rest is important, you deserve it."
}

# 🎯 NEW: Emotion-based supportive messaging
EMOTION_RESPONSES = {
    "sadness": "I’m really sorry you're feeling this way. Want to talk about what’s hurting you?",
    "anger": "I hear your frustration. You are allowed to feel this. If you want, we can try a calming breath — just say *breath*.",
    "fear": "It’s okay to feel scared. You’re safe here. I'm with you.",
    "joy": "I'm glad you're feeling good! What made this moment nice?",
    "love": "That's warm and meaningful. It's okay to hold onto that feeling.",
    "surprise": "That sounded unexpected. How did it make you feel?",
    "disgust": "That must have felt uncomfortable. You can share if you want.",
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.form["message"].lower().strip()

    # ✅ 1 — Greeting Handling
    for g in GREETINGS:
        if msg.startswith(g):
            return jsonify({"reply": GREETINGS[g]})

    # ✅ 2 — Guided Breathing Command
    if msg == "breath" or msg == "breathe":
        return jsonify({"reply": "Okay, let's do this together:\n\nInhale deeply for 4 seconds... 🌬️\nHold for 4...\nExhale slowly for 6...\nRepeat this 3 times.\n\nYou're doing great."})

    # ✅ 3 — Emotion-Aware Response
    emotion = detect_emotion(msg)
    save_mood(msg, emotion)
    reply = EMOTION_RESPONSES.get(emotion, "I'm here with you. Tell me more when you're ready.")
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
