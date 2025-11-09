from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import json
from datetime import datetime

MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Emotion → Supportive reply dictionary
RESPONSES = {
    "sadness": (
        "I'm really sorry you're going through this. It’s okay to feel how you feel.\n"
        "You don’t have to handle everything alone.\n"
        "If it feels right, try expressing what part of today felt the heaviest.\n"
        "Or type: journal — I’ll gently guide your thoughts."
    ),

    "anger": (
        "I can sense how overwhelming this feels. Your emotions are valid.\n"
        "Let’s slow the body down, just for a moment.\n"
        "Type: breath — and I’ll walk you through a calming breathing exercise."
    ),

    "fear": (
        "It sounds like things feel uncertain or intense right now.\n"
        "You’re safe here. I’m with you.\n"
        "Try grounding yourself: look around and name 3 things you can see.\n"
        "If you'd like deeper support, type: journal — we can explore gently."
    ),

    "joy": (
        "That’s wonderful to hear. Please take a moment to notice how that feels inside your body.\n"
        "What made this moment meaningful for you?"
    ),

    "love": (
        "Your heart feels warm today — that’s something to treasure.\n"
        "Maybe let that warmth touch someone else today. Even a small message counts."
    ),

    "surprise": (
        "Wow — that was unexpected!\n"
        "How did that moment make you feel emotionally?"
    ),

    "disgust": (
        "That sounds uncomfortable, and it’s okay to feel this way.\n"
        "You can share more when you’re ready. I’m here with patience."
    ),

    "_default": (
        "I’m here with you. You can share as much or as little as you feel comfortable.\n"
        "What’s on your mind right now?"
    )
}


def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
    pred_id = torch.argmax(probs).item()
    return model.config.id2label[pred_id]

def save_mood(text, emotion):
    entry = {
        "text": text,
        "emotion": emotion,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    try:
        with open("mood_history.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    data.append(entry)

    with open("mood_history.json", "w") as f:
        json.dump(data, f, indent=4)

def breathing_exercise():
    print("\n💨 Let's breathe together:\n")
    print("Inhale slowly through your nose for 4 seconds...")
    print("Hold for 4 seconds...")
    print("Exhale gently through your mouth for 6 seconds...")
    print("Repeat this 3 times.\n")
    print("It's okay if your mind wanders — just return to the breath.\n")


def journal_prompt():
    print("\n📝 Let's reflect together.\n")
    print("1) What emotion feels strongest inside you right now?")
    print("2) What do you think triggered this emotion?")
    print("3) If your closest friend felt this way, what would you tell them?")
    print("\nYou can write the answers in your notebook or speak them out loud.\n")


print("\nMindMate AI (Console Version) 💙")
print("Type 'bye' to exit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["bye", "quit", "exit"]:
        print("Bot: Take care. I’m here whenever you need me 🤍")
        break

    if user_input.lower() == "breath":
        breathing_exercise()
        continue

    if user_input.lower() == "journal":
        journal_prompt()
        continue

    emotion = detect_emotion(user_input)
    save_mood(user_input, emotion)
    reply = RESPONSES.get(emotion, RESPONSES["_default"])

    print(f"\n✨ Emotion Detected: {emotion.upper()}")
    print("🤍 Response:")
    print(reply + "\n")
