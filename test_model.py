from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

text = "I feel very tired and lonely today."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]

# Find the emotion with highest probability
pred_id = int(torch.argmax(probs))
label = model.config.id2label[pred_id]
confidence = float(probs[pred_id])

print(f"\nText: {text}")
print(f"Emotion Detected: {label}")
print(f"Confidence: {confidence:.2f}")
