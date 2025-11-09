import json
from collections import Counter
import matplotlib.pyplot as plt

def load_moods():
    try:
        with open("mood_history.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("No mood history found.")
        return

    emotions = [entry["emotion"] for entry in data]
    counts = Counter(emotions)

    # Show text summary
    print("\nYour Mood Summary:")
    for emotion, count in counts.items():
        print(f" - {emotion}: {count} times")
    
    # Create bar chart
    emotions_list = list(counts.keys())
    frequency_list = list(counts.values())

    plt.bar(emotions_list, frequency_list)
    plt.xlabel("Emotion")
    plt.ylabel("Frequency")
    plt.title("Your Mood Frequency Chart")
    plt.show()

load_moods()
