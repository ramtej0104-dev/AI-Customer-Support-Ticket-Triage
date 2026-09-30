import joblib
import csv
import os
from datetime import datetime

vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")
urgency_model = joblib.load("urgency_model.joblib")

THRESHOLD = 0.60
URGENT_WORDS = ["urgent", "asap", "immediately", "emergency", "right now"]
LOG_FILE = "flagged_tickets.csv"

def log_flagged_ticket(ticket_text, category, confidence, urgency):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "text", "category", "confidence", "urgency"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            ticket_text,
            category,
            f"{confidence:.2f}",
            urgency,
        ])

def triage(ticket_text):
    numbers = vectorizer.transform([ticket_text])

    probabilities = model.predict_proba(numbers)[0]
    best = probabilities.argmax()
    category = model.classes_[best]
    confidence = probabilities[best]

    urgency = urgency_model.predict(numbers)[0]

    if any(word in ticket_text.lower() for word in URGENT_WORDS):
        urgency = "high"

    if confidence < THRESHOLD:
        status = "NEEDS HUMAN REVIEW"
        log_flagged_ticket(ticket_text, category, confidence, urgency)
    else:
        status = "auto-routed"

    return category, confidence, urgency, status

if __name__ == "__main__":
    tests = [
        "I was charged twice for my subscription",
        "I want to talk to a human agent",
        "urgent, I can't log in to my account",
    ]
    for t in tests:
        category, confidence, urgency, status = triage(t)
        print(f"{t} -> {category} ({confidence:.0%}), urgency: {urgency} [{status}]")