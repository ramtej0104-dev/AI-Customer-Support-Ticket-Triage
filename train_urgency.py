import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("support_tickets_urgency.csv")

# Part 1: practice/exam check
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["urgency"],
    test_size=0.2, random_state=42, stratify=df["urgency"]
)

check_vectorizer = TfidfVectorizer()
X_train_tfidf = check_vectorizer.fit_transform(X_train)
X_test_tfidf = check_vectorizer.transform(X_test)

check_model = LogisticRegression(max_iter=1000)
check_model.fit(X_train_tfidf, y_train)

print(classification_report(y_test, check_model.predict(X_test_tfidf)))

# Part 2: train the final model on all tickets and save it
vectorizer = joblib.load("vectorizer.joblib")
X_all = vectorizer.transform(df["text"])

urgency_model = LogisticRegression(max_iter=1000)
urgency_model.fit(X_all, df["urgency"])

joblib.dump(urgency_model, "urgency_model.joblib")
print("Urgency model saved.")