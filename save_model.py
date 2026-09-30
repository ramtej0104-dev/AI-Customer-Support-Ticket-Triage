import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("support_tickets_clean.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

model = LogisticRegression(max_iter=1000)
model.fit(X, df["label"])

joblib.dump(vectorizer, "vectorizer.joblib")
joblib.dump(model, "model.joblib")

print("Model and vectorizer saved.")