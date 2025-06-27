import pandas as pd
import numpy as np
import re
import string
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# -------------------- Clean Text --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|rt", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------- Load and Prepare Data --------------------
df = pd.read_csv("D:/mental-health-monitor/data/Combined Data.csv")
df = df[['statement', 'status']].dropna()
df['clean_text'] = df['statement'].apply(clean_text)

# Encode target
le = LabelEncoder()
df['label'] = le.fit_transform(df['status'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# -------------------- TF-IDF Vectorization --------------------
tfidf = TfidfVectorizer(max_features=5100, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -------------------- Train Passive-Aggressive Classifier --------------------
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
pac.fit(X_train_tfidf, y_train)
y_pred = pac.predict(X_test_tfidf)

# -------------------- Evaluate --------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# -------------------- Save Models --------------------
os.makedirs("models", exist_ok=True)
joblib.dump(pac, "models/pac_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer_pac.pkl")
joblib.dump(le, "models/label_encoder.pkl")
