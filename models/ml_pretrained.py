import pandas as pd
import numpy as np
import re
import string
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------- Clean Text --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|rt", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------- Load Data --------------------
df = pd.read_csv("D:/mental-health-monitor/data/Combined Data.csv")
df = df[['statement', 'status']].dropna()
df['clean_text'] = df['statement'].apply(clean_text)

# -------------------- Encode Labels --------------------
le = LabelEncoder()
df['label'] = le.fit_transform(df['status'])

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# -------------------- Load GloVe Embeddings --------------------
def load_glove_model(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

glove_model = load_glove_model("D:\mental-health-monitor\models\GloVe\glove.6B.300d.txt")
embedding_dim = 300

# -------------------- Average Word Embeddings --------------------
def get_average_vector(text, embeddings_index, dim):
    words = text.split()
    vectors = [embeddings_index[w] for w in words if w in embeddings_index]
    if len(vectors) == 0:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)

X_train_vec = np.vstack([get_average_vector(t, glove_model, embedding_dim) for t in tqdm(X_train, desc="Vectorizing Train")])
X_test_vec = np.vstack([get_average_vector(t, glove_model, embedding_dim) for t in tqdm(X_test, desc="Vectorizing Test")])

# Assume tfidf_vecs = TF-IDF vectors (sparse), glove_vecs = GloVe average vectors (dense)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_combined = hstack([tfidf.fit_transform(X_train), X_train_vec])
X_test_combined = hstack([tfidf.transform(X_test), X_test_vec])

# -------------------- Train Classifier --------------------
clf = LogisticRegression(max_iter=1000)
# clf.fit(X_train_vec, y_train)
# preds = clf.predict(X_test_vec)
clf.fit(X_train_combined, y_train)
preds = clf.predict(X_test_combined)

# -------------------- Evaluate --------------------
print("\nAccuracy:", accuracy_score(y_test, preds))
print("Classification Report:\n", classification_report(y_test, preds, target_names=le.classes_))

import os
import joblib

# Ensure the 'models' directory exists
os.makedirs("models", exist_ok=True)

# Now save your artifacts
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(glove_model, "models/glove_embeddings.pkl")
joblib.dump(clf, "models/final_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")