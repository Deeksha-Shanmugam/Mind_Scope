import pandas as pd
import numpy as np
import re, string
from tqdm import tqdm
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


# -------------------- Load & Clean Data --------------------
df = pd.read_csv("D:/mental-health-monitor/data/Combined Data.csv")
df = df[['statement', 'status']].dropna()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_text'] = df['statement'].apply(clean_text)

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['status'])

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# -------------------- Load FastText --------------------
# Path to downloaded file
fasttext_model = KeyedVectors.load_word2vec_format('D:\mental-health-monitor\models\FastText\wiki-news-300d-1M.vec', binary=False)

# -------------------- Vectorization --------------------
def get_fasttext_vector(text, model, dim=300):
    words = text.split()
    vecs = [model[word] for word in words if word in model]
    if not vecs:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)

X_train_vec = np.vstack([get_fasttext_vector(text, fasttext_model) for text in tqdm(X_train)])
X_test_vec = np.vstack([get_fasttext_vector(text, fasttext_model) for text in tqdm(X_test)])

# -------------------- Train Model --------------------
# clf = LogisticRegression(max_iter=1000)
# clf.fit(X_train_vec, y_train)
# preds = clf.predict(X_test_vec)

# # -------------------- Evaluate --------------------
# print("\nAccuracy:", accuracy_score(y_test, preds))
# print("Classification Report:\n", classification_report(y_test, preds, target_names=le.classes_))


# Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "MLP Classifier": MLPClassifier(class_weight='balanced',hidden_layer_sizes=(128,), max_iter=300, random_state=42)
}


# Train and evaluate each model
for name, model in models.items():
    print(f"\n{name} Model Results:")
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds, target_names=le.classes_))