import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("D:\mental-health-monitor\data\Combined Data.csv")  # change name if needed
df = df[['statement', 'status']].dropna()

# Text Cleaning Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|RT", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply text cleaning
df['clean_text'] = df['statement'].apply(clean_text)

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['status'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Models to compare
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n{name} Model Results:")
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds, target_names=le.classes_))


#version 2
# import pandas as pd
# import re
# import string
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# # Simple text cleaner without nltk
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)  # remove links and mentions
#     text = re.sub(r"[{}]".format(string.punctuation), " ", text)  # remove punctuation
#     text = re.sub(r"\d+", "", text)  # remove numbers
#     text = re.sub(r"\s+", " ", text).strip()  # remove extra whitespace
#     return text

# # Load data
# df = pd.read_csv("D:\mental-health-monitor\data\Combined Data.csv")
# df = df[['statement', 'status']].dropna()
# df['clean_text'] = df['statement'].apply(clean_text)

# # Encode labels
# le = LabelEncoder()
# df['label'] = le.fit_transform(df['status'])

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# # TF-IDF vectorization
# tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
# X_train_vec = tfidf.fit_transform(X_train)
# X_test_vec = tfidf.transform(X_test)

# # Grid search on Logistic Regression
# param_grid = {
#     'C': [0.01, 0.1, 1, 10],
#     'solver': ['lbfgs', 'saga'],
#     'penalty': ['l2']
# }
# grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
# grid.fit(X_train_vec, y_train)

# best_model = grid.best_estimator_
# preds = best_model.predict(X_test_vec)

# # Results
# print("Best Parameters:", grid.best_params_)
# print("Improved Accuracy:", accuracy_score(y_test, preds))
# print("Classification Report:\n", classification_report(y_test, preds, target_names=le.classes_))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

# Collect results
results = []

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    results.append({'Model': name, 'Accuracy': accuracy, 'Macro F1-score': f1})

# Convert to DataFrame
results_df = pd.DataFrame(results).sort_values(by="Macro F1-score", ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='Macro F1-score', y='Model', data=results_df, palette='viridis')
plt.title("Model Comparison based on Macro F1-score")
plt.xlabel("Macro F1-score")
plt.xlim(0, 1)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Optionally: Accuracy plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=results_df, palette='magma')
plt.title("Model Comparison based on Accuracy")
plt.xlabel("Accuracy")
plt.xlim(0, 1)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
