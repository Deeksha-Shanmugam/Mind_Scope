import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

# Load data
df = pd.read_csv("D:/mental-health-monitor/data/Combined Data.csv")[['statement', 'status']].dropna()

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['status'])

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['statement'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label']
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize(example):
    return tokenizer(example["text"], truncation=True)

# Prepare Dataset for Hugging Face
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Define model
num_labels = len(set(train_labels))
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",  # Make sure this is supported
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)


# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Predict
preds_output = trainer.predict(test_dataset)
y_pred = preds_output.predictions.argmax(axis=1)
y_true = test_labels

# Evaluation
print("\nTest Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=le.classes_))
