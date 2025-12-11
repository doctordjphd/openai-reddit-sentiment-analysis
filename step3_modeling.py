from pathlib import Path
import os
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
)
import joblib
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import matplotlib.pyplot as plt

CLEAN_PATH = Path("data") / "reddit_openai_clean.parquet"
WITH_VADER_PATH = Path("data") / "reddit_with_vader.parquet"
MODELS_DIR = Path("models")
LOGREG_MODEL_PATH = MODELS_DIR / "logreg_tfidf.joblib"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
DISTILBERT_DIR = MODELS_DIR / "distilbert_openai_sentiment"
MAX_SAMPLES_FOR_BERT = 10000
MAX_SAMPLES_FOR_LOGREG = 50000


def load_clean_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"clean data not found at {path}. run step2_preprocessing.py first."
        )
    df = pd.read_parquet(path)
    print(f"loaded cleaned data: {len(df)} rows, columns: {list(df.columns)}")
    return df


def add_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    print("adding vader sentiment scores/labels...")

    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    def vader_scores(text):
        if not isinstance(text, str):
            text = ""
        return sia.polarity_scores(text)

    scores = df["raw_text"].apply(vader_scores)
    df["vader_neg"] = scores.apply(lambda d: d["neg"])
    df["vader_neu"] = scores.apply(lambda d: d["neu"])
    df["vader_pos"] = scores.apply(lambda d: d["pos"])
    df["vader_compound"] = scores.apply(lambda d: d["compound"])

    def label_from_compound(c):
        if c >= 0.05:
            return "pos"
        elif c <= -0.05:
            return "neg"
        else:
            return "neu"

    df["vader_label"] = df["vader_compound"].apply(label_from_compound)

    print("vader label distribution:")
    print(df["vader_label"].value_counts())

    return df


def train_logreg_tfidf(df: pd.DataFrame):
    print("\n=== training tf–idf + logistic regression ===")

    if len(df) > MAX_SAMPLES_FOR_LOGREG:
        df = df.sample(MAX_SAMPLES_FOR_LOGREG, random_state=42).reset_index(drop=True)
        print(f"subsampled to {len(df)} rows for logistic regression.")

    X = df["clean_text_lemma"]
    y = df["vader_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("fitting tf–idf vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        lowercase=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("fitting logistic regression model...")
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    print("\nlogistic regression classification report:")
    print(classification_report(y_test, y_pred))

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"macro f1 (logreg): {macro_f1:.4f}")

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("confusion matrix (rows=true, cols=pred):")
    print("labels:", labels)
    print(cm)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, LOGREG_MODEL_PATH)
    joblib.dump(vectorizer, TFIDF_PATH)
    print(f"\nsaved logistic regression model to {LOGREG_MODEL_PATH}")
    print(f"saved tf–idf vectorizer to {TFIDF_PATH}")


def train_distilbert(df):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.optim import AdamW

    print("\n=== training distilbert classifier (simple pytorch loop) ===")

    if len(df) > 5000:
        df = df.sample(5000, random_state=42).reset_index(drop=True)
        print(f"subsampled to {len(df)} rows for cpu training.")

    labels = sorted(df["vader_label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df["label_id"] = df["vader_label"].map(label2id)

    class RedditDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = list(texts)
            self.labels = list(labels)
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in encoding.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = RedditDataset(train_df["raw_text"], train_df["label_id"], tokenizer)
    val_dataset = RedditDataset(val_df["raw_text"], val_df["label_id"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(labels),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    EPOCHS = 1

    for epoch in range(EPOCHS):
        print(f"\nepoch {epoch+1}/{EPOCHS}")
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"  training loss: {avg_train_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                labels_batch = batch["labels"].to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels_batch).sum().item()
                total += labels_batch.size(0)

        acc = correct / total
        print(f"  validation accuracy: {acc:.4f}")

    DISTILBERT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(DISTILBERT_DIR)
    tokenizer.save_pretrained(DISTILBERT_DIR)
    print(f"\nsaved distilbert model to {DISTILBERT_DIR}")


def main():
    df = load_clean_data(CLEAN_PATH)
    df = add_vader_sentiment(df)
    WITH_VADER_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WITH_VADER_PATH, index=False)
    print(f"saved data with vader sentiment to {WITH_VADER_PATH}")
    train_logreg_tfidf(df)
    train_distilbert(df)


if __name__ == "__main__":
    main()
