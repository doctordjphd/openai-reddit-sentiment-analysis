from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_PATH = Path("data") / "reddit_with_vader.parquet"
MODELS_DIR = Path("models")
LOGREG_MODEL_PATH = MODELS_DIR / "logreg_tfidf.joblib"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
DISTILBERT_DIR = MODELS_DIR / "distilbert_openai_sentiment"
FIG_DIR = Path("figures")


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


def plot_confusion(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="true label",
        xlabel="predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def eval_logreg(df):
    print("\n=== logistic regression final evaluation ===")
    labels = sorted(df["vader_label"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}

    df_small = df.sample(15000, random_state=42) if len(df) > 15000 else df.copy()
    X = df_small["clean_text_lemma"]
    y = df_small["vader_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    vectorizer = joblib.load(TFIDF_PATH)
    clf = joblib.load(LOGREG_MODEL_PATH)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    print(classification_report(y_test, y_pred))
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print("macro f1 (logreg):", macro_f1)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("confusion matrix:\n", cm)
    plot_confusion(cm, labels, "logreg confusion matrix", FIG_DIR / "cm_logreg.png")


def eval_distilbert(df):
    print("\n=== distilbert final evaluation ===")

    labels = sorted(df["vader_label"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}

    df_small = df.sample(3000, random_state=42).reset_index(drop=True)

    df_small["label_id"] = df_small["vader_label"].map(label2id)
    train_df, test_df = train_test_split(
        df_small, test_size=0.3, random_state=42, stratify=df_small["label_id"]
    )

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device for eval:", device)
    model.to(device)
    model.eval()

    test_dataset = RedditDataset(test_df["raw_text"], test_df["label_id"], tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32)

    preds = []
    trues = []

    with torch.no_grad():
        for batch in test_loader:
            labels_tensor = batch["labels"].to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1)
            preds.extend(batch_preds.cpu().numpy().tolist())
            trues.extend(labels_tensor.cpu().numpy().tolist())

    print(
        classification_report(
            trues,
            preds,
            target_names=labels,
        )
    )
    macro_f1 = f1_score(trues, preds, average="macro")
    print("macro f1 (distilbert):", macro_f1)

    cm = confusion_matrix(trues, preds, labels=list(range(len(labels))))
    plot_confusion(
        cm, labels, "distilbert confusion matrix", FIG_DIR / "cm_distilbert.png"
    )


def summarize_vader(df):
    print("=== vader baseline summary ===")
    print(df["vader_label"].value_counts(normalize=True).round(3))


def main():
    df = pd.read_parquet(DATA_PATH)
    summarize_vader(df)
    eval_logreg(df)
    eval_distilbert(df)


if __name__ == "__main__":
    main()
