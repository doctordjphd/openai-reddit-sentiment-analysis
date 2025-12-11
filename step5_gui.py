import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_PATH = Path("data") / "reddit_with_vader.parquet"
MODELS_DIR = Path("models")
LOGREG_MODEL_PATH = MODELS_DIR / "logreg_tfidf.joblib"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
DISTILBERT_DIR = MODELS_DIR / "distilbert_openai_sentiment"


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    return df


@st.cache_resource
def load_logreg():
    clf = joblib.load(LOGREG_MODEL_PATH)
    vectorizer = joblib.load(TFIDF_PATH)
    return clf, vectorizer


@st.cache_resource
def load_distilbert():
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


class RedditDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item


def distilbert_predict_labels(texts, tokenizer, model, device, batch_size=32):
    dataset = RedditDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            batch_pred_ids = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds.extend([DISTILBERT_ID2LABEL[i] for i in batch_pred_ids])
    return preds


st.title("Reddit OpenAI Sentiment Analysis")

st.markdown(
    """
This GUI lets you explore how different sentiment models see Reddit comments about OpenAI.

Enter a **keyword** (e.g. `sora`, `gpt-4`, `agents`) to analyze comments containing that term.
"""
)

df = load_data()
clf, vectorizer = load_logreg()
tokenizer, db_model, device = load_distilbert()

labels_order = ["neg", "neu", "pos"]
DISTILBERT_ID2LABEL = {0: "neg", 1: "neu", 2: "pos"}

st.markdown(
    "<h3 style='margin-bottom: 0.5rem;'>Keyword sentiment analysis</h3>",
    unsafe_allow_html=True,
)

keyword = st.text_input("Keyword to search for:", value="gpt")
max_rows = st.slider(
    "Max number of comments to analyze (lower numbers will produce a faster result :)):",
    min_value=100,
    max_value=1000,
    value=500,
    step=100,
)

if st.button("Analyze keyword"):
    if not keyword.strip():
        st.warning("Please enter a non-empty keyword.")
    else:
        mask = df["raw_text"].str.contains(keyword, case=False, na=False)
        subset = df[mask].copy()

        if subset.empty:
            st.write(f"No comments found containing '{keyword}'.")
        else:
            st.write(f"Found {len(subset)} comments containing '{keyword}'.")
            if len(subset) > max_rows:
                subset = subset.sample(max_rows, random_state=42)
                st.write(f"Subsampled to {len(subset)} comments for analysis.")
            vader_dist = (
                subset["vader_label"].value_counts(normalize=True)
                .reindex(labels_order)
                .fillna(0)
            )
            X_sub = subset["clean_text_lemma"]
            X_sub_vec = vectorizer.transform(X_sub)
            logreg_preds = clf.predict(X_sub_vec)
            logreg_dist = (
                pd.Series(logreg_preds)
                .value_counts(normalize=True)
                .reindex(labels_order)
                .fillna(0)
            )
            db_preds = distilbert_predict_labels(
                subset["raw_text"], tokenizer, db_model, device
            )
            db_dist = (
                pd.Series(db_preds)
                .value_counts(normalize=True)
                .reindex(labels_order)
                .fillna(0)
            )
            import altair as alt

            dist_df = pd.DataFrame(
                {
                    "VADER": vader_dist,
                    "LogReg": logreg_dist,
                    "DistilBERT": db_dist,
                }
            )
            dist_df.index.name = "Sentiment"
            label_display = {"neg": "Negative", "neu": "Neutral", "pos": "Positive"}
            dist_df_display = dist_df.copy()
            dist_df_display.index = [label_display[i] for i in dist_df_display.index]

            st.write("Sentiment distribution for comments containing the keyword:")
            st.dataframe((dist_df_display * 100).round(1))
            plot_df = (
                (dist_df * 100)
                .round(1)
                .reset_index()
                .melt(id_vars="Sentiment", var_name="Model", value_name="Percent")
            )

            sentiment_order = ["neg", "neu", "pos"]
            sentiment_names = {"neg": "Negative", "neu": "Neutral", "pos": "Positive"}
            plot_df["SentimentName"] = plot_df["Sentiment"].map(sentiment_names)

            sentiment_colors = alt.Scale(
                domain=["Negative", "Neutral", "Positive"],
                range=["#e74c3c", "#3498db", "#2ecc71"],
            )

            chart = (
                alt.Chart(plot_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Model:N",
                        title="Model",
                        axis=alt.Axis(labelAngle=0),
                    ),
                    xOffset=alt.X("SentimentName:N"),
                    y=alt.Y(
                        "Percent:Q",
                        title="Percent of comments",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    color=alt.Color(
                        "SentimentName:N",
                        scale=sentiment_colors,
                        title="Sentiment",
                    ),
                    tooltip=["Model", "SentimentName", "Percent"],
                )
                .properties(height=350)
            )

            st.altair_chart(chart, use_container_width=True)
