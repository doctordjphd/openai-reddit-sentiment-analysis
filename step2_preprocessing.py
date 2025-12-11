import re
from pathlib import Path

import pandas as pd
import spacy

RAW_PATH = Path("data") / "raw_reddit_openai_from_kaggle.parquet"
OUT_PATH = Path("data") / "reddit_openai_clean.parquet"

MIN_LEN_CHARS = 5

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("[deleted]", " ").replace("[removed]", " ")
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    text = text.strip()
    return text


def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        raise SystemExit(
            "spaCy model 'en_core_web_sm' is not installed.\n"
            "install it with:\n"
            "    python -m spacy download en_core_web_sm"
        )
    return nlp


def lemmatize_series(texts: pd.Series, nlp: spacy.language.Language) -> pd.Series:
    lemmas = []
    for doc in nlp.pipe(texts.tolist(), batch_size=512):
        tokens = [token.lemma_ for token in doc if token.is_alpha]
        lemmas.append(" ".join(tokens))
    return pd.Series(lemmas, index=texts.index)


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"expected input file not found: {RAW_PATH}\n"
            "run step1_data_collection.py or step1_fix_from_csv.py first."
        )

    print(f"loading raw data from {RAW_PATH} ...")
    df = pd.read_parquet(RAW_PATH)
    print(f"loaded {len(df)} rows, columns: {list(df.columns)}")

    print("combining title + selftext into 'raw_text' ...")
    df["title"] = df["title"].fillna("")
    df["selftext"] = df["selftext"].fillna("")
    df["raw_text"] = (df["title"].astype(str) + " " + df["selftext"].astype(str)).str.strip()

    print("applying basic cleaning ...")
    df["clean_text_basic"] = df["raw_text"].apply(basic_clean)

    before = len(df)
    df = df[df["clean_text_basic"].str.len() >= MIN_LEN_CHARS].copy()
    after = len(df)
    print(f"dropped {before - after} very short/empty rows after basic cleaning; remaining: {after}")

    print("loading spacy model (en_core_web_sm) ...")
    nlp = load_spacy_model()

    print("lemmatizing text ...")
    df["clean_text_lemma"] = lemmatize_series(df["clean_text_basic"], nlp)

    before = len(df)
    df = df[df["clean_text_lemma"].str.len() >= MIN_LEN_CHARS].copy()
    after = len(df)
    print(f"dropped {before - after} rows after lemmatization; remaining: {after}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"saved cleaned data with lemmatization to {OUT_PATH}")
    print("final row count:", len(df))
    print("\nsample rows:")
    print(df[["raw_text", "clean_text_basic", "clean_text_lemma"]].head(5))


if __name__ == "__main__":
    main()
