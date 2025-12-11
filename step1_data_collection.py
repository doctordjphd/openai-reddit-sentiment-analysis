import os
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

KAGGLE_USERNAME = "djhaskell"
KAGGLE_KEY = "KGAT_5f09003d56cc2390585b1d65ca565856"

os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = KAGGLE_KEY

KAGGLE_DATASET_SLUG = "armitaraz/chatgpt-reddit"
RAW_DOWNLOAD_DIR = Path("data") / "kaggle_raw"
OUTPUT_PARQUET = Path("data") / "raw_reddit_openai_from_kaggle.parquet"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset(slug: str, download_dir: Path) -> None:
    ensure_dir(download_dir)
    api = KaggleApi()
    api.authenticate()
    print(f"downloading kaggle dataset '{slug}' into '{download_dir}'...")
    api.dataset_download_files(
        slug,
        path=str(download_dir),
        unzip=True,
    )
    print("download & unzip complete.")


def find_first_csv(download_dir: Path) -> Path:
    csv_files = list(download_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"no csv files found in {download_dir}. check the dataset contents or adjust this script."
        )
    return csv_files[0]


def normalize_to_reddit_schema(df: pd.DataFrame) -> pd.DataFrame:
    print("original columns:", list(df.columns))

    col_map = {
        "id": ["id", "comment_id", "post_id"],
        "subreddit": ["subreddit"],
        "created_utc": ["created_utc", "created", "timestamp"],
        "title": ["title", "post_title"],
        "selftext": ["selftext", "body", "comment_text", "text"],
        "score": ["score", "ups", "upvotes"],
        "num_comments": ["num_comments", "n_comments", "comments"],
        "permalink": ["permalink", "url", "link"],
    }

    def pick_column(possible_names):
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    id_col = pick_column(col_map["id"])
    sub_col = pick_column(col_map["subreddit"])
    ts_col = pick_column(col_map["created_utc"])
    title_col = pick_column(col_map["title"])
    text_col = pick_column(col_map["selftext"])
    score_col = pick_column(col_map["score"])
    ncom_col = pick_column(col_map["num_comments"])
    link_col = pick_column(col_map["permalink"])

    norm = pd.DataFrame()
    norm["id"] = df[id_col] if id_col else df.index.astype(str)
    norm["subreddit"] = df[sub_col] if sub_col else ""
    norm["created_utc"] = df[ts_col] if ts_col else 0

    if pd.api.types.is_datetime64_any_dtype(norm["created_utc"]):
        norm["created_utc"] = norm["created_utc"].view("int64") // 10**9

    norm["title"] = df[title_col] if title_col else ""
    norm["selftext"] = df[text_col] if text_col else ""
    norm["score"] = df[score_col] if score_col else 0
    norm["num_comments"] = df[ncom_col] if ncom_col else 0
    norm["permalink"] = df[link_col] if link_col else ""

    return norm


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    print(f"saved {len(df)} rows to {path}")


def main():
    download_kaggle_dataset(KAGGLE_DATASET_SLUG, RAW_DOWNLOAD_DIR)
    csv_path = find_first_csv(RAW_DOWNLOAD_DIR)
    print(f"using csv file: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"loaded {len(df_raw)} rows from csv.")
    df_norm = normalize_to_reddit_schema(df_raw)
    print("normalized columns:", list(df_norm.columns))
    save_parquet(df_norm, OUTPUT_PARQUET)


if __name__ == "__main__":
    main()
