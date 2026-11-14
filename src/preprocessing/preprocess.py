import pandas as pd
from src.preprocessing.cleaner import clean_dataframe
from src.preprocessing.tokenizer import tokenize_dataframe
from src.preprocessing.vectorizer import build_tfidf
from src.preprocessing.feature_store import cache_exists, save_to_cache, load_from_cache


def run_preprocessing():
    print("── Phase 2: Preprocessing ──\n")

    # load raw data
    train_df = pd.read_parquet("data/raw/ag_news_train.parquet")
    test_df = pd.read_parquet("data/raw/ag_news_test.parquet")

    # --- cleaning ---
    if cache_exists(train_df, "clean_train"):
        train_df = load_from_cache(train_df, "clean_train")
    else:
        train_df = clean_dataframe(train_df)
        save_to_cache(train_df, train_df, "clean_train")

    if cache_exists(test_df, "clean_test"):
        test_df = load_from_cache(test_df, "clean_test")
    else:
        test_df = clean_dataframe(test_df)
        save_to_cache(test_df, test_df, "clean_test")

    # --- tokenization ---
    if cache_exists(train_df, "token_train"):
        train_df = load_from_cache(train_df, "token_train")
    else:
        train_df = tokenize_dataframe(train_df)
        save_to_cache(train_df, train_df, "token_train")

    if cache_exists(test_df, "token_test"):
        test_df = load_from_cache(test_df, "token_test")
    else:
        test_df = tokenize_dataframe(test_df)
        save_to_cache(test_df, test_df, "token_test")

    # --- TF-IDF ---
    build_tfidf(train_df, test_df)

    print("\n── Preprocessing Complete ✓ ──")


if __name__ == "__main__":
    run_preprocessing()
