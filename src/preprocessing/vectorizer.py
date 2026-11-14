import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp


def build_tfidf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/features",
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
) -> dict:
    """
    Fit TF-IDF on train, transform both train and test.

    Why fit on train only?
    - Prevents data leakage — test vocabulary must be unknown at fit time
    - Simulates real production: new articles use the trained vectorizer

    Why (1,2) ngrams?
    - Unigrams: individual word semantics
    - Bigrams: local context ("stock market" vs "market")
    - Trigrams add noise for news classification — not worth the memory cost
    """
    os.makedirs(output_dir, exist_ok=True)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,  # log(1+tf) — reduces impact of very frequent terms
    )

    print("[Vectorizer] Fitting TF-IDF on training data...")
    X_train = vectorizer.fit_transform(train_df["tokenized_text"])

    print("[Vectorizer] Transforming test data...")
    X_test = vectorizer.transform(test_df["tokenized_text"])

    # save
    sp.save_npz(os.path.join(output_dir, "X_train.npz"), X_train)
    sp.save_npz(os.path.join(output_dir, "X_test.npz"), X_test)
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))

    train_df[["category"]].to_parquet(os.path.join(output_dir, "y_train.parquet"))
    test_df[["category"]].to_parquet(os.path.join(output_dir, "y_test.parquet"))

    print(f"[Vectorizer] ✓ Feature matrix: {X_train.shape}")
    print(f"[Vectorizer] ✓ Saved to {output_dir}/")

    return {
        "vectorizer": vectorizer,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": train_df["category"].values,
        "y_test": test_df["category"].values,
    }
