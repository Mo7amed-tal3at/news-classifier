import os
import hashlib
import joblib
import pandas as pd


CACHE_DIR = "data/features/.cache"


def _get_cache_key(df: pd.DataFrame, step: str) -> str:
    """
    Generate a hash key based on dataframe content + step name.
    Same data + same step = same key = use cache.
    """
    content_hash = hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:12]
    return f"{step}_{content_hash}"


def cache_exists(df: pd.DataFrame, step: str) -> bool:
    key = _get_cache_key(df, step)
    return os.path.exists(os.path.join(CACHE_DIR, f"{key}.joblib"))


def save_to_cache(obj, df: pd.DataFrame, step: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _get_cache_key(df, step)
    path = os.path.join(CACHE_DIR, f"{key}.joblib")
    joblib.dump(obj, path)
    print(f"[FeatureStore] Cached: {key}")


def load_from_cache(df: pd.DataFrame, step: str):
    key = _get_cache_key(df, step)
    path = os.path.join(CACHE_DIR, f"{key}.joblib")
    print(f"[FeatureStore] Cache hit: {key}")
    return joblib.load(path)
