import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Clean raw news text.
    Order matters — each step builds on the previous.
    """
    if not isinstance(text, str):
        return ""

    # 1. lowercase
    text = text.lower()

    # 2. remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 4. remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 5. keep only letters and spaces — remove numbers and special chars
    text = re.sub(r"[^a-z\s]", " ", text)

    # 6. collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning to title column.
    AG News text field contains title + description combined.
    """
    df = df.copy()
    df["clean_text"] = df["title"].apply(clean_text)

    # drop empty rows after cleaning
    before = len(df)
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"[Cleaner] Dropped {before - after} empty/short rows.")

    print(f"[Cleaner] ✓ Cleaned {after:,} articles.")
    return df
