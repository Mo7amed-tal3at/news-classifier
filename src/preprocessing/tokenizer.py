import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def tokenize_and_lemmatize(text: str) -> str:
    """
    Tokenize, lemmatize, and remove stopwords + punctuation.

    Why lemmatization over stemming?
    - Stemming is faster but produces non-words ("running" → "run" vs "runn")
    - Lemmatization preserves real words — better for TF-IDF vocabulary quality
    - With 120k articles the speed difference is acceptable
    """
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.lemma_) > 2
    ]
    return " ".join(tokens)


def tokenize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply tokenization to clean_text column."""
    df = df.copy()

    print("[Tokenizer] Tokenizing... (this may take a few minutes)")

    # pipe is faster than apply for spaCy
    texts = list(df["clean_text"])
    tokenized = []

    for doc in nlp.pipe(texts, batch_size=512):
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.lemma_) > 2
        ]
        tokenized.append(" ".join(tokens))

    df["tokenized_text"] = tokenized
    print(f"[Tokenizer] ✓ Tokenized {len(df):,} articles.")
    return df
