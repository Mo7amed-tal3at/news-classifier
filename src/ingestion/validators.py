import pandas as pd
import numpy as np
from scipy import stats

REQUIRED_COLUMNS = ["title", "description", "category", "source"]
VALID_CATEGORIES = {"World", "Sports", "Business", "Sci/Tech"}


def validate_schema(df: pd.DataFrame) -> None:
    """Check required columns and no nulls in critical fields."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"[Schema] Missing columns: {missing}")

    null_counts = df[["title", "category"]].isnull().sum()
    if null_counts.any():
        raise ValueError(f"[Schema] Null values found:\n{null_counts[null_counts > 0]}")

    unknown = set(df["category"].unique()) - VALID_CATEGORIES
    if unknown:
        raise ValueError(f"[Schema] Unknown categories: {unknown}")

    print("[Schema] ✓ Passed")


def validate_distribution(df: pd.DataFrame, min_samples_per_class: int = 100) -> None:
    """
    Statistical validation of class distribution.

    Uses Chi-Square goodness-of-fit test to check whether
    the observed distribution deviates significantly from
    a uniform distribution across categories.

    H0: distribution is uniform across categories
    H1: distribution is not uniform (imbalance exists)

    We EXPECT to reject H0 only if imbalance is severe.
    """
    counts = df["category"].value_counts()

    # 1. كل category عندها minimum samples؟
    below_min = counts[counts < min_samples_per_class]
    if not below_min.empty:
        raise ValueError(
            f"[Distribution] Categories below minimum ({min_samples_per_class}):\n{below_min}"
        )

    # 2. Chi-Square test — هل في imbalance إحصائي معنوي؟
    n_categories = len(counts)
    expected = np.full(n_categories, len(df) / n_categories)
    chi2, p_value = stats.chisquare(counts.values, f_exp=expected)

    imbalance_ratio = counts.max() / counts.min()

    print(f"[Distribution] Category counts:\n{counts.to_string()}")
    print(f"[Distribution] Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    print(f"[Distribution] Chi-Square: {chi2:.2f}, p-value: {p_value:.4f}")

    # لو الـ p-value أقل من 0.05 والـ imbalance ratio أكبر من 2
    # يبقى في مشكلة حقيقية محتاجين نتعامل معاها
    if p_value < 0.05 and imbalance_ratio > 2.0:
        raise ValueError(
            f"[Distribution] Severe class imbalance detected "
            f"(χ²={chi2:.2f}, p={p_value:.4f}, ratio={imbalance_ratio:.2f}). "
            f"Consider resampling before training."
        )

    print("[Distribution] ✓ Passed")


def run_all_validations(df: pd.DataFrame) -> None:
    """Run full validation suite — call this before saving any data."""
    print("\n── Running Data Validations ──")
    validate_schema(df)
    validate_distribution(df)
    print("── All Validations Passed ✓ ──\n")
