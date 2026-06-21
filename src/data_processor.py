"""Task 1: EDA helpers and complaint text preprocessing."""

import re
import pandas as pd


TARGET_PRODUCTS = {
    "credit card": "Credit Card",
    "credit card or prepaid card": "Credit Card",
    "personal loans": "Personal Loan",
    "payday loan, title loan, or personal loan": "Personal Loan",
    "checking or savings account": "Savings Account",
    "money transfers": "Money Transfer",
    "money transfer, virtual currency, or money service": "Money Transfer",
}

# Boilerplate phrases commonly prepended by consumers
_BOILERPLATE = re.compile(
    r"(i am writing to (file|submit|report) a complaint.*?[.!]"
    r"|this is a complaint (about|regarding).*?[.!]"
    r"|i would like to (file|report|submit).*?[.!])",
    re.IGNORECASE,
)

_SPECIAL_CHARS = re.compile(r"[^a-z0-9\s.,!?;:()\-']")
_WHITESPACE = re.compile(r"\s+")


def load_dataset(path: str) -> pd.DataFrame:
    """Load the raw CFPB CSV."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def filter_products(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the four target product categories and drop empty narratives."""
    product_col = _find_col(df, ["product"])
    narrative_col = _find_col(df, ["consumer_complaint_narrative", "narrative"])

    df = df.copy()
    df["product_category"] = (
        df[product_col].str.lower().str.strip().map(TARGET_PRODUCTS)
    )
    df = df[df["product_category"].notna()].copy()
    df = df[df[narrative_col].notna() & (df[narrative_col].str.strip() != "")].copy()
    df = df.rename(columns={narrative_col: "narrative"})
    return df.reset_index(drop=True)


def clean_narrative(text: str) -> str:
    """Lowercase, remove boilerplate and special characters, normalise whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _BOILERPLATE.sub("", text)
    # Remove CFPB redaction tokens like XX, XXXX
    text = re.sub(r"\bx{2,}\b", "", text, flags=re.IGNORECASE)
    text = _SPECIAL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning to all narratives; add word_count column."""
    df = df.copy()
    df["clean_narrative"] = df["narrative"].apply(clean_narrative)
    df = df[df["clean_narrative"].str.len() > 20].copy()
    df["word_count"] = df["clean_narrative"].str.split().str.len()
    return df.reset_index(drop=True)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")

