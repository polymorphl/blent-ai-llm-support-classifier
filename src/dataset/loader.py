from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["subject", "body", "queue", "language", "business_type"]


def load_tickets(path: str | Path) -> pd.DataFrame:
    """Load the support tickets CSV and return only the allowed columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return df[REQUIRED_COLUMNS].copy()
