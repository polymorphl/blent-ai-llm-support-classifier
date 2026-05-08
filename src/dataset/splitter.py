import pandas as pd
from sklearn.model_selection import train_test_split


def split(
    df: pd.DataFrame,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a stratified train/test split on the queue column."""
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df["queue"],
    )
    return train, test
