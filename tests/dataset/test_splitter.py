import pandas as pd

from dataset.splitter import split


def _make_df():
    # 20 rows: 10x "A", 6x "B", 4x "C"
    # Each class has ≥2 members so sklearn stratified split works cleanly
    return pd.DataFrame({
        "subject": ["s"] * 20,
        "body": ["b"] * 20,
        "queue": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "language": ["en"] * 20,
        "business_type": ["Tech"] * 20,
    })


def test_split_sizes():
    df = _make_df()
    train, test = split(df, test_size=0.2, random_seed=42)
    assert len(train) == 16
    assert len(test) == 4


def test_split_is_reproducible():
    df = _make_df()
    train1, test1 = split(df, test_size=0.2, random_seed=42)
    train2, test2 = split(df, test_size=0.2, random_seed=42)
    assert list(train1.index) == list(train2.index)
    assert list(test1.index) == list(test2.index)


def test_split_different_seeds_differ():
    df = _make_df()
    train1, _ = split(df, test_size=0.2, random_seed=42)
    train2, _ = split(df, test_size=0.2, random_seed=0)
    assert list(train1.index) != list(train2.index)


def test_no_overlap_between_train_and_test():
    df = _make_df()
    train, test = split(df, test_size=0.2, random_seed=42)
    assert set(train.index).isdisjoint(set(test.index))


def test_stratification_preserves_majority_class():
    # With 10 "A" out of 20 and test_size=0.2, test should contain "A"
    df = _make_df()
    _, test = split(df, test_size=0.2, random_seed=42)
    assert "A" in test["queue"].values
