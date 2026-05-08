import pandas as pd
import pytest

from dataset.loader import load_tickets

from dataset.loader import REQUIRED_COLUMNS


def test_returns_dataframe(tmp_path):
    csv = tmp_path / "tickets.csv"
    csv.write_text(
        "id,subject,body,queue,language,business_type,extra\n"
        "1,Hello,World,Technical Support,en,Tech Online Store,ignored\n"
    )
    df = load_tickets(csv)
    assert isinstance(df, pd.DataFrame)


def test_keeps_only_required_columns(tmp_path):
    csv = tmp_path / "tickets.csv"
    csv.write_text(
        "id,subject,body,queue,language,business_type,extra\n"
        "1,Hello,World,Technical Support,en,Tech Online Store,ignored\n"
    )
    df = load_tickets(csv)
    assert set(df.columns) == set(REQUIRED_COLUMNS)


def test_raises_on_missing_column(tmp_path):
    csv = tmp_path / "tickets.csv"
    csv.write_text("id,subject,body,queue,language\n1,Hello,World,Technical Support,en\n")
    with pytest.raises(ValueError, match="business_type"):
        load_tickets(csv)


def test_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tickets(tmp_path / "nonexistent.csv")
