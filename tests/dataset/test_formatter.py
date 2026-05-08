import pandas as pd

from dataset.formatter import format_example, format_dataset


def _make_row(**kwargs):
    defaults = {
        "subject": "Test subject",
        "body": "Test body",
        "queue": "Technical Support",
        "language": "en",
        "business_type": "Tech Online Store",
    }
    return pd.Series({**defaults, **kwargs})


def test_format_example_has_three_messages():
    result = format_example(_make_row())
    assert len(result["messages"]) == 3


def test_format_example_roles():
    result = format_example(_make_row())
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_format_example_assistant_is_queue():
    result = format_example(_make_row(queue="Billing and Payments"))
    assert result["messages"][2]["content"] == "Billing and Payments"


def test_format_example_user_contains_fields():
    row = _make_row(
        language="fr",
        business_type="IT Services",
        subject="Mon écran est cassé",
        body="Bonjour, mon écran...",
    )
    user_content = format_example(row)["messages"][1]["content"]
    assert "Language: fr" in user_content
    assert "Business type: IT Services" in user_content
    assert "Subject: Mon écran est cassé" in user_content
    assert "Body: Bonjour, mon écran..." in user_content


def test_format_dataset_returns_list_of_dicts():
    df = pd.DataFrame([_make_row().to_dict(), _make_row(queue="IT Support").to_dict()])
    result = format_dataset(df)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all("messages" in item for item in result)
