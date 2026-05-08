import pandas as pd

import config


def format_example(row: pd.Series) -> dict:
    """Convert a single ticket row into an OpenAI-compatible chat messages dict."""
    user_content = (
        f"Language: {row['language']}\n"
        f"Business type: {row['business_type']}\n"
        f"Subject: {row['subject']}\n"
        f"Body: {row['body']}"
    )
    return {
        "messages": [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row["queue"]},
        ]
    }


def format_dataset(df: pd.DataFrame) -> list[dict]:
    """Format all rows of a DataFrame into a list of chat messages dicts."""
    return [format_example(row) for _, row in df.iterrows()]
