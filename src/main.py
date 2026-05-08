import json
import sys
from pathlib import Path

# Ensure src/ is on sys.path so dataset.* and config are importable
# whether this file is run as `python -m src.main` or `python src/main.py`
sys.path.insert(0, str(Path(__file__).parent))

import config
from dataset.formatter import format_dataset
from dataset.loader import load_tickets
from dataset.splitter import split


def main():
    df = load_tickets(config.SEED_CSV)
    train_df, test_df = split(df, test_size=config.TEST_SIZE, random_seed=config.RANDOM_SEED)

    config.DATA_DIR.mkdir(exist_ok=True)

    for path, subset_df in [(config.TRAIN_PATH, train_df), (config.TEST_PATH, test_df)]:
        examples = format_dataset(subset_df)
        with open(path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"train: {len(train_df)} examples → {config.TRAIN_PATH}")
    print(f"test:  {len(test_df)} examples → {config.TEST_PATH}")


if __name__ == "__main__":
    main()
