import json
from pathlib import Path

from sklearn.metrics import classification_report, f1_score

from config import QUEUES, TEST_PATH
from finetune.predictor import predict, predict_clf


def evaluate(model, tokenizer, test_path: Path = TEST_PATH) -> dict:
    y_true, y_pred = [], []

    with open(test_path, encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            messages = example["messages"][:-1]   # system + user only
            label = example["messages"][-1]["content"]
            prediction = predict(model, tokenizer, messages)
            y_true.append(label)
            y_pred.append(prediction)

    weighted_f1 = f1_score(
        y_true, y_pred, labels=QUEUES, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=QUEUES, zero_division=0, output_dict=True
    )

    return {"weighted_f1": weighted_f1, "classification_report": report}


def evaluate_clf(model, tokenizer, test_path: Path = TEST_PATH) -> dict:
    y_true, y_pred = [], []

    with open(test_path, encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            user_content = example["messages"][1]["content"]
            label = example["messages"][-1]["content"]
            prediction = predict_clf(model, tokenizer, user_content)
            y_true.append(label)
            y_pred.append(prediction)

    weighted_f1 = f1_score(
        y_true, y_pred, labels=QUEUES, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=QUEUES, zero_division=0, output_dict=True
    )
    return {"weighted_f1": weighted_f1, "classification_report": report}
