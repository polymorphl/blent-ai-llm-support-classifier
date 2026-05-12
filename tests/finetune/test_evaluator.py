import json
from pathlib import Path
from unittest.mock import patch
from finetune.evaluator import evaluate, evaluate_clf


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def _make_example(label: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "classify"},
            {"role": "user", "content": "some ticket"},
            {"role": "assistant", "content": label},
        ]
    }


def test_perfect_predictions_give_f1_one(tmp_path):
    path = tmp_path / "test.jsonl"
    _write_jsonl(path, [
        _make_example("Technical Support"),
        _make_example("Billing and Payments"),
    ])

    with patch("finetune.evaluator.predict",
               side_effect=["Technical Support", "Billing and Payments"]):
        result = evaluate(None, None, test_path=path)

    assert result["weighted_f1"] == 1.0


def test_all_wrong_predictions_give_f1_zero(tmp_path):
    path = tmp_path / "test.jsonl"
    _write_jsonl(path, [_make_example("Technical Support")])

    with patch("finetune.evaluator.predict", return_value="Billing and Payments"):
        result = evaluate(None, None, test_path=path)

    assert result["weighted_f1"] == 0.0


def test_result_contains_classification_report(tmp_path):
    path = tmp_path / "test.jsonl"
    _write_jsonl(path, [_make_example("Technical Support")])

    with patch("finetune.evaluator.predict", return_value="Technical Support"):
        result = evaluate(None, None, test_path=path)

    assert "classification_report" in result
    assert "weighted_f1" in result


def test_evaluate_clf_perfect_predictions_give_f1_one(tmp_path):
    path = tmp_path / "test.jsonl"
    _write_jsonl(path, [
        _make_example("Technical Support"),
        _make_example("Billing and Payments"),
    ])

    with patch("finetune.evaluator.predict_clf",
               side_effect=["Technical Support", "Billing and Payments"]):
        result = evaluate_clf(None, None, test_path=path)

    assert result["weighted_f1"] == 1.0


def test_evaluate_clf_all_wrong_predictions_give_f1_zero(tmp_path):
    path = tmp_path / "test.jsonl"
    _write_jsonl(path, [_make_example("Technical Support")])

    with patch("finetune.evaluator.predict_clf", return_value="Billing and Payments"):
        result = evaluate_clf(None, None, test_path=path)

    assert result["weighted_f1"] == 0.0


def test_evaluate_clf_result_contains_classification_report(tmp_path):
    path = tmp_path / "test.jsonl"
    _write_jsonl(path, [_make_example("Technical Support")])

    with patch("finetune.evaluator.predict_clf", return_value="Technical Support"):
        result = evaluate_clf(None, None, test_path=path)

    assert "classification_report" in result
    assert "weighted_f1" in result
