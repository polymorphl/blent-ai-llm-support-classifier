import argparse
import json
import os
import sys
from pathlib import Path

# Must be set before any torch import to ensure the GPU is visible
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Cache HF model weights in the workspace so they survive container restarts
os.environ.setdefault("HF_HOME", str(Path(__file__).parent.parent / ".hf_cache"))

sys.path.insert(0, str(Path(__file__).parent))

import config
from dataset.formatter import format_dataset
from dataset.loader import load_tickets
from dataset.splitter import split


def step_dataset():
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


def step_finetune():
    from finetune.trainer import train
    train()


def step_evaluate():
    import torch
    from peft import PeftModel
    from finetune.trainer import load_base_model
    from finetune.evaluator import evaluate

    print("=== Evaluating base model (zero-shot) ===")
    base_model, tokenizer = load_base_model()
    base_results = evaluate(base_model, tokenizer)
    print(f"Base weighted F1: {base_results['weighted_f1']:.4f}")
    del base_model
    torch.cuda.empty_cache()

    # Collect checkpoints sorted by step, plus the final saved model
    checkpoints = sorted(
        config.MODEL_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    candidates = checkpoints + [config.MODEL_DIR]

    print(f"\n=== Evaluating {len(candidates)} checkpoint(s) ===")
    checkpoint_results = {}
    for ckpt_path in candidates:
        label = ckpt_path.name if ckpt_path != config.MODEL_DIR else "final"
        ft_base, ft_tokenizer = load_base_model()
        ft_model = PeftModel.from_pretrained(ft_base, str(ckpt_path))
        result = evaluate(ft_model, ft_tokenizer)
        f1 = result["weighted_f1"]
        checkpoint_results[label] = result
        print(f"  {label:<30} F1={f1:.4f}")
        del ft_model, ft_base
        torch.cuda.empty_cache()

    best_label = max(checkpoint_results, key=lambda k: checkpoint_results[k]["weighted_f1"])
    best_f1 = checkpoint_results[best_label]["weighted_f1"]

    print(f"\n=== Summary ===")
    print(f"{'Model':<35} {'Weighted F1':>12}")
    print("-" * 48)
    print(f"{'Mistral-7B base':<35} {base_results['weighted_f1']:>12.4f}")
    for label, result in checkpoint_results.items():
        marker = " ← best" if label == best_label else ""
        print(f"{'Mistral-7B QLoRA ' + label:<35} {result['weighted_f1']:>12.4f}{marker}")

    output = {"base": base_results, "checkpoints": checkpoint_results, "best": best_label}
    results_path = config.DATA_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nBest: {best_label} (F1={best_f1:.4f})")
    print(f"Detailed results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="LLM Support Classifier")
    parser.add_argument(
        "--step",
        choices=["dataset", "finetune", "evaluate"],
        default="dataset",
        help="Pipeline step to run (default: dataset)",
    )
    args = parser.parse_args()

    if args.step == "dataset":
        step_dataset()
    elif args.step == "finetune":
        step_finetune()
    elif args.step == "evaluate":
        step_evaluate()


if __name__ == "__main__":
    main()
