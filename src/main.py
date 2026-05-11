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

    print("\n=== Evaluating fine-tuned model ===")
    ft_base, ft_tokenizer = load_base_model()
    ft_model = PeftModel.from_pretrained(ft_base, str(config.MODEL_DIR))
    ft_results = evaluate(ft_model, ft_tokenizer)
    print(f"Fine-tuned weighted F1: {ft_results['weighted_f1']:.4f}")

    print("\n=== Summary ===")
    print(f"{'Model':<25} {'Weighted F1':>12}")
    print("-" * 38)
    print(f"{'Mistral-7B base':<25} {base_results['weighted_f1']:>12.4f}")
    print(f"{'Mistral-7B QLoRA':<25} {ft_results['weighted_f1']:>12.4f}")

    output = {"base": base_results, "finetuned": ft_results}
    results_path = config.DATA_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")


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
