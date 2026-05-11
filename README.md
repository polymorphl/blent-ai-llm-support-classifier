# LLM Fine-tuning for Multilingual Support Ticket Classification

An open-weight LLM fine-tuned to automatically classify customer support requests across multiple languages and business sectors.

## Context

A large support-outsourcing company handles thousands of tickets daily on behalf of dozens of client companies, in four languages (French, English, German, Portuguese). Topics range from refund requests to technical issues to product questions.

Currently, human agents read and manually categorize every incoming message. The growing volume makes this approach unsustainable in terms of response time and scalability.

This project fine-tunes an open-weight LLM (Llama, Mistral, etc.) to automatically assign each ticket to the correct support queue, routing it to the right team faster and improving the end-customer experience.

## Data

The dataset contains 600 labelled support tickets in multiple languages. Only the following columns are used:

| Column | Description |
|---|---|
| `subject` | Request title, written by the user |
| `body` | Request body, written by the user |
| `queue` | Category assigned by a human support operator *(target label)* |
| `language` | Language of the request |
| `business_type` | Business sector of the client company |

## Development steps

### 1. Dataset construction

- Build instruction-tuned prompts from the five allowed columns
- Stratified train/test split that preserves the original class distribution

**Output format.** Each split is serialised as a JSONL file (`data/train.jsonl`, `data/test.jsonl`) where every line is an OpenAI-compatible chat example:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a support ticket classification agent …"},
    {"role": "user",      "content": "Language: French\nBusiness type: E-commerce\nSubject: …\nBody: …"},
    {"role": "assistant", "content": "Returns and Exchanges"}
  ]
}
```

**Design decisions.**

| Decision | Rationale |
|---|---|
| Stratified split on `queue` (80 / 20) | Preserves per-class frequency so the test set reflects the real distribution across all 10 queues. |
| All four metadata fields in the user turn | `language` and `business_type` are strong priors for routing; including them at inference time lets the model generalise across client sectors without leaking label information. |
| Assistant turn = bare queue label | Minimises the generation target to a fixed vocabulary, which reduces training loss noise and makes decoding deterministic. |
| Fixed `random_seed = 42` | Ensures reproducible splits for fair comparison between the base model and the fine-tuned model. |

### 2. LLM fine-tuning

- Base model: **Mistral-7B-v0.3**
- Adaptation method: **SFT + QLoRA** (4-bit quantisation via bitsandbytes, LoRA r=16 α=32 on all attention + MLP projection layers)
- Training library: **Unsloth** + TRL `SFTTrainer` (2× faster training, ~50 % less VRAM)
- Hardware: NVIDIA GPU with ≥ 24 GB VRAM
- Target: weighted F1-score ≥ 92 % on the held-out test set

**Design decisions.**

| Decision | Rationale |
|---|---|
| QLoRA (4-bit) over full LoRA | Fits Mistral-7B in 24 GB with room for batch size 4; quality gap is negligible on 10 well-separated classes. |
| Target modules include MLP (gate/up/down) | Classification tasks benefit from fine-tuning feed-forward layers, not just attention. |
| `max_new_tokens=10` at inference | Longest label ("Service Outages and Maintenance") is 4 tokens; capping generation avoids runaway output and makes decoding deterministic. |
| Label normalisation fallback | Strips whitespace/punctuation, tries exact then substring match against the 10 queues, falls back to "General Inquiry" to protect evaluation from rare malformed outputs. |


### 3. Evaluation and comparison

Compute the **weighted F1-score** for both the base model and the fine-tuned model on the same test set:

$$F1_j = \frac{2 \times TP_j}{2 \times TP_j + FP_j + FN_j}$$

$$F1 = \sum_{j=1}^{|C|} \alpha_j \, F1_j \qquad \alpha_j = \frac{n_j}{n}$$

**Minimum required threshold: weighted F1-score ≥ 92 % (fine-tuned model).**


## Getting started

```bash
uv sync
```

### Step 1 — Build the dataset

```bash
uv run python -m src.main --step dataset
```

### Step 2 — Fine-tune (GPU sandbox required)

```bash
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
uv run python -m src.main --step finetune
```

> **Note:** Unsloth dependency resolution (torch, triton, xformers…) can take 5-10 minutes on first install

Training time: ~10-15 minutes on a 24 GB GPU.

### Step 3 — Evaluate base vs fine-tuned

```bash
uv run python -m src.main --step evaluate
```

Results are printed to the console and saved to `data/evaluation_results.json`.

## Tests

```bash
uv run pytest
```