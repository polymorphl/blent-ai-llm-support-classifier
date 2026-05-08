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

- Base model: an open-weight LLM (Llama / Mistral family)
- Adaptation method: SFT with LoRA / QLoRA (choice left to the engineer)
- Target: weighted F1-score ≥ 92 % on the held-out test set

### 3. Evaluation and comparison

Compute the **weighted F1-score** for both the base model and the fine-tuned model on the same test set:

$$F1_j = \frac{2 \times TP_j}{2 \times TP_j + FP_j + FN_j}$$

$$F1 = \sum_{j=1}^{|C|} \alpha_j \, F1_j \qquad \alpha_j = \frac{n_j}{n}$$

**Minimum required threshold: weighted F1-score ≥ 92 % (fine-tuned model).**


## Getting started

```bash
uv sync
uv run python -m src.main
```

## Tests

```bash
uv run pytest
```