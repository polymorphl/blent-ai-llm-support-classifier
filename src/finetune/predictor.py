import re
from config import QUEUES


def normalise_label(text: str) -> str:
    """Strip punctuation and map raw model output to the nearest valid queue label."""
    text = re.sub(r"[.,!?]+$", "", text.strip())
    for queue in QUEUES:
        if queue.lower() == text.lower():
            return queue
    for queue in QUEUES:
        if queue.lower() in text.lower() or text.lower() in queue.lower():
            return queue
    return "General Inquiry"


def predict(model, tokenizer, messages: list[dict], max_new_tokens: int = 10) -> str:
    """Generate a queue prediction from the Mistral generative model."""
    import torch

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return normalise_label(raw)


def predict_clf(model, tokenizer, user_content: str) -> str:
    """Predict the queue label with the XLM-RoBERTa classifier via argmax over logits."""
    import torch
    from config import ID2LABEL, MAX_SEQ_LENGTH

    inputs = tokenizer(
        user_content,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(-1).item()
    return ID2LABEL[pred_id]
