import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}[INST] {{ message['content'] }}\n\n"
    "{% elif message['role'] == 'user' %}{{ message['content'] }} [/INST]"
    "{% elif message['role'] == 'assistant' %} {{ message['content'] }}{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %} {% endif %}"
)


def load_base_model():
    """Load Mistral base model — used for zero-shot evaluation only."""
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = _CHAT_TEMPLATE
    return model, tokenizer


def train():
    """Fine-tune XLM-RoBERTa-large as a sequence classifier on the training tickets."""
    import transformers
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    transformers.set_seed(config.TRAIN_SEED)

    tokenizer = AutoTokenizer.from_pretrained(config.CLF_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.CLF_MODEL,
        num_labels=len(config.QUEUES),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )

    dataset = load_dataset("json", data_files=str(config.TRAIN_PATH), split="train")

    def preprocess(example):
        user_content = example["messages"][1]["content"]
        encoding = tokenizer(user_content, truncation=True, max_length=config.MAX_SEQ_LENGTH)
        encoding["labels"] = config.LABEL2ID[example["messages"][-1]["content"]]
        return encoding

    dataset = dataset.map(preprocess, remove_columns=["messages"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(config.CLF_MODEL_DIR),
            num_train_epochs=config.NUM_EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRAD_ACCUMULATION,
            learning_rate=config.LEARNING_RATE,
            warmup_steps=config.WARMUP_STEPS,
            lr_scheduler_type="cosine",
            fp16=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            seed=config.TRAIN_SEED,
            data_seed=config.TRAIN_SEED,
        ),
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=dataset,
    )

    trainer.train()
    config.CLF_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.CLF_MODEL_DIR))
    tokenizer.save_pretrained(str(config.CLF_MODEL_DIR))
    print(f"Model saved to {config.CLF_MODEL_DIR}")
