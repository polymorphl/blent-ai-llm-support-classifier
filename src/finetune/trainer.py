from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

import config


def load_base_model():
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.BASE_MODEL,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=config.LOAD_IN_4BIT,
        dtype=None,
    )
    return model, tokenizer


def apply_lora(model):
    from unsloth import FastLanguageModel
    return FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.RANDOM_SEED,
    )


def train():
    model, tokenizer = load_base_model()
    model = apply_lora(model)

    dataset = load_dataset("json", data_files=str(config.TRAIN_PATH), split="train")

    dataset = dataset.map(
        lambda ex: {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )},
        batched=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        args=TrainingArguments(
            output_dir=str(config.MODEL_DIR),
            num_train_epochs=config.NUM_EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRAD_ACCUMULATION,
            learning_rate=config.LEARNING_RATE,
            warmup_ratio=config.WARMUP_RATIO,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        ),
    )

    trainer.train()
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.MODEL_DIR))
    tokenizer.save_pretrained(str(config.MODEL_DIR))
    print(f"Model saved to {config.MODEL_DIR}")
