import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

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


def apply_lora(model):
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def train():
    model, tokenizer = load_base_model()
    model = apply_lora(model)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(config.TRAIN_PATH), split="train")
    dataset = dataset.map(
        lambda ex: {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )},
        batched=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        args=TrainingArguments(
            output_dir=str(config.MODEL_DIR),
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
        ),
    )

    trainer.train()
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.MODEL_DIR))
    tokenizer.save_pretrained(str(config.MODEL_DIR))
    print(f"Model saved to {config.MODEL_DIR}")
