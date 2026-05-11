from pathlib import Path

ROOT = Path(__file__).parent.parent

SEED_CSV = ROOT / "seed" / "helpdesk_customer_tickets.csv"
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"

TEST_SIZE = 0.2
RANDOM_SEED = 42

QUEUES = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Human Resources",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "General Inquiry",
]

SYSTEM_PROMPT = (
    "You are a support ticket classification agent. "
    "Classify the ticket into exactly one of the following queues:\n"
    + "\n".join(f"- {q}" for q in QUEUES)
    + "\nRespond with only the queue name, nothing else."
)

# Fine-tuning
BASE_MODEL = "mistralai/Mistral-7B-v0.3"
MODEL_DIR = ROOT / "models" / "mistral-finetuned"

# QLoRA
LOAD_IN_4BIT = True
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training
NUM_EPOCHS = 12
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4        # effective batch size = 16
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 1024
WARMUP_STEPS = 36       # 10 % of 360 total steps (480 / (batch 4 × grad_accum 4) × 12 epochs)
