from pathlib import Path

ROOT = Path(__file__).parent.parent

SEED_CSV = ROOT / "seed" / "helpdesk_customer_tickets.csv"
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"

TEST_SIZE = 0.2
RANDOM_SEED = 42   # dataset split — keep fixed for comparable evaluation
TRAIN_SEED = 42    # training seed — tune this to find a good initialization

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
    "- Technical Support: external customer-facing technical problems — hardware failures, network or connectivity malfunctions, server errors, application crashes or slowness affecting the customer\n"
    "- Product Support: help with how to use a product — compatibility questions, setup, configuration, feature how-to, or firmware issues with a device or software\n"
    "- Customer Service: dissatisfaction with the service experience — poor support received, delivery delays, wrong or damaged item received, unresolved complaint; NOT about product functionality\n"
    "- IT Support: company's internal IT infrastructure — server configuration, workstation, VPN, account access, hardware provisioning, or internal tool installation\n"
    "- Billing and Payments: invoices, charges, refunds, payment issues\n"
    "- Returns and Exchanges: return or exchange requests for purchased items\n"
    "- Human Resources: HR, payroll, leave, employee administration\n"
    "- Service Outages and Maintenance: the service or platform is currently down, unavailable, or degraded; unexpected outages or scheduled maintenance affecting access\n"
    "- Sales and Pre-Sales: pricing, availability, pre-purchase questions\n"
    "- General Inquiry: use when the request does not clearly fit any other queue — vague requests, miscellaneous questions, or anything ambiguous\n"
    "Respond with only the queue name, nothing else."
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
NUM_EPOCHS = 15
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4        # effective batch size = 16
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 1024
WARMUP_STEPS = 45       # 10 % of 450 total steps (480 / (batch 4 × grad_accum 4) × 15 epochs)
