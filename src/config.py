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
    "- Technical Support: external customer-facing issues with network, infrastructure, servers, or IT systems\n"
    "- Product Support: technical assistance using a product or software — setup, configuration, bugs, compatibility, feature questions\n"
    "- Customer Service: dissatisfaction with the service experience — poor support received, delivery delays, wrong or damaged item received, unresolved complaint; NOT about product functionality\n"
    "- IT Support: internal employee requests — workstation, VPN, account access, hardware provisioning\n"
    "- Billing and Payments: invoices, charges, refunds, payment issues\n"
    "- Returns and Exchanges: return or exchange requests for purchased items\n"
    "- Human Resources: HR, payroll, leave, employee administration\n"
    "- Service Outages and Maintenance: the service or platform is currently down, unavailable, or degraded; unexpected outages or scheduled maintenance affecting access\n"
    "- Sales and Pre-Sales: pricing, availability, pre-purchase questions\n"
    "- General Inquiry: use when the request does not clearly fit any other queue — vague requests, miscellaneous questions, or anything ambiguous\n"
    "Respond with only the queue name, nothing else."
)

LABEL2ID = {q: i for i, q in enumerate(QUEUES)}
ID2LABEL = {i: q for i, q in enumerate(QUEUES)}

# Base LLM — zero-shot evaluation only
BASE_MODEL = "mistralai/Mistral-7B-v0.3"

# Classifier fine-tuning (RoBERTa)
CLF_MODEL = "FacebookAI/xlm-roberta-base"
CLF_MODEL_DIR = ROOT / "models" / "xlm-roberta-finetuned"

# Training
NUM_EPOCHS = 20
BATCH_SIZE = 16
GRAD_ACCUMULATION = 1
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512
WARMUP_STEPS = 60       # 10 % of 600 total steps (480 / 16 × 20 epochs)
