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
