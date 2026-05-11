from finetune.predictor import normalise_label


def test_exact_match():
    assert normalise_label("Technical Support") == "Technical Support"


def test_case_insensitive_match():
    assert normalise_label("technical support") == "Technical Support"


def test_strips_leading_trailing_whitespace():
    assert normalise_label("  Billing and Payments  ") == "Billing and Payments"


def test_strips_trailing_punctuation():
    assert normalise_label("Product Support.") == "Product Support"
    assert normalise_label("IT Support!") == "IT Support"


def test_substring_match():
    # Model sometimes generates extra words around the queue name
    assert normalise_label("This is IT Support team") == "IT Support"


def test_fallback_for_unknown_output():
    assert normalise_label("something completely unrecognised xyz") == "General Inquiry"


def test_all_queues_round_trip():
    queues = [
        "Technical Support", "Product Support", "Customer Service",
        "IT Support", "Billing and Payments", "Returns and Exchanges",
        "Human Resources", "Service Outages and Maintenance",
        "Sales and Pre-Sales", "General Inquiry",
    ]
    for q in queues:
        assert normalise_label(q) == q
