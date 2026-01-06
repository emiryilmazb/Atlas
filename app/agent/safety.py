CRITICAL_KEYWORDS = [
    "visa",
    "sponsorship",
    "legally authorized",
    "right to work",
    "citizenship",
    "nationality",
    "disability",
    "criminal",
    "conviction",
    "gender",
    "marital",
]


def is_critical_question(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in CRITICAL_KEYWORDS)
