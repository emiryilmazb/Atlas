from enum import Enum

from app.agent.safety import is_critical_question


class QuestionType(Enum):
    CRITICAL = "CRITICAL"
    MOTIVATIONAL = "MOTIVATIONAL"
    CULTURAL_FIT = "CULTURAL_FIT"
    EXPERIENCE = "EXPERIENCE"
    OPEN_TEXT = "OPEN_TEXT"


_EXPERIENCE_KEYWORDS = {
    "years",
    "experience",
    "technology",
    "degree",
    "education",
    "project",
}
_CULTURAL_FIT_KEYWORDS = {
    "values",
    "culture",
    "team",
    "collaboration",
}
_MOTIVATIONAL_KEYWORDS = {
    "why",
    "motivated",
    "excited",
    "interested",
}


def llm_classify(question_text: str) -> QuestionType:
    lowered = question_text.lower()

    if any(keyword in lowered for keyword in _EXPERIENCE_KEYWORDS):
        return QuestionType.EXPERIENCE
    if any(keyword in lowered for keyword in _CULTURAL_FIT_KEYWORDS):
        return QuestionType.CULTURAL_FIT
    if any(keyword in lowered for keyword in _MOTIVATIONAL_KEYWORDS):
        return QuestionType.MOTIVATIONAL
    return QuestionType.OPEN_TEXT


def classify_question(question_text: str) -> QuestionType:
    if is_critical_question(question_text):
        return QuestionType.CRITICAL
    return llm_classify(question_text)
