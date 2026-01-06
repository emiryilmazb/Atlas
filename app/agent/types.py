from dataclasses import dataclass
from enum import Enum


class Intent(Enum):
    MOTIVATION_COMPANY = "MOTIVATION_COMPANY"
    MOTIVATION_ROLE = "MOTIVATION_ROLE"
    CULTURAL_FIT = "CULTURAL_FIT"
    EXPERIENCE_TECH = "EXPERIENCE_TECH"
    EXPERIENCE_GENERAL = "EXPERIENCE_GENERAL"
    LEGAL_WORK_AUTHORIZATION = "LEGAL_WORK_AUTHORIZATION"
    SENSITIVE_PERSONAL = "SENSITIVE_PERSONAL"
    OPEN_TEXT = "OPEN_TEXT"


@dataclass(frozen=True)
class NormalizedQuestion:
    raw_text: str
    normalized_intent: Intent


class AnswerLength(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class AnswerFormat(Enum):
    PARAGRAPH = "paragraph"
    BULLETS = "bullets"


class AnswerTone(Enum):
    PROFESSIONAL_FIRST_PERSON = "professional_first_person"


@dataclass(frozen=True)
class AnswerStyle:
    length: AnswerLength
    format: AnswerFormat
    tone: AnswerTone
