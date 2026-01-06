from __future__ import annotations

import logging

from app.agent.llm_client import LLMClient
from app.agent.safety import is_critical_question
from app.agent.types import Intent, NormalizedQuestion

logger = logging.getLogger(__name__)

_MOTIVATION_COMPANY_KEYWORDS = {
    "why our company",
    "why this company",
    "why do you want to work here",
    "company",
}
_MOTIVATION_ROLE_KEYWORDS = {
    "why this role",
    "why this position",
    "role",
    "position",
}
_CULTURAL_FIT_KEYWORDS = {
    "values",
    "culture",
    "team",
    "collaboration",
    "work style",
}
_EXPERIENCE_GENERAL_KEYWORDS = {
    "experience",
    "background",
    "career",
    "projects",
}
_EXPERIENCE_TECH_KEYWORDS = {
    "technology",
    "stack",
    "tools",
    "python",
    "java",
    "typescript",
    "sql",
}

_LEGAL_KEYWORDS = {
    "visa",
    "sponsorship",
    "legally authorized",
    "right to work",
    "citizenship",
    "nationality",
}


def normalize_question(question_text: str, llm_client: LLMClient | None) -> NormalizedQuestion:
    if is_critical_question(question_text):
        return NormalizedQuestion(question_text, _classify_critical_intent(question_text))

    rule_intent = _rule_based_intent(question_text)
    if rule_intent != Intent.OPEN_TEXT:
        return NormalizedQuestion(question_text, rule_intent)

    if llm_client is None:
        return NormalizedQuestion(question_text, Intent.OPEN_TEXT)

    llm_intent = _llm_intent(question_text, llm_client)
    return NormalizedQuestion(question_text, llm_intent)


def _classify_critical_intent(question_text: str) -> Intent:
    lowered = question_text.lower()
    if any(keyword in lowered for keyword in _LEGAL_KEYWORDS):
        return Intent.LEGAL_WORK_AUTHORIZATION
    return Intent.SENSITIVE_PERSONAL


def _rule_based_intent(question_text: str) -> Intent:
    lowered = question_text.lower()
    if any(keyword in lowered for keyword in _MOTIVATION_COMPANY_KEYWORDS):
        return Intent.MOTIVATION_COMPANY
    if any(keyword in lowered for keyword in _MOTIVATION_ROLE_KEYWORDS):
        return Intent.MOTIVATION_ROLE
    if any(keyword in lowered for keyword in _CULTURAL_FIT_KEYWORDS):
        return Intent.CULTURAL_FIT
    if any(keyword in lowered for keyword in _EXPERIENCE_TECH_KEYWORDS):
        return Intent.EXPERIENCE_TECH
    if any(keyword in lowered for keyword in _EXPERIENCE_GENERAL_KEYWORDS):
        return Intent.EXPERIENCE_GENERAL
    return Intent.OPEN_TEXT


def _llm_intent(question_text: str, llm_client: LLMClient) -> Intent:
    allowed = ", ".join(intent.value for intent in Intent if intent not in {
        Intent.LEGAL_WORK_AUTHORIZATION,
        Intent.SENSITIVE_PERSONAL,
    })
    prompt = (
        "Map the question to exactly one intent from the allowed list.\n"
        "Return ONLY the intent value.\n"
        f"Allowed intents: {allowed}\n"
        f"Question: {question_text}\n"
        "If unsure, return OPEN_TEXT."
    )
    response = llm_client.generate_text(prompt).strip().upper()
    for intent in Intent:
        if response == intent.value:
            return intent
    logger.warning("LLM intent output invalid: %s", response)
    return Intent.OPEN_TEXT
