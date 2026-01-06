from __future__ import annotations

from enum import Enum
import logging
import re

from app.agent.llm_client import LLMClient
from app.agent.types import Intent, NormalizedQuestion
from app.storage.memory import MemoryContext

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


_CONFIDENCE_FALLBACK = {
    Intent.MOTIVATION_COMPANY: 0.75,
    Intent.MOTIVATION_ROLE: 0.72,
    Intent.CULTURAL_FIT: 0.7,
    Intent.EXPERIENCE_GENERAL: 0.6,
    Intent.EXPERIENCE_TECH: 0.55,
    Intent.OPEN_TEXT: 0.4,
    Intent.LEGAL_WORK_AUTHORIZATION: 0.0,
    Intent.SENSITIVE_PERSONAL: 0.0,
}


def assess_risk(question_text: str, intent: Intent) -> RiskLevel:
    lowered = question_text.lower()
    if intent in {Intent.LEGAL_WORK_AUTHORIZATION, Intent.SENSITIVE_PERSONAL}:
        return RiskLevel.HIGH
    if any(keyword in lowered for keyword in {"salary", "compensation", "pay", "rate", "visa"}):
        return RiskLevel.HIGH
    if any(keyword in lowered for keyword in {"relocation", "relocate", "move to"}):
        return RiskLevel.MEDIUM
    if intent in {Intent.MOTIVATION_COMPANY, Intent.MOTIVATION_ROLE, Intent.CULTURAL_FIT}:
        return RiskLevel.LOW
    if intent in {Intent.EXPERIENCE_GENERAL, Intent.EXPERIENCE_TECH}:
        return RiskLevel.MEDIUM
    return RiskLevel.MEDIUM


def score_confidence(
    normalized: NormalizedQuestion,
    llm_client: LLMClient | None,
    memory: MemoryContext,
) -> float:
    if llm_client is None:
        return _fallback_score(normalized.normalized_intent)

    try:
        prompt = _build_prompt(normalized, memory)
        response = llm_client.generate_text(prompt)
        logger.info("LLM confidence raw response: %s", response)
        parsed = _parse_confidence(response)
        if parsed is None:
            logger.warning("LLM confidence output invalid: %s", response)
            return _fallback_score(normalized.normalized_intent)
        return parsed
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("LLM confidence error: %s", exc)
        return _fallback_score(normalized.normalized_intent)


def _fallback_score(intent: Intent) -> float:
    return _CONFIDENCE_FALLBACK.get(intent, 0.4)


def _build_prompt(normalized: NormalizedQuestion, memory: MemoryContext) -> str:
    summary = str(memory.cv_profile.get("summary", "")).strip()
    experience = memory.cv_profile.get("experience", [])
    facts = memory.personal_facts
    facts_text = ", ".join(f"{key}={value.get('value', '')}" for key, value in facts.items())
    return (
        "Rate how confidently you can answer the question using ONLY the provided context. "
        "Return ONLY a float between 0.0 and 1.0.\n"
        f"Question: {normalized.raw_text}\n"
        f"CV summary: {summary or 'None provided.'}\n"
        f"Experience: {', '.join(str(item) for item in experience) or 'None provided.'}\n"
        f"Personal facts: {facts_text or 'None provided.'}\n"
        "If the context is insufficient, return 0.0."
    )


def _parse_confidence(response_text: str) -> float | None:
    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", response_text.strip())
    if not match:
        return None
    value = float(match.group(1))
    if 0.0 <= value <= 1.0:
        return value
    return None
