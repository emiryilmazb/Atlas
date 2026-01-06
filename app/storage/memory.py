from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from app.agent.types import Intent
from app.storage.database import get_database

logger = logging.getLogger(__name__)

_DEFAULT_PROFILE = {"summary": "", "experience": [], "skills": [], "education": []}
_DEFAULT_FACTS: dict[str, Any] = {}
_DEFAULT_HISTORY = {"entries": []}


@dataclass
class MemoryContext:
    cv_profile: dict[str, Any]
    personal_facts: dict[str, Any]
    past_answers: dict[str, Any]


def load_memory_context() -> MemoryContext:
    db = get_database()
    cv_profile = db.load_cv_profile() or dict(_DEFAULT_PROFILE)
    personal_facts = _normalize_personal_facts(db.load_personal_facts())
    past_answers = _normalize_answer_history(db.load_answer_history())

    logger.info("Memory read: loading memory context from SQLite")
    return MemoryContext(
        cv_profile=cv_profile,
        personal_facts=personal_facts,
        past_answers=past_answers,
    )


def append_personal_fact(
    intent: Intent,
    value: str,
    source: str,
    confirmed_at: str,
    country: str | None = None,
) -> None:
    db = get_database()
    db.add_personal_fact(
        intent=intent.value,
        value=value,
        source=source,
        confirmed_at=confirmed_at,
        country=country,
    )
    logger.info("Memory write: personal fact saved for intent=%s", intent.value)


def append_past_answer(
    intent: Intent,
    question: str,
    answer: str,
    source: str,
    submitted_at: str,
) -> None:
    db = get_database()
    db.add_answer_history(
        intent=intent.value,
        question=question,
        answer=answer,
        source=source,
        submitted_at=submitted_at,
        pending=False,
    )
    logger.info("Memory write: answer history saved for intent=%s", intent.value)


def get_personal_fact(personal_facts: dict[str, Any], intent: Intent) -> dict[str, Any] | None:
    payload = personal_facts.get(intent.value)
    if isinstance(payload, dict) and payload.get("value"):
        return payload
    return None


def get_history_answer(past_answers: dict[str, Any], intent: Intent) -> dict[str, Any] | None:
    entries = past_answers.get("entries", [])
    for entry in reversed(entries):
        if entry.get("intent") != intent.value:
            continue
        if not entry.get("answer"):
            continue
        if not entry.get("submitted_at"):
            continue
        if entry.get("pending"):
            continue
        return entry
    return None


def _normalize_personal_facts(items: list[dict[str, Any]]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for item in items:
        intent = item.get("intent")
        value = str(item.get("value", "")).strip()
        if not intent or not value:
            continue
        normalized[str(intent)] = {
            "value": value,
            "confirmed_at": item.get("confirmed_at") or "",
            "source": item.get("source") or "unknown",
            "country": item.get("country"),
        }
    return normalized or dict(_DEFAULT_FACTS)


def _normalize_answer_history(items: list[dict[str, Any]]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for entry in items:
        if entry.get("pending"):
            continue
        if not entry.get("answer"):
            continue
        if not entry.get("submitted_at"):
            continue
        normalized.append(entry)
    return {"entries": normalized or list(_DEFAULT_HISTORY["entries"])}
