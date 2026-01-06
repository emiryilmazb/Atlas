from __future__ import annotations

from app.agent.llm_client import LLMClient
from app.agent.types import Intent, NormalizedQuestion
from app.storage.memory import MemoryContext

ANSWER_PROMPT = (
    "Answer this job application question professionally using the context below.\n"
    "If the question is legal or requires factual confirmation, do NOT answer and say NEEDS_HUMAN."
)


def get_answer_prompt() -> str:
    return ANSWER_PROMPT


def generate_answer(
    normalized: NormalizedQuestion,
    llm_client: LLMClient | None,
    memory: MemoryContext,
    company_name: str,
    role: str,
) -> str:
    if llm_client is None:
        return "NEEDS_HUMAN"
    prompt = build_answer_prompt(normalized, memory, company_name, role)
    return llm_client.generate_text(prompt).strip()


def build_answer_prompt(
    normalized: NormalizedQuestion,
    memory: MemoryContext,
    company_name: str,
    role: str,
) -> str:
    summary = str(memory.cv_profile.get("summary", "")).strip()
    experience_items = memory.cv_profile.get("experience", [])
    experience_text = _format_list(experience_items)
    skills_items = memory.cv_profile.get("skills", [])
    skills_text = _format_list(skills_items)
    education_items = memory.cv_profile.get("education", [])
    education_text = _format_list(education_items)

    facts_text = _format_facts(memory.personal_facts)

    past_answers = _filter_past_answers(memory.past_answers, normalized.normalized_intent)
    past_answers_text = _format_past_answers(past_answers)

    return "\n\n".join(
        [
            f"Company: {company_name}",
            f"Role: {role}",
            "CV Summary:",
            summary if summary else "Not provided.",
            "Relevant Experience:",
            experience_text,
            "Skills:",
            skills_text,
            "Education:",
            education_text,
            "Personal Facts:",
            facts_text,
            f"Past Answers (intent: {normalized.normalized_intent.value}):",
            past_answers_text,
            "Question:",
            normalized.raw_text,
            "Instructions:",
            ANSWER_PROMPT,
        ]
    )


def is_needs_human_response(response: str) -> bool:
    stripped = response.strip()
    return not stripped or stripped.upper().startswith("NEEDS_HUMAN")


def _filter_past_answers(past_answers: dict, intent: Intent) -> list[dict]:
    entries = past_answers.get("entries", [])
    return [
        entry
        for entry in entries
        if entry.get("intent") == intent.value
        and entry.get("answer")
        and entry.get("submitted_at")
    ]


def _format_list(items: list) -> str:
    if not items:
        return "- None provided."
    return "\n".join(f"- {item}" for item in items)


def _format_facts(items: dict) -> str:
    if not items:
        return "- None provided."
    formatted = []
    for intent, payload in items.items():
        if not isinstance(payload, dict):
            continue
        value = str(payload.get("value", "")).strip()
        if value:
            formatted.append(f"- {intent}: {value}")
    return "\n".join(formatted) if formatted else "- None provided."


def _format_past_answers(items: list[dict]) -> str:
    if not items:
        return "- None provided."
    return "\n".join(
        f"- Q: {item.get('question', '')} A: {item.get('answer', '')}" for item in items
    )
