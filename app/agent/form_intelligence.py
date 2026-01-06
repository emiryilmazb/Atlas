from __future__ import annotations

from dataclasses import dataclass, field
import logging

from app.agent.answer_generator import generate_answer, is_needs_human_response
from app.agent.confidence import RiskLevel, assess_risk, score_confidence
from app.agent.llm_client import LLMClient
from app.agent.normalization import normalize_question
from app.agent.safety import is_critical_question
from app.agent.types import Intent, NormalizedQuestion
from app.storage.memory import MemoryContext, get_history_answer, get_personal_fact

logger = logging.getLogger(__name__)


@dataclass
class DecisionTrace:
    entries: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.entries.append(message)

    def log(self) -> None:
        for entry in self.entries:
            logger.info("Trace: %s", entry)


@dataclass
class DecisionOutcome:
    normalized: NormalizedQuestion
    answer: str | None
    needs_human: bool
    confidence_score: float
    risk_level: RiskLevel
    reason: str
    trace: DecisionTrace


def process_question(
    question_text: str,
    llm_client: LLMClient | None,
    memory: MemoryContext,
    company_name: str,
    role: str,
) -> DecisionOutcome:
    trace = DecisionTrace()
    trace.add(f"Raw question received: {question_text}")

    is_critical = is_critical_question(question_text)
    trace.add(f"Safety gate: {'FAILED' if is_critical else 'PASSED'}")

    normalized = normalize_question(question_text, llm_client)
    trace.add(f"Normalized intent: {normalized.normalized_intent.value}")

    personal_fact = get_personal_fact(memory.personal_facts, normalized.normalized_intent)
    if personal_fact:
        risk_level = assess_risk(question_text, normalized.normalized_intent)
        decision_reason = "memory_has_answer"
        logger.info(
            "Decision: confidence=%.2f risk=%s reason=%s",
            1.0,
            risk_level.value,
            decision_reason,
        )
        trace.add(f"Memory read: personal fact hit for {normalized.normalized_intent.value}")
        trace.add("Confidence score: 1.00")
        trace.add(f"Risk level: {risk_level.value}")
        trace.add(f"Decision reason: {decision_reason}")
        logger.info("Memory read: personal fact used for intent=%s", normalized.normalized_intent.value)
        return DecisionOutcome(
            normalized=normalized,
            answer=str(personal_fact.get("value", "")).strip(),
            needs_human=False,
            confidence_score=1.0,
            risk_level=risk_level,
            reason=decision_reason,
            trace=trace,
        )

    reusable_intent = normalized.normalized_intent not in {
        Intent.MOTIVATION_COMPANY,
        Intent.MOTIVATION_ROLE,
        Intent.OPEN_TEXT,
    }
    if reusable_intent:
        history_answer = get_history_answer(memory.past_answers, normalized.normalized_intent)
        if history_answer:
            risk_level = assess_risk(question_text, normalized.normalized_intent)
            decision_reason = "memory_has_answer"
            logger.info(
                "Decision: confidence=%.2f risk=%s reason=%s",
                1.0,
                risk_level.value,
                decision_reason,
            )
            trace.add(f"Memory read: history hit for {normalized.normalized_intent.value}")
            trace.add("Confidence score: 1.00")
            trace.add(f"Risk level: {risk_level.value}")
            trace.add(f"Decision reason: {decision_reason}")
            logger.info("Memory read: answer history used for intent=%s", normalized.normalized_intent.value)
            return DecisionOutcome(
                normalized=normalized,
                answer=str(history_answer.get("answer", "")).strip(),
                needs_human=False,
                confidence_score=1.0,
                risk_level=risk_level,
                reason=decision_reason,
                trace=trace,
            )

    risk_level = assess_risk(question_text, normalized.normalized_intent)
    if is_critical:
        risk_level = RiskLevel.HIGH
    confidence_score = score_confidence(normalized, llm_client, memory)

    if risk_level == RiskLevel.LOW and confidence_score >= 0.75:
        decision_reason = "auto_answer_low_risk"
        should_answer = True
    elif risk_level == RiskLevel.MEDIUM and confidence_score >= 0.85:
        decision_reason = "auto_answer_medium_risk"
        should_answer = True
    else:
        decision_reason = "ask_human"
        should_answer = False

    logger.info(
        "Decision: confidence=%.2f risk=%s reason=%s",
        confidence_score,
        risk_level.value,
        decision_reason,
    )
    trace.add(f"Confidence score: {confidence_score:.2f}")
    trace.add(f"Risk level: {risk_level.value}")
    trace.add(f"Decision reason: {decision_reason}")

    answer = None
    needs_human = True
    if should_answer:
        logger.info("Memory read: cv profile used for LLM context")
        answer = generate_answer(normalized, llm_client, memory, company_name, role)
        needs_human = is_critical or is_needs_human_response(answer)
        trace.add(f"LLM response: {'NEEDS_HUMAN' if needs_human else 'ANSWERED'}")
        if needs_human:
            decision_reason = "llm_requested_human"
            trace.add(f"Decision reason: {decision_reason}")
            logger.info(
                "Decision: confidence=%.2f risk=%s reason=%s",
                confidence_score,
                risk_level.value,
                decision_reason,
            )
            answer = None

    trace.log()
    return DecisionOutcome(
        normalized=normalized,
        answer=answer,
        needs_human=needs_human,
        confidence_score=confidence_score,
        risk_level=risk_level,
        reason=decision_reason,
        trace=trace,
    )
