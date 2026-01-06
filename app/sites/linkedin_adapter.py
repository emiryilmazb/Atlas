from __future__ import annotations

import logging
from uuid import uuid4

from app.application.session import ApplicationStep, FailureReason, JobApplicationSession, StepType
from app.sites.base import StepExecutionResult

logger = logging.getLogger(__name__)


class LinkedInAdapter:
    site_name = "linkedin"

    async def get_application_steps(
        self,
        session: JobApplicationSession,
        page,
        agent_context=None,
        llm_client=None,
        telegram_app=None,
        chat_id: str | None = None,
    ) -> list[ApplicationStep]:
        return [
            ApplicationStep(step_id=str(uuid4()), step_type=StepType.FORM_FILL, payload={"section": "profile"}),
            ApplicationStep(step_id=str(uuid4()), step_type=StepType.DOCUMENT_UPLOAD, payload={"document": "resume"}),
            ApplicationStep(
                step_id=str(uuid4()),
                step_type=StepType.QUESTION,
                payload={
                    "question": "Why do you want to work with us?",
                    "field_type": "text",
                    "submit_answer": True,
                },
            ),
        ]

    async def execute_step(
        self,
        session: JobApplicationSession,
        step: ApplicationStep,
        answer: str | None = None,
        page=None,
        agent_context=None,
        llm_client=None,
        telegram_app=None,
        chat_id: str | None = None,
    ) -> StepExecutionResult:
        logger.info("LinkedIn step executed: %s", step.step_type.value)
        if step.step_type == StepType.QUESTION and not answer:
            return StepExecutionResult(
                success=False,
                failure_reason=FailureReason.UNEXPECTED_STRUCTURE,
                details="Missing answer",
            )
        return StepExecutionResult(success=True, submitted=step.payload.get("submit_answer", False))
