from __future__ import annotations

import logging
from uuid import uuid4

from app.application.session import ApplicationStep, FailureReason, JobApplicationSession, StepType
from app.sites.base import StepExecutionResult

logger = logging.getLogger(__name__)


class ExternalAdapter:
    site_name = "external"

    async def get_application_steps(
        self,
        session: JobApplicationSession,
        page,
        agent_context=None,
        llm_client=None,
        telegram_app=None,
        chat_id: str | None = None,
    ) -> list[ApplicationStep]:
        fields = session.job_metadata.extra.get("fields", [])
        steps: list[ApplicationStep] = []
        for field in fields:
            question = str(field.get("label", "")).strip()
            if not question:
                continue
            steps.append(
                ApplicationStep(
                    step_id=str(uuid4()),
                    step_type=StepType.QUESTION,
                    payload={
                        "question": question,
                        "field_type": field.get("type", "text"),
                        "submit_answer": True,
                    },
                )
            )
        if not steps:
            steps.append(
                ApplicationStep(
                    step_id=str(uuid4()),
                    step_type=StepType.FORM_FILL,
                    payload={"section": "generic_form"},
                )
            )
        return steps

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
        logger.info("External ATS step executed: %s", step.step_type.value)
        if step.step_type == StepType.QUESTION and not answer:
            return StepExecutionResult(
                success=False,
                failure_reason=FailureReason.UNEXPECTED_STRUCTURE,
                details="Missing answer",
            )
        return StepExecutionResult(success=True, submitted=step.payload.get("submit_answer", False))
