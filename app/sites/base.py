from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.application.session import ApplicationStep, FailureReason, JobApplicationSession


@dataclass
class StepExecutionResult:
    success: bool
    submitted: bool = False
    failure_reason: FailureReason | None = None
    details: str | None = None


class SiteAdapter(Protocol):
    site_name: str

    async def get_application_steps(
        self,
        session: JobApplicationSession,
        page,
        agent_context=None,
        llm_client=None,
        telegram_app=None,
        chat_id: str | None = None,
    ) -> list[ApplicationStep]:
        ...

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
        ...
