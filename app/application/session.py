from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SiteType(Enum):
    LINKEDIN = "linkedin"
    KARIYER = "kariyer"
    EXTERNAL = "external"


class StepType(Enum):
    FORM_FILL = "form_fill"
    QUESTION = "question"
    DOCUMENT_UPLOAD = "document_upload"


class StepStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class SessionStatus(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"


class FailureReason(Enum):
    TIMEOUT = "timeout"
    UNEXPECTED_STRUCTURE = "unexpected_structure"
    MISSING_DOCUMENT = "missing_document"
    UNKNOWN = "unknown"


@dataclass
class JobMetadata:
    company: str
    role: str
    location: str
    job_description: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApplicationStep:
    step_id: str
    step_type: StepType
    payload: dict[str, Any]
    status: StepStatus = StepStatus.PENDING


@dataclass
class JobApplicationSession:
    job_metadata: JobMetadata
    site_type: SiteType
    steps: list[ApplicationStep] = field(default_factory=list)
    status: SessionStatus = SessionStatus.RUNNING
    step_index: int = 0
    human_interactions_count: int = 0
    logs: list[str] = field(default_factory=list)

    def current_step(self) -> ApplicationStep | None:
        if 0 <= self.step_index < len(self.steps):
            return self.steps[self.step_index]
        return None

    def mark_step_completed(self) -> None:
        step = self.current_step()
        if step is not None:
            step.status = StepStatus.COMPLETED
            self.step_index += 1

    def pause(self) -> None:
        self.status = SessionStatus.PAUSED

    def resume(self) -> None:
        self.status = SessionStatus.RUNNING

    def fail(self, reason: FailureReason) -> None:
        self.status = SessionStatus.FAILED
        self.logs.append(f"Failure: {reason.value}")

    def complete(self) -> None:
        self.status = SessionStatus.COMPLETED

    def log_event(self, message: str) -> None:
        self.logs.append(message)
