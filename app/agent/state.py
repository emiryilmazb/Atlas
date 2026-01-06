from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class PendingHumanRequest:
    question_id: str
    intent: str
    asked_at: datetime
    source: str
    question: str
    category: str


@dataclass(frozen=True)
class HumanReply:
    question_id: str
    text: str
    timestamp: str
    source: str = "telegram"


@dataclass(frozen=True)
class AnswerRegistryEntry:
    intent: str
    value: str
    source: str
    confirmed_at: str


@dataclass
class AgentContext:
    current_state: AgentState = AgentState.IDLE
    pause_reason: str | None = None
    last_human_input: str | None = None
    transition_log: list[str] = field(default_factory=list)
    pending_request: PendingHumanRequest | None = None
    human_reply_queue: asyncio.Queue[HumanReply] = field(default_factory=asyncio.Queue)
    answer_registry: dict[str, AnswerRegistryEntry] = field(default_factory=dict)
    stop_requested: bool = False
    stop_reason: str | None = None
    pc_state: dict = field(default_factory=dict)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)

    def start(self) -> None:
        self._transition(AgentState.RUNNING)

    def pause_for_human(self, reason: str) -> None:
        self.pause_reason = reason
        self._transition(AgentState.WAITING_FOR_HUMAN)
        logger.info("WAITING_FOR_HUMAN")

    def resume_with_input(self, user_input: str) -> None:
        self.last_human_input = user_input
        if self.current_state == AgentState.RUNNING:
            return
        self._transition(AgentState.RUNNING)

    def complete(self) -> None:
        self._transition(AgentState.COMPLETED)

    def create_pending_request(
        self,
        question_id: str,
        intent: str,
        question: str,
        category: str,
        source: str = "telegram",
    ) -> PendingHumanRequest:
        pending = PendingHumanRequest(
            question_id=question_id,
            intent=intent,
            asked_at=datetime.now(timezone.utc),
            source=source,
            question=question,
            category=category,
        )
        self.pending_request = pending
        return pending

    def resolve_pending_request(self, reply: HumanReply) -> None:
        self.pending_request = None
        self.resume_with_input(reply.text)

    async def wait_for_human_reply(self) -> HumanReply:
        return await self.human_reply_queue.get()

    async def wait_for_human_reply_or_stop(self) -> HumanReply | None:
        reply_task = asyncio.create_task(self.human_reply_queue.get())
        stop_task = asyncio.create_task(self.stop_event.wait())
        done, pending = await asyncio.wait(
            {reply_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if stop_task in done:
            return None
        return reply_task.result()

    def request_stop(self, reason: str = "human_stop") -> None:
        self.stop_requested = True
        self.stop_reason = reason
        self.stop_event.set()
        logger.warning("STOP requested: %s", reason)

    def clear_stop(self) -> None:
        self.stop_requested = False
        self.stop_reason = None
        self.stop_event.clear()

    def register_answer(
        self,
        intent: str,
        value: str,
        source: str,
        confirmed_at: str,
    ) -> None:
        self.answer_registry[intent] = AnswerRegistryEntry(
            intent=intent,
            value=value,
            source=source,
            confirmed_at=confirmed_at,
        )

    def get_registered_answer(self, intent: str) -> AnswerRegistryEntry | None:
        return self.answer_registry.get(intent)

    def _transition(self, next_state: AgentState) -> None:
        if not self._is_allowed(self.current_state, next_state):
            message = f"Invalid transition: {self.current_state.value} -> {next_state.value}"
            raise ValueError(message)
        self.transition_log.append(f"{self.current_state.value} -> {next_state.value}")
        logger.info("Transition: %s -> %s", self.current_state.value, next_state.value)
        self.current_state = next_state

    @staticmethod
    def _is_allowed(current: AgentState, next_state: AgentState) -> bool:
        allowed = {
            AgentState.IDLE: {AgentState.RUNNING},
            AgentState.RUNNING: {AgentState.WAITING_FOR_HUMAN, AgentState.COMPLETED},
            AgentState.WAITING_FOR_HUMAN: {AgentState.RUNNING},
            AgentState.COMPLETED: set(),
        }
        return next_state in allowed.get(current, set())
