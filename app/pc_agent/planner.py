from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.pc_agent.safety import RiskLevel, is_high_risk


@dataclass
class PlannedAction:
    name: str
    description: str
    risk: RiskLevel
    reversible: bool
    payload: dict[str, Any]


def build_action_plan(task: str, context: dict[str, Any] | None = None) -> list[PlannedAction]:
    context = context or {}
    steps = context.get("steps", [])
    actions: list[PlannedAction] = []

    for step in steps:
        name = step.get("name")
        if not name:
            continue
        risk = step.get("risk", RiskLevel.MEDIUM)
        if isinstance(risk, str):
            risk = RiskLevel[risk.upper()]
        actions.append(
            PlannedAction(
                name=name,
                description=step.get("description", name),
                risk=risk,
                reversible=bool(step.get("reversible", True)),
                payload=step.get("payload", {}),
            )
        )

    if actions:
        return actions

    default_steps = [
        PlannedAction(
            name="open_browser",
            description="Open Chrome",
            risk=RiskLevel.LOW,
            reversible=True,
            payload={},
        )
    ]
    if context.get("url"):
        default_steps.append(
            PlannedAction(
                name="navigate",
                description=f"Go to {context.get('url')}",
                risk=RiskLevel.MEDIUM,
                reversible=True,
                payload={"url": context.get("url")},
            )
        )
    return default_steps


def requires_extra_confirmation(actions: list[PlannedAction]) -> bool:
    return any(is_high_risk(action.name) for action in actions)
