from __future__ import annotations

from enum import Enum


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


HIGH_RISK_ACTIONS = {
    "run_executable",
    "install_software",
    "delete_file",
}


def is_high_risk(action_name: str) -> bool:
    return action_name in HIGH_RISK_ACTIONS
