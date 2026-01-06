from enum import Enum


class DecisionAction(Enum):
    WAIT_FOR_HUMAN = "WAIT_FOR_HUMAN"
    AUTO_ANSWER = "AUTO_ANSWER"


CONFIDENCE_THRESHOLD = 0.7


def decide_action(is_critical: bool, confidence_score: float) -> DecisionAction:
    if is_critical:
        return DecisionAction.WAIT_FOR_HUMAN
    if confidence_score < CONFIDENCE_THRESHOLD:
        return DecisionAction.WAIT_FOR_HUMAN
    return DecisionAction.AUTO_ANSWER
