from app.agent.types import AnswerFormat, AnswerLength, AnswerStyle, AnswerTone, Intent


def select_answer_style(intent: Intent) -> AnswerStyle:
    if intent in {Intent.MOTIVATION_COMPANY, Intent.MOTIVATION_ROLE}:
        return AnswerStyle(
            length=AnswerLength.MEDIUM,
            format=AnswerFormat.PARAGRAPH,
            tone=AnswerTone.PROFESSIONAL_FIRST_PERSON,
        )
    if intent == Intent.CULTURAL_FIT:
        return AnswerStyle(
            length=AnswerLength.MEDIUM,
            format=AnswerFormat.PARAGRAPH,
            tone=AnswerTone.PROFESSIONAL_FIRST_PERSON,
        )
    if intent == Intent.EXPERIENCE_TECH:
        return AnswerStyle(
            length=AnswerLength.MEDIUM,
            format=AnswerFormat.BULLETS,
            tone=AnswerTone.PROFESSIONAL_FIRST_PERSON,
        )
    if intent == Intent.EXPERIENCE_GENERAL:
        return AnswerStyle(
            length=AnswerLength.SHORT,
            format=AnswerFormat.BULLETS,
            tone=AnswerTone.PROFESSIONAL_FIRST_PERSON,
        )
    return AnswerStyle(
        length=AnswerLength.MEDIUM,
        format=AnswerFormat.PARAGRAPH,
        tone=AnswerTone.PROFESSIONAL_FIRST_PERSON,
    )
