from __future__ import annotations

import asyncio
import logging

from telegram import Bot

from app.config import get_settings

logger = logging.getLogger(__name__)


async def _send(bot_token: str, chat_id: str, text: str) -> None:
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text=text)


def send_message(text: str) -> bool:
    settings = get_settings()
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.warning("Telegram settings are missing; cannot send message.")
        return False
    try:
        asyncio.run(_send(settings.telegram_bot_token, settings.telegram_chat_id, text))
        return True
    except RuntimeError:
        logger.error("Telegram send failed due to event loop state.")
        return False
    except Exception as exc:  # pragma: no cover - best effort notifications
        logger.error("Telegram send failed: %s", exc)
        return False
