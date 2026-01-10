from __future__ import annotations

import asyncio
import logging
import re

from app.config import get_settings
from app.services.gmail_service import GMAIL_LABELS, build_gmail_service
from app.storage.database import get_database

logger = logging.getLogger(__name__)

_PRIORITY_LABELS = ("Critical", "Action Required")


def _parse_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _extract_sender_email(sender: str | None) -> str:
    if not sender:
        return ""
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", sender)
    return match.group(0).lower() if match else ""


def schedule_gmail_polling(application) -> None:
    settings = application.bot_data.get("settings") or get_settings()
    if not getattr(settings, "gmail_poll_enabled", False):
        return
    if not getattr(settings, "gmail_credentials_path", ""):
        logger.warning("Gmail polling skipped: GMAIL_OAUTH_CREDENTIALS_PATH not set.")
        return
    interval = max(30, int(getattr(settings, "gmail_poll_interval_seconds", 120)))
    if not getattr(settings, "telegram_chat_id", ""):
        logger.warning("Gmail polling skipped: TELEGRAM_CHAT_ID not set.")
        return
    application.job_queue.run_repeating(
        _poll_gmail_job,
        interval=interval,
        first=10,
        data={"chat_id": settings.telegram_chat_id},
    )
    logger.info("Gmail polling scheduled every %s seconds.", interval)


async def _poll_gmail_job(context) -> None:
    application = context.application
    settings = application.bot_data.get("settings") or get_settings()
    chat_id = context.job.data.get("chat_id") if context.job and context.job.data else None
    if not chat_id:
        return
    llm_client = application.bot_data.get("llm_client")
    try:
        gmail_service = application.bot_data.get("gmail_service")
        if gmail_service is None:
            gmail_service = build_gmail_service(settings)
            application.bot_data["gmail_service"] = gmail_service
    except Exception as exc:
        logger.warning("Gmail polling unavailable: %s", exc)
        return
    db = application.bot_data.get("db") or get_database()
    max_results = int(getattr(settings, "gmail_poll_max_results", 10))
    vip_senders = _parse_csv(getattr(settings, "gmail_vip_senders", ""))
    focus_mode = bool(getattr(settings, "gmail_focus_mode", False))
    focus_labels = _parse_csv(getattr(settings, "gmail_focus_labels", "Critical,Action Required"))
    vip_only = bool(getattr(settings, "gmail_vip_only", False))
    later_label = getattr(settings, "gmail_later_label", "Later") or "Later"
    try:
        messages = await asyncio.to_thread(gmail_service.list_unread, max_results)
    except Exception as exc:
        logger.warning("Gmail polling fetch failed: %s", exc)
        return
    if not messages:
        return
    for message in messages:
        message_id = message.message_id
        if db.has_gmail_notification(str(chat_id), message_id):
            continue
        sender_email = _extract_sender_email(message.sender)
        label = "Action Required"
        if llm_client is not None:
            try:
                label = await asyncio.to_thread(
                    gmail_service.categorize_message, llm_client, message
                )
            except Exception as exc:
                logger.warning("Gmail polling categorization failed: %s", exc)
                label = "Action Required"
        is_vip = sender_email in vip_senders if sender_email else False
        should_notify = label in _PRIORITY_LABELS
        if focus_mode and not (is_vip or label in focus_labels):
            should_notify = False
        if vip_only and not is_vip:
            should_notify = False
            try:
                await asyncio.to_thread(gmail_service.label_message, message_id, later_label)
            except Exception as exc:
                logger.warning("Gmail label Later failed: %s", exc)
        if should_notify:
            await _notify_priority_email(application, chat_id, message, label)
            db.mark_gmail_notified(str(chat_id), message_id, label=label)
        if label in GMAIL_LABELS:
            try:
                await asyncio.to_thread(
                    gmail_service.label_message, message_id, label
                )
            except Exception as exc:
                logger.warning("Gmail label apply failed: %s", exc)


async def _notify_priority_email(application, chat_id: str, message, label: str) -> None:
    subject = message.subject or "(no subject)"
    sender = message.sender or "Unknown sender"
    snippet = message.snippet or ""
    text = (
        f"High priority email ({label}):\n"
        f"From: {sender}\n"
        f"Subject: {subject}\n"
        f"Snippet: {snippet}"
    )
    try:
        await application.bot.send_message(chat_id=chat_id, text=text)
    except Exception as exc:
        logger.warning("Gmail notification failed: %s", exc)
