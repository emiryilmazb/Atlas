import asyncio
import inspect
import io
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import mimetypes
from pathlib import Path
import random
import re

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from PIL import Image, ImageDraw, ImageFont

from app.agent.intent_router import RouterDecision, RouterIntent, route_intent
from app.agent.llm_client import (
    GEMINI_FLASH_MAX_DOCUMENT_BYTES,
    GEMINI_FLASH_MAX_INLINE_IMAGE_BYTES,
    GEMINI_IMAGE_MIME_TYPES,
)
from app.config import get_settings
from app.session_manager import (
    clear_session,
    get_session_context,
    record_image_result,
    record_user_message,
    should_reset_context,
)
from app.storage.database import get_database
from app.telegram.streaming import AsyncStreamHandler, MessageManager

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
IMAGE_OUTPUT_DIR = ROOT_DIR / "artifacts" / "images"
TEMP_IMAGE_DIR = ROOT_DIR / "artifacts" / "tmp_images"
_IMAGE_EDIT_TOKENS = (
    "edit",
    "duzenle",
    "degistir",
    "modify",
    "change",
    "remove",
    "add",
    "replace",
    "sil",
    "ekle",
    "kaldir",
    "renk",
    "color",
    "background",
    "arka plan",
)
_EXPLICIT_IMAGE_TOKENS = (
    "resim",
    "gorsel",
    "image",
    "photo",
    "picture",
    "foto",
    "fotograf",
)

_ANALYSIS_TOKENS = (
    "analyze",
    "analyse",
    "describe",
    "identify",
    "explain",
    "suitable",
    "appropriate",
    "good for",
    "is this good",
    "is this suitable",
    "is it good",
    "is it ok",
    "is it okay",
    "is it professional",
    "how is it",
    "how does it look",
    "does it look",
    "what is",
    "whats",
    "nedir",
    "tanimla",
    "acikla",
    "incele",
    "ne goruyorsun",
    "uygun mu",
    "sence",
    "degerlendir",
    "degerlendirir misin",
    "yorumla",
    "fikir ver",
    "nasil duruyor",
    "nasil gorunuyor",
    "bu hali nasil",
    "bu halin nasil",
    "simdi nasil",
)
_CHAT_CONTEXT_LIMIT = 20
_SUGGESTION_PREFIX = "SUGG:"
_TELEGRAM_TEXT_CHUNK = 3500
_TELEGRAM_CAPTION_LIMIT = 1024
_STREAM_RETRY_ATTEMPTS = 3
_STREAM_RETRY_BASE_DELAY = 0.6
_STREAM_RETRY_MAX_DELAY = 3.0
_SUGGESTION_RETRY_ATTEMPTS = 3
_SUGGESTION_RETRY_BASE_DELAY = 0.4
_SUGGESTION_RETRY_MAX_DELAY = 2.0
_TELEGRAM_REPLY_RETRY_ATTEMPTS = 2
_TELEGRAM_REPLY_RETRY_BASE_DELAY = 0.4
_TELEGRAM_REPLY_RETRY_MAX_DELAY = 2.0
_MAX_SOURCE_COUNT = 5
_CHART_TOKENS = (
    "chart",
    "grafik",
    "plot",
    "bar",
    "bar chart",
    "candlestick",
    "kandil",
    "ohlc",
    "line",
    "cizgi",
    "moving average",
    "hareketli ortalama",
    "sutun",
    "column",
)
_CANDLE_TOKENS = (
    "candlestick",
    "kandil",
    "ohlc",
)
_LINE_TOKENS = (
    "line",
    "cizgi",
    "moving average",
    "hareketli ortalama",
)
_MA_TOKENS = (
    "moving average",
    "hareketli ortalama",
    "ortalama",
    "ma",
)
_CHART_MAX_ITEMS = 8
_CHART_WIDTH = 900
_CHART_HEIGHT = 600
_BYTES_PER_MB = 1024 * 1024
_DEFAULT_AUDIO_PROMPT = "Transcribe the audio and provide a concise summary."
_DEFAULT_VIDEO_PROMPT = "Summarize the video and transcribe any spoken audio."
_CLEAR_HISTORY_COMMANDS = {"clear_history", "clear history", "/clear_history"}
_CLEAR_HISTORY_CONFIRM_TOKENS = {"yes", "y", "evet", "onay", "onayla"}
_CLEAR_HISTORY_CANCEL_TOKENS = {"no", "n", "hayir", "iptal", "vazgec"}
_CLEAR_HISTORY_FLAG = "awaiting_clear_history_confirmation"
_THINKING_ON_COMMAND = "thinking_on"
_THINKING_OFF_COMMAND = "thinking_off"
_SCREENSHOT_ON_COMMAND = "screenshot on"
_SCREENSHOT_OFF_COMMAND = "screenshot off"
_COMMAND_LIST_COMMANDS = {"komutlar", "/komutlar",
                          "commands", "/commands", "/help", "help"}
_CLEAR_HISTORY_CONFIRM_DATA = "clear_history_yes"
_CLEAR_HISTORY_CANCEL_DATA = "clear_history_no"
_APPROVAL_ID_PATTERN = re.compile(
    r"approval_id=([A-Za-z0-9-]+)", re.IGNORECASE)

_ANALYSIS_CLASSIFIER_PROMPT = """
You classify the user's request about an image.
Return ONLY JSON with this shape:
{{"intent": "ANALYZE|EDIT|UNKNOWN"}}

ANALYZE = user asks for evaluation, description, suitability, or feedback.
EDIT = user asks to change, modify, add/remove elements, or transform the image.
UNKNOWN = unclear.

User message: {message}
""".strip()

_BUTTONS = [
    ("Job Search", "job_search"),
]


def _resolve_user_id(update: Update) -> str | None:
    if update.effective_user and update.effective_user.id:
        return str(update.effective_user.id)
    if update.effective_chat and update.effective_chat.id:
        return str(update.effective_chat.id)
    return None


def _memory_text(message_text: str) -> str:
    cleaned = message_text.strip()
    lowered = cleaned.lower()
    if lowered.startswith("pc:"):
        return cleaned.split(":", 1)[1].strip()
    if lowered.startswith("/pc"):
        parts = cleaned.split(maxsplit=1)
        if len(parts) > 1:
            return parts[1].strip()
    return cleaned


def _extract_approval_id(text: str | None) -> str | None:
    if not text:
        return None
    match = _APPROVAL_ID_PATTERN.search(text)
    if not match:
        return None
    return match.group(1)


def _extract_approval_id_from_reply(message) -> str | None:
    reply = getattr(message, "reply_to_message", None)
    if reply is None:
        return None
    reply_text = getattr(reply, "text", None) or getattr(
        reply, "caption", None) or ""
    return _extract_approval_id(reply_text)


def _build_reply_context_note(message, user_id: str | None = None) -> str | None:
    reply = getattr(message, "reply_to_message", None)
    if reply is None:
        return None
    entry = _lookup_reply_entry(reply, user_id)
    if entry is not None:
        note = _format_reply_entry(entry)
        if note:
            return note
    content = _extract_reply_content(reply)
    if not content:
        return None
    cleaned = content.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace('"', "'")
    return f'Sistem Notu: "Kullanici su mesaja yanitliyor: {cleaned}"'


def _lookup_reply_entry(reply, user_id: str | None):
    if not user_id:
        return None
    source_message_id = getattr(reply, "message_id", None)
    source_chat_id = getattr(reply, "chat_id", None)
    try:
        db = get_database()
    except Exception as exc:
        logger.warning("Reply context lookup failed: %s", exc)
        return None
    if source_message_id is not None:
        for source in ("telegram_callback", "telegram"):
            entry = db.get_message_by_source_id(
                user_id,
                source,
                str(source_message_id),
                str(source_chat_id) if source_chat_id is not None else None,
            )
            if entry is not None:
                return entry
    content = (getattr(reply, "text", None) or getattr(
        reply, "caption", None) or "").strip()
    if not content:
        return None
    reply_from = getattr(reply, "from_user", None)
    role = None
    if reply_from is not None:
        role = "assistant" if getattr(reply_from, "is_bot", False) else "user"
    return db.find_message_by_content(user_id, content, role=role)


def _format_reply_entry(entry) -> str | None:
    line = _format_history_line(entry)
    cleaned = line.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace('"', "'")
    return f'Sistem Notu: "Kullanici su mesaja yanitliyor: {cleaned}"'


def _extract_reply_content(reply) -> str | None:
    text = (getattr(reply, "text", None) or getattr(
        reply, "caption", None) or "").strip()
    if text:
        return text
    return _format_reply_metadata(reply)


def _format_reply_metadata(reply) -> str | None:
    parts: list[str] = []
    message_id = getattr(reply, "message_id", None)
    if message_id:
        parts.append(f"message_id={message_id}")
    photo = getattr(reply, "photo", None)
    if photo:
        photo_item = photo[-1]
        photo_id = getattr(photo_item, "file_unique_id", None) or getattr(
            photo_item, "file_id", None)
        if photo_id:
            parts.append(f"photo_id={photo_id}")
    document = getattr(reply, "document", None)
    if document:
        doc_name = getattr(document, "file_name", None)
        doc_id = getattr(document, "file_unique_id", None) or getattr(
            document, "file_id", None)
        if doc_name:
            parts.append(f"document_name={doc_name}")
        if doc_id:
            parts.append(f"document_id={doc_id}")
    video = getattr(reply, "video", None)
    if video:
        video_id = getattr(video, "file_unique_id", None) or getattr(
            video, "file_id", None)
        if video_id:
            parts.append(f"video_id={video_id}")
    video_note = getattr(reply, "video_note", None)
    if video_note:
        video_note_id = getattr(video_note, "file_unique_id", None) or getattr(
            video_note, "file_id", None)
        if video_note_id:
            parts.append(f"video_note_id={video_note_id}")
    audio = getattr(reply, "audio", None)
    if audio:
        audio_id = getattr(audio, "file_unique_id", None) or getattr(
            audio, "file_id", None)
        if audio_id:
            parts.append(f"audio_id={audio_id}")
    voice = getattr(reply, "voice", None)
    if voice:
        voice_id = getattr(voice, "file_unique_id", None) or getattr(
            voice, "file_id", None)
        if voice_id:
            parts.append(f"voice_id={voice_id}")
    sticker = getattr(reply, "sticker", None)
    if sticker:
        sticker_id = getattr(sticker, "file_unique_id",
                             None) or getattr(sticker, "file_id", None)
        if sticker_id:
            parts.append(f"sticker_id={sticker_id}")
    animation = getattr(reply, "animation", None)
    if animation:
        animation_id = getattr(animation, "file_unique_id", None) or getattr(
            animation, "file_id", None)
        if animation_id:
            parts.append(f"animation_id={animation_id}")
    if not parts:
        return None
    return ", ".join(parts)


def _extract_reply_photo(message):
    reply = getattr(message, "reply_to_message", None)
    if reply is None:
        return None
    photo = getattr(reply, "photo", None)
    if photo:
        return photo[-1]
    document = getattr(reply, "document", None)
    if document and (document.mime_type or "").startswith("image/"):
        return document
    return None


def _build_callback_source_meta(query) -> dict[str, str | None] | None:
    if query is None:
        return None
    message = getattr(query, "message", None)
    if message is None:
        return None
    message_id = getattr(message, "message_id", None)
    if message_id is None:
        return None
    chat_id = getattr(message, "chat_id", None)
    return {
        "source": "telegram_callback",
        "source_message_id": str(message_id),
        "source_chat_id": str(chat_id) if chat_id is not None else None,
    }


def _user_source_kwargs(message, source_meta: dict[str, str | None] | None) -> dict[str, str | None]:
    if source_meta:
        return source_meta
    return {"message": message}


def _inject_reply_context(prompt_text: str, reply_note: str | None) -> str:
    if not reply_note:
        return prompt_text
    if not prompt_text:
        return reply_note
    return f"{reply_note}\n\n{prompt_text}"


def _resolve_settings(context: ContextTypes.DEFAULT_TYPE):
    settings = context.application.bot_data.get("settings")
    return settings or get_settings()


def build_main_keyboard() -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton(
        text=label, callback_data=value)] for label, value in _BUTTONS]
    return InlineKeyboardMarkup(keyboard)


def _normalize_command_text(value: str) -> str:
    normalized = _normalize_text(value or "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _is_clear_history_command(normalized: str) -> bool:
    return normalized in _CLEAR_HISTORY_COMMANDS


def _build_clear_history_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(
                text="YES", callback_data=_CLEAR_HISTORY_CONFIRM_DATA),
            InlineKeyboardButton(
                text="NO", callback_data=_CLEAR_HISTORY_CANCEL_DATA),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def _is_thinking_command(normalized: str) -> bool:
    return normalized in {_THINKING_ON_COMMAND, _THINKING_OFF_COMMAND}


def _set_show_thoughts(context: ContextTypes.DEFAULT_TYPE, enabled: bool) -> bool:
    settings = _resolve_settings(context)
    if bool(getattr(settings, "show_thoughts", True)) == enabled:
        return False
    context.application.bot_data["settings"] = replace(
        settings, show_thoughts=enabled)
    return True


def _is_screenshot_command(normalized: str) -> bool:
    return normalized in {_SCREENSHOT_ON_COMMAND, _SCREENSHOT_OFF_COMMAND}


def _set_screenshot_enabled(context: ContextTypes.DEFAULT_TYPE, enabled: bool) -> bool:
    settings = _resolve_settings(context)
    if bool(getattr(settings, "screenshot_enabled", True)) == enabled:
        return False
    context.application.bot_data["settings"] = replace(
        settings, screenshot_enabled=enabled)
    return True


def _is_command_list_command(normalized: str) -> bool:
    return normalized in _COMMAND_LIST_COMMANDS


def _build_command_list_text() -> str:
    return "\n".join(
        (
            "Kullanilabilir komutlar:",
            "- clear_history veya /clear_history: sohbet gecmisini temizler.",
            "- thinking_on / thinking_off: dusunce akisini acar/kapatir.",
            "- screenshot on / screenshot off: adimlardan sonra ekran goruntusu gonderimini acar/kapatir.",
            "- job_search: is aramayi baslatir (buton).",
            "- job_search stop: is aramasini durdurur.",
            "- stop: tum aksiyonlari durdurur.",
            "- /pc <gorev> veya pc: <gorev>: PC kontrol gorevi baslatir.",
        )
    )


def _resolve_clear_history_action(value: str) -> str | None:
    if value == _CLEAR_HISTORY_CONFIRM_DATA:
        return "confirm"
    if value == _CLEAR_HISTORY_CANCEL_DATA:
        return "cancel"
    return None


def _is_job_search_stop_command(normalized: str) -> bool:
    return normalized in {"job_search stop", "job search stop"}


def _has_clear_history_pending(context: ContextTypes.DEFAULT_TYPE) -> bool:
    return bool(context.user_data.get(_CLEAR_HISTORY_FLAG))


def _set_clear_history_pending(context: ContextTypes.DEFAULT_TYPE, pending: bool) -> None:
    if pending:
        context.user_data[_CLEAR_HISTORY_FLAG] = True
        return
    context.user_data.pop(_CLEAR_HISTORY_FLAG, None)


def _confirmation_token(normalized: str) -> str:
    return normalized.split()[0] if normalized else ""


async def _request_clear_history_confirmation(message, context: ContextTypes.DEFAULT_TYPE) -> None:
    _set_clear_history_pending(context, True)
    await message.reply_text(
        "Clear chat history? Tap YES to confirm or NO to cancel.",
        reply_markup=_build_clear_history_keyboard(),
    )


async def _handle_clear_history_confirmation(
    message,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: str | None,
    normalized: str,
) -> bool:
    if not _has_clear_history_pending(context):
        return False
    token = _confirmation_token(normalized)
    if token in _CLEAR_HISTORY_CONFIRM_TOKENS:
        _set_clear_history_pending(context, False)
        await _clear_user_history(message, context, user_id)
        return True
    if token in _CLEAR_HISTORY_CANCEL_TOKENS:
        _set_clear_history_pending(context, False)
        await message.reply_text("Clear history canceled.")
        return True
    await message.reply_text("Please reply YES or NO to confirm clear history.")
    return True


async def _clear_user_history(
    message,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: str | None,
) -> None:
    if not user_id:
        await message.reply_text("Unable to clear history (missing user id).")
        return
    db = get_database()
    session_context = get_session_context(user_id)
    extra_paths: list[str] = []
    if session_context:
        last_image_path = session_context.get("last_image_path")
        if last_image_path:
            extra_paths.append(str(last_image_path))
    file_paths = _safe_delete_messages(db, user_id)
    if extra_paths:
        file_paths = list({*file_paths, *extra_paths})
    deleted_files = _delete_local_files(file_paths)
    clear_session(user_id)
    _clear_legacy_session_file(user_id)
    if deleted_files:
        await message.reply_text("History cleared. Stored messages and local files were removed.")
        return
    await message.reply_text("History cleared. Stored messages were removed.")


def _safe_delete_messages(db, user_id: str) -> list[str]:
    try:
        return db.delete_messages(user_id)
    except Exception as exc:
        logger.warning("Message history delete failed: %s", exc)
        return []


def _delete_local_files(file_paths: list[str]) -> int:
    if not file_paths:
        return 0
    deleted = 0
    root = ROOT_DIR.resolve()
    for value in file_paths:
        if not value:
            continue
        try:
            path = Path(value).expanduser().resolve()
        except Exception as exc:
            logger.warning("Failed to resolve path for deletion: %s", exc)
            continue
        if root not in path.parents and path != root:
            logger.warning("Skipping delete for non-project path: %s", path)
            continue
        if not path.exists() or not path.is_file():
            continue
        try:
            path.unlink()
            deleted += 1
        except Exception as exc:
            logger.warning("File delete failed: %s", exc)
    return deleted


def _clear_legacy_session_file(user_id: str | None) -> bool:
    if not user_id:
        return False
    path = ROOT_DIR / "sessions.json"
    if not path.exists():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read sessions.json: %s", exc)
        return False
    if not isinstance(raw, dict):
        logger.warning("sessions.json is not a dict; skipping cleanup.")
        return False
    if str(user_id) not in raw:
        return False
    raw.pop(str(user_id), None)
    try:
        path.write_text(json.dumps(raw, ensure_ascii=True,
                        indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to update sessions.json: %s", exc)
        return False
    return True


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "ApplyWise bot is online",
        reply_markup=build_main_keyboard(),
    )


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    try:
        await query.answer()
    except Exception as exc:
        logger.warning("Callback query answer failed: %s", exc)
    logger.info("Telegram button selected: %s", query.data)

    clear_action = _resolve_clear_history_action(str(query.data or ""))
    if clear_action:
        message = query.message
        if message is None:
            return
        if not _has_clear_history_pending(context):
            await message.reply_text("Clear history request expired. Send clear_history again.")
            return
        _set_clear_history_pending(context, False)
        if clear_action == "confirm":
            user_id = _resolve_user_id(update)
            await _clear_user_history(message, context, user_id)
        else:
            await message.reply_text("Clear history canceled.")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception as exc:
            logger.debug("Failed to clear history buttons: %s", exc)
        return

    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        return

    pending = agent_context.pending_request
    callback_text = str(query.data or "")
    suggestion_text = _strip_suggestion_prefix(callback_text)
    is_suggestion = suggestion_text is not None

    # Check if we should treat this as a pending reply
    # - If pending is NOT startup: always reply to it
    # - If pending IS startup: only reply if button is "job_search"
    should_reply_to_pending = False
    if pending is not None and not is_suggestion:
        if pending.category != "startup":
            should_reply_to_pending = True
        elif str(query.data) in {"job_search", "job search"}:
            should_reply_to_pending = True

    if should_reply_to_pending:
        # If there's a pending request, treat the button click as a reply to it
        from app.agent.state import HumanReply

        reply = HumanReply(
            question_id=pending.question_id,
            text=str(query.data),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="telegram",
        )
        await agent_context.human_reply_queue.put(reply)
    else:
        # If no pending request (or ignoring startup pending), treat it as a new message/command
        message_text = suggestion_text if is_suggestion else callback_text
        if not message_text:
            return
        source_meta = _build_callback_source_meta(query)
        user_id = _resolve_user_id(update)
        memory_text = _memory_text(message_text)
        if user_id:
            if should_reset_context(user_id, memory_text):
                clear_session(user_id)
            record_user_message(user_id, memory_text)

        # We need a message object to reply to. The query.message is the message with the buttons.
        # We can simulate a new message handling or call route_and_handle directly.
        # Since we want to show it as if the user typed it, let's just use the existing message context
        # but with the new text.

        await _route_and_handle_message(
            update=update,
            context=context,
            agent_context=agent_context,
            user_id=user_id,
            message_text=message_text,
            has_image=False,
            photo=None,
            source_meta=source_meta,
            # Force ignore pending (e.g. startup) for dynamic buttons
            pending=None,
        )


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    user_id = _resolve_user_id(update)
    normalized = _normalize_command_text(message.text)
    approval_id = _extract_approval_id_from_reply(message)
    reply_photo = _extract_reply_photo(message)

    if not approval_id:
        if _is_clear_history_command(normalized):
            await _request_clear_history_confirmation(message, context)
            return
        if _has_clear_history_pending(context):
            handled = await _handle_clear_history_confirmation(message, context, user_id, normalized)
            if handled:
                return
        if _is_thinking_command(normalized):
            enabled = normalized == _THINKING_ON_COMMAND
            changed = _set_show_thoughts(context, enabled)
            status = "enabled" if enabled else "disabled"
            suffix = " now" if changed else " already"
            await message.reply_text(f"Thinking is{suffix} {status}.")
            return
        if _is_screenshot_command(normalized):
            enabled = normalized == _SCREENSHOT_ON_COMMAND
            changed = _set_screenshot_enabled(context, enabled)
            status = "enabled" if enabled else "disabled"
            suffix = " now" if changed else " already"
            await message.reply_text(f"Screenshots are{suffix} {status}.")
            return
        if _is_command_list_command(normalized):
            await message.reply_text(_build_command_list_text())
            return

    memory_text = _memory_text(message.text)
    if user_id:
        if should_reset_context(user_id, memory_text):
            clear_session(user_id)
        record_user_message(user_id, memory_text)

    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        await message.reply_text("Agent is not ready. Please try again later.")
        return

    if approval_id:
        handled = await _handle_approval_reply(message, agent_context, approval_id)
        if handled:
            return

    if _is_job_search_stop_command(normalized):
        agent_context.request_stop("telegram_job_search_stop")
        await message.reply_text("Job search stopped.")
        return
    if normalized == "stop":
        agent_context.request_stop("telegram_stop")
        await message.reply_text("Stopping all actions. You can send RESUME to continue later.")
        return

    pending = agent_context.pending_request
    if pending is not None and pending.category != "startup":
        handled = await _handle_pending_reply(message, agent_context, message.text)
        if handled:
            return

    await _route_and_handle_message(
        update=update,
        context=context,
        agent_context=agent_context,
        user_id=user_id,
        message_text=message.text,
        has_image=bool(reply_photo),
        photo=reply_photo,
        pending=pending,
    )


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not (message.photo or message.document):
        return
    media = None
    if message.photo:
        media = message.photo[-1]
    elif message.document and (message.document.mime_type or "").startswith("image/"):
        media = message.document
    if media is None:
        return
    user_id = _resolve_user_id(update)
    caption = message.caption or ""
    if caption:
        memory_text = _memory_text(caption)
        if user_id:
            if should_reset_context(user_id, memory_text):
                clear_session(user_id)
            record_user_message(user_id, memory_text)

    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        await message.reply_text("Agent is not ready. Please try again later.")
        return

    pending = agent_context.pending_request
    if pending is not None and pending.category != "startup":
        handled = await _handle_pending_reply(message, agent_context, caption)
        if handled:
            return

    await _route_and_handle_message(
        update=update,
        context=context,
        agent_context=agent_context,
        user_id=user_id,
        message_text=caption,
        has_image=True,
        photo=media,
        pending=pending,
    )


async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.document:
        return
    document = message.document
    if (document.mime_type or "").startswith("image/"):
        return
    user_id = _resolve_user_id(update)
    caption = message.caption or ""
    if caption:
        memory_text = _memory_text(caption)
        if user_id:
            if should_reset_context(user_id, memory_text):
                clear_session(user_id)
            record_user_message(user_id, memory_text)

    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        await message.reply_text("Agent is not ready. Please try again later.")
        return

    pending = agent_context.pending_request
    if pending is not None and pending.category != "startup":
        handled = await _handle_pending_reply(message, agent_context, caption)
        if handled:
            return

    if pending is not None and pending.category == "startup" and _is_startup_reply(caption):
        await _handle_pending_reply(message, agent_context, caption)
        return

    await _handle_document_request(
        message=message,
        context=context,
        caption=caption,
        document=document,
        user_id=user_id,
        session_context=get_session_context(user_id),
    )


async def handle_media_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    media, media_kind, default_prompt = _extract_media_payload(message)
    if media is None:
        return
    user_id = _resolve_user_id(update)
    caption = message.caption or ""
    if caption:
        memory_text = _memory_text(caption)
        if user_id:
            if should_reset_context(user_id, memory_text):
                clear_session(user_id)
            record_user_message(user_id, memory_text)

    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        await message.reply_text("Agent is not ready. Please try again later.")
        return

    pending = agent_context.pending_request
    if pending is not None and pending.category != "startup":
        handled = await _handle_pending_reply(message, agent_context, caption)
        if handled:
            return

    if pending is not None and pending.category == "startup" and _is_startup_reply(caption):
        await _handle_pending_reply(message, agent_context, caption)
        return

    await _handle_media_request(
        message=message,
        context=context,
        caption=caption,
        media=media,
        user_id=user_id,
        session_context=get_session_context(user_id),
        media_kind=media_kind,
        default_prompt=default_prompt,
    )


async def handle_pc_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    task = message.text.split(maxsplit=1)
    if len(task) < 2:
        await message.reply_text("Usage: /pc <task>. Example: /pc Steam uygulamasini ac")
        return
    user_id = _resolve_user_id(update)
    memory_text = _memory_text(message.text)
    if user_id:
        if should_reset_context(user_id, memory_text):
            clear_session(user_id)
        record_user_message(user_id, memory_text)
    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        await message.reply_text("Agent is not ready. Please try again later.")
        return
    pending = agent_context.pending_request
    await _route_and_handle_message(
        update=update,
        context=context,
        agent_context=agent_context,
        user_id=user_id,
        message_text=message.text,
        has_image=False,
        photo=None,
        pending=pending,
    )


async def _route_and_handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    agent_context,
    user_id: str | None,
    message_text: str,
    *,
    has_image: bool,
    photo,
    source_meta: dict[str, str | None] | None = None,
    pending,
) -> None:
    message = update.message
    if message is None and update.callback_query is not None:
        message = update.callback_query.message
    if message is None:
        return
    llm_client = context.application.bot_data.get("llm_client")
    image_client = context.application.bot_data.get("image_client")
    session_context = get_session_context(user_id)
    settings = _resolve_settings(context)

    decision = await asyncio.to_thread(
        route_intent,
        llm_client,
        message_text,
        has_image=has_image,
        session_context=session_context,
    )
    task_text, has_pc_prefix = _extract_pc_task(message_text)
    if has_pc_prefix:
        decision = RouterDecision(
            intent=RouterIntent.COMPUTER_USE, reason="forced:pc_prefix")
    if (
        not has_image
        and session_context
        and session_context.get("last_image_path")
        and await _is_analysis_request_with_llm(llm_client, message_text)
        and _mentions_explicit_image_reference(message_text)
    ):
        decision = RouterDecision(
            intent=RouterIntent.IMAGE_EDIT, reason="override:analysis_last_image")

    if pending is not None and pending.category == "startup" and _is_startup_reply(message_text):
        await _handle_pending_reply(message, agent_context, message_text)
        return

    if decision.intent == RouterIntent.COMPUTER_USE:
        await _start_pc_task(
            task_text,
            update,
            context,
            user_id=user_id,
            require_prefix=has_pc_prefix,
            silent_fail=False,
        )
        return
    if decision.intent == RouterIntent.IMAGE_GEN:
        await _handle_image_generation(
            message,
            image_client,
            llm_client,
            message_text,
            user_id=user_id,
            session_context=session_context,
            source_meta=source_meta,
            settings=settings,
            telegram_app=context.application,
        )
        return
    if decision.intent == RouterIntent.IMAGE_EDIT:
        await _handle_image_edit(
            message,
            image_client,
            llm_client,
            message_text,
            user_id=user_id,
            photo=photo,
            session_context=session_context,
            source_meta=source_meta,
            settings=settings,
            telegram_app=context.application,
        )
        return

    await _handle_chat(
        message,
        llm_client,
        message_text,
        user_id,
        session_context,
        source_meta=source_meta,
        settings=settings,
        telegram_app=context.application,
    )


async def _reply_with_suggestions(
    message,
    text: str,
    llm_client,
    history_text: str | None = None,
    photo=None,
    caption: str | None = None,
) -> object | None:
    """Sends a message with dynamic suggestions generated by LLM."""
    reply_markup = await _build_suggestion_markup(llm_client, history_text)

    message_text = (text or "").strip()
    if photo:
        photo_caption = (caption or message_text).strip()
        formatted_caption = _format_telegram_html(
            photo_caption) if photo_caption else ""
        if formatted_caption and len(formatted_caption) > _TELEGRAM_CAPTION_LIMIT:
            photo_message = await message.reply_photo(photo=photo, caption="Result")
            if message_text:
                await _send_text_in_chunks(
                    message,
                    message_text,
                    chunk_size=_TELEGRAM_TEXT_CHUNK,
                    reply_markup=reply_markup,
                )
            return photo_message
        return await message.reply_photo(
            photo=photo,
            caption=formatted_caption or None,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
        )
    if message_text:
        if len(message_text) > _TELEGRAM_TEXT_CHUNK:
            return await _send_text_in_chunks(
                message,
                message_text,
                chunk_size=_TELEGRAM_TEXT_CHUNK,
                reply_markup=reply_markup,
            )
        return await message.reply_text(
            _format_telegram_html(message_text),
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
        )
    return None


async def _handle_approval_reply(message, agent_context, approval_id: str) -> bool:
    reply_text = (message.text or message.caption or "").strip()
    if not reply_text:
        await message.reply_text(
            "Please reply YES or NO to approve the plan, or send a change request.",
        )
        return True
    pending = agent_context.pending_request
    if pending is not None and pending.question_id == approval_id:
        return await _handle_pending_reply(message, agent_context, reply_text)

    from app.agent.state import HumanReply

    reply = HumanReply(
        question_id=approval_id,
        text=reply_text,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="telegram",
    )
    await agent_context.human_reply_queue.put(reply)
    logger.info("Approval reply received for approval_id=%s", approval_id)
    await message.reply_text(f"Received reply for approval_id={approval_id}. Thank you!")
    return True


async def _handle_pending_reply(message, agent_context, reply_text: str | None) -> bool:
    pending = agent_context.pending_request
    if pending is None:
        return False
    if not reply_text or not reply_text.strip():
        await message.reply_text(
            "There is a pending request already. Please reply to it first or send STOP.",
        )
        return True

    from app.agent.state import HumanReply

    message_time = message.date.astimezone(timezone.utc)
    if message_time < pending.asked_at:
        logger.info("Ignored Telegram message (sent before question was asked)")
        return True

    reply = HumanReply(
        question_id=pending.question_id,
        text=reply_text,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="telegram",
    )
    await agent_context.human_reply_queue.put(reply)
    logger.info("Human reply received for question_id=%s", pending.question_id)
    await message.reply_text(f"Received reply for question_id={pending.question_id}. Thank you!")
    return True


async def _handle_chart_request(
    message,
    llm_client,
    prompt_text: str,
    user_id: str | None,
    session_context: dict | None,
    *,
    source_meta: dict[str, str | None] | None = None,
    telegram_app,
) -> None:
    db = get_database() if user_id else None
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=prompt_text,
            timestamp=_timestamp_from_message(message),
            **_user_source_kwargs(message, source_meta),
        )
    reply_note = _build_reply_context_note(message, user_id)
    llm_prompt_text = _inject_reply_context(prompt_text, reply_note)
    chart_type = _resolve_chart_type(prompt_text)
    show_ma = _should_show_moving_average(prompt_text)
    payload = None
    if chart_type in {"line", "candlestick"}:
        payload = _generate_time_series_payload(
            prompt_text,
            chart_type=chart_type,
            show_ma=show_ma,
        )
    else:
        try:
            chart_prompt = _build_chart_data_prompt(llm_prompt_text)
            payload_text = await asyncio.to_thread(llm_client.generate_text, chart_prompt)
            payload = _parse_chart_payload(payload_text)
        except Exception as exc:  # pragma: no cover - network/proxy errors
            logger.warning("Chart data generation failed: %s", exc)
        if payload is None:
            payload = _fallback_chart_payload(prompt_text)
    if payload is None:
        await message.reply_text("Could not build chart data. Please try again.")
        return

    image_bytes = None
    if payload.get("type") == "candlestick":
        image_bytes = _render_candlestick_chart(payload)
    elif payload.get("type") == "line":
        image_bytes = _render_line_chart(payload)
    else:
        image_bytes = _render_bar_chart(payload)
    if not image_bytes:
        await message.reply_text("Chart rendering failed. Please try again.")
        return

    output_path = _save_image_bytes(image_bytes, "chart")
    record_image_result(user_id, output_path, prompt_text, source="chart")
    assistant_row_id = None
    if db and user_id:
        assistant_row_id = _safe_add_message(
            db,
            user_id=user_id,
            role="assistant",
            content="Chart generated.",
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_path=output_path,
            file_type="image/png",
        )

    lines = []
    title = payload.get("title")
    if title:
        lines.append(title)
    lines.append("Data:")
    if payload.get("type") == "candlestick":
        for label, point in zip(payload.get("labels", []), payload.get("points", [])):
            close_price = point.get("close")
            lines.append(f"- {label}: {close_price}")
    else:
        for item in payload.get("items", payload.get("points", [])):
            label = item.get("label")
            value = item.get("value")
            if label is None or value is None:
                continue
            lines.append(f"- {label}: {value}")
    summary_text = "\n".join(lines).strip()

    suggestion_context = _build_suggestion_context(
        user_id=user_id,
        db=db,
        session_context=session_context,
        user_message=prompt_text,
        assistant_message=summary_text or "Chart generated.",
        task_type="chart",
        extra_context="Generated a bar chart from fictional data.",
    )
    with open(output_path, "rb") as handle:
        sent_message = await _reply_with_suggestions(
            message,
            summary_text or "Chart generated.",
            llm_client,
            history_text=suggestion_context,
            photo=handle,
            caption=summary_text or "Chart generated.",
        )
    _safe_update_message_source(db, assistant_row_id, sent_message)


async def _handle_chat(
    message,
    llm_client,
    prompt_text: str,
    user_id: str | None,
    session_context: dict | None,
    *,
    source_meta: dict[str, str | None] | None = None,
    settings,
    telegram_app,
) -> None:
    prompt_text = prompt_text.strip()
    if not prompt_text:
        await message.reply_text("Please send a message.")
        return
    if llm_client is None:
        await message.reply_text("Gemini is not configured. Please set GEMINI_API_KEY.")
        return
    reply_note = _build_reply_context_note(message, user_id)
    llm_prompt_text = _inject_reply_context(prompt_text, reply_note)
    if _is_chart_request(prompt_text):
        await _handle_chart_request(
            message,
            llm_client,
            prompt_text,
            user_id,
            session_context,
            source_meta=source_meta,
            telegram_app=telegram_app,
        )
        return
    db = get_database() if user_id else None
    history = _safe_get_history(
        db, user_id, _CHAT_CONTEXT_LIMIT) if db and user_id else []
    timestamp = _timestamp_from_message(message)
    prompt = _build_chat_prompt(llm_prompt_text, history)
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=prompt_text,
            timestamp=timestamp,
            **_user_source_kwargs(message, source_meta),
        )
    streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
    show_thoughts = bool(getattr(settings, "show_thoughts", True))
    if streaming_enabled and hasattr(llm_client, "stream_text"):
        async def _stream_runner() -> None:
            try:
                live_message = await message.reply_text("⏳ Thinking...", parse_mode=ParseMode.HTML)
                manager = MessageManager(
                    bot=telegram_app.bot,
                    chat_id=str(message.chat_id),
                    message_id=live_message.message_id,
                )
                handler = None
                result = None
                sources = []
                for attempt in range(1, _STREAM_RETRY_ATTEMPTS + 1):
                    handler = AsyncStreamHandler(
                        manager,
                        show_thoughts=show_thoughts,
                        response_label="",
                    )
                    if hasattr(llm_client, "stream_text_with_sources"):
                        stream_state = await llm_client.stream_text_with_sources(
                            prompt,
                            include_thoughts=show_thoughts,
                        )
                        result = await handler.stream_chunks(
                            stream_state.chunks,
                            recovery=lambda: handler._attempt_recovery(
                                llm_client, prompt),
                        )
                        sources = stream_state.get_sources()
                    else:
                        result = await handler.stream(
                            llm_client,
                            prompt,
                            include_thoughts=show_thoughts,
                        )
                    if result.error is None:
                        break
                    if attempt < _STREAM_RETRY_ATTEMPTS:
                        await manager.update("⏳ Retrying...", force=True)
                        await asyncio.sleep(_retry_delay(attempt, _STREAM_RETRY_BASE_DELAY, _STREAM_RETRY_MAX_DELAY))
                if handler is None or result is None:
                    raise RuntimeError("Streaming did not produce a result.")
                response_text = result.response_text.strip()
                sources_text = _format_sources_text(sources)
                display_text = response_text
                if sources_text:
                    display_text = f"{response_text}\n\n{sources_text}"
                assistant_row_id = None
                if response_text and result.completed and db and user_id:
                    assistant_row_id = _safe_add_message(
                        db,
                        user_id=user_id,
                        role="assistant",
                        content=response_text,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                if response_text and result.completed:
                    suggestion_context = _build_suggestion_context(
                        user_id=user_id,
                        db=db,
                        session_context=session_context,
                        user_message=prompt_text,
                        assistant_message=response_text,
                        task_type="chat",
                    )
                    reply_markup = await _build_suggestion_markup(llm_client, suggestion_context)
                    response_limit = 3000 if not handler.thought_text else 2200
                    if len(display_text) > response_limit:
                        await manager.finalize("Response sent below.")
                        sent_message = await _send_text_in_chunks(
                            message,
                            display_text,
                            chunk_size=_TELEGRAM_TEXT_CHUNK,
                            reply_markup=reply_markup,
                        )
                        _safe_update_message_source(
                            db, assistant_row_id, sent_message)
                    else:
                        final_message = handler.format_message(
                            include_thoughts=False)
                        if sources_text:
                            final_message = f"{final_message}\n\n{_format_telegram_html(sources_text)}"
                        await manager.finalize(final_message, reply_markup=reply_markup)
                        _safe_update_message_source(
                            db, assistant_row_id, live_message)
                else:
                    await manager.finalize(handler.format_message(include_thoughts=False))
                if not response_text and not handler.thought_text:
                    await message.reply_text("No response generated.")
            except Exception as exc:  # pragma: no cover - network/telegram errors
                logger.error("Streaming chat failed: %s", exc)
                await message.reply_text("Chat response failed. Please try again.")

        telegram_app.create_task(_stream_runner())
        return

    sources = []
    try:
        if hasattr(llm_client, "generate_text_with_sources"):
            response_text, sources = await asyncio.to_thread(
                llm_client.generate_text_with_sources,
                prompt,
            )
        else:
            response_text = await asyncio.to_thread(llm_client.generate_text, prompt)
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.error("Chat response failed: %s", exc)
        await message.reply_text("Chat response failed. Please try again.")
        return
    if not response_text:
        await message.reply_text("No response generated.")
        return
    sources_text = _format_sources_text(sources)
    display_text = response_text
    if sources_text:
        display_text = f"{response_text}\n\n{sources_text}"
    assistant_row_id = None
    if db and user_id:
        assistant_row_id = _safe_add_message(
            db,
            user_id=user_id,
            role="assistant",
            content=response_text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    suggestion_context = _build_suggestion_context(
        user_id=user_id,
        db=db,
        session_context=session_context,
        user_message=prompt_text,
        assistant_message=response_text,
        task_type="chat",
    )
    sent_message = await _reply_with_suggestions(
        message,
        display_text,
        llm_client,
        history_text=suggestion_context,
    )
    _safe_update_message_source(db, assistant_row_id, sent_message)


async def _handle_image_generation(
    message,
    image_client,
    llm_client,
    prompt_text: str,
    user_id: str | None,
    session_context: dict | None,
    *,
    source_meta: dict[str, str | None] | None = None,
    settings,
    telegram_app,
) -> None:
    if image_client is None:
        await message.reply_text("Gemini image model is not configured. Please set GEMINI_API_KEY.")
        return
    prompt_text = prompt_text.strip()
    if not prompt_text:
        await message.reply_text("Please describe the image you want to generate.")
        return
    reply_note = _build_reply_context_note(message, user_id)
    llm_prompt_text = _inject_reply_context(prompt_text, reply_note)
    db = get_database() if user_id else None
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=prompt_text,
            timestamp=_timestamp_from_message(message),
            **_user_source_kwargs(message, source_meta),
        )
    streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
    show_thoughts = bool(getattr(settings, "show_thoughts", True))
    if streaming_enabled and show_thoughts and hasattr(image_client, "stream_generate_image"):
        async def _stream_runner() -> None:
            async def _finalize_generation(image_bytes: bytes) -> None:
                output_path = _save_image_bytes(image_bytes, "generated")
                record_image_result(user_id, output_path,
                                    prompt_text, source="generated")
                session_context = get_session_context(
                    user_id) or session_context
                assistant_row_id = None
                if db and user_id:
                    assistant_row_id = _safe_add_message(
                        db,
                        user_id=user_id,
                        role="assistant",
                        content="Image generated.",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        file_path=output_path,
                        file_type="image/png",
                    )

                suggestion_context = _build_suggestion_context(
                    user_id=user_id,
                    db=db,
                    session_context=session_context,
                    user_message=prompt_text,
                    assistant_message="Image generated.",
                    task_type="image_generation",
                    extra_context="User asked to generate an image.",
                )
                with open(output_path, "rb") as handle:
                    sent_message = await _reply_with_suggestions(
                        message,
                        "Image generated.",
                        llm_client,
                        history_text=suggestion_context,
                        photo=handle,
                        caption="Image generated.",
                    )
                _safe_update_message_source(db, assistant_row_id, sent_message)

            try:
                live_message = await message.reply_text(
                    "⏳ Generating image...",
                    parse_mode=ParseMode.HTML,
                )
                manager = MessageManager(
                    bot=telegram_app.bot,
                    chat_id=str(message.chat_id),
                    message_id=live_message.message_id,
                )
                handler = None
                image_bytes = None
                last_error: Exception | None = None
                for attempt in range(1, _STREAM_RETRY_ATTEMPTS + 1):
                    handler = AsyncStreamHandler(
                        manager,
                        show_thoughts=show_thoughts,
                        response_label="Decision",
                    )
                    stream_state = await image_client.stream_generate_image(
                        llm_prompt_text,
                        include_thoughts=show_thoughts,
                    )
                    result = await handler.stream_chunks(stream_state.chunks)
                    image_bytes = stream_state.get_image_bytes()
                    if result.error is None and image_bytes:
                        break
                    last_error = result.error or ValueError(
                        "No image bytes received from stream.")
                    if attempt < _STREAM_RETRY_ATTEMPTS:
                        await manager.update("⏳ Retrying...", force=True)
                        await asyncio.sleep(_retry_delay(attempt, _STREAM_RETRY_BASE_DELAY, _STREAM_RETRY_MAX_DELAY))
                if handler is None:
                    raise RuntimeError("Stream did not initialize.")
                if not image_bytes or (last_error is not None and image_bytes is None):
                    raise last_error or ValueError(
                        "No image bytes received from stream.")
                await _finalize_generation(image_bytes)
                await manager.finalize(handler.format_message(include_thoughts=False))
            except Exception as exc:  # pragma: no cover - network/telegram errors
                logger.error("Image generation stream failed: %s", exc)
                try:
                    image_bytes = await _retry_image_bytes(
                        lambda: asyncio.to_thread(
                            image_client.generate_image, llm_prompt_text),
                        label="Image generation",
                        message=message,
                        attempts=_STREAM_RETRY_ATTEMPTS,
                        base_delay=_STREAM_RETRY_BASE_DELAY,
                        max_delay=_STREAM_RETRY_MAX_DELAY,
                    )
                except Exception as fallback_exc:
                    logger.error("Image generation failed: %s", fallback_exc)
                    await _safe_reply_text(
                        message,
                        "Image generation failed. Please try again.",
                        label="Image generation failure reply",
                    )
                    return
                await _finalize_generation(image_bytes)

        telegram_app.create_task(_stream_runner())
        return
    try:
        image_bytes = await _retry_image_bytes(
            lambda: asyncio.to_thread(
                image_client.generate_image, llm_prompt_text),
            label="Image generation",
            message=message,
            attempts=_STREAM_RETRY_ATTEMPTS,
            base_delay=_STREAM_RETRY_BASE_DELAY,
            max_delay=_STREAM_RETRY_MAX_DELAY,
        )
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.error("Image generation failed: %s", exc)
        await _safe_reply_text(
            message,
            "Image generation failed. Please try again.",
            label="Image generation failure reply",
        )
        return
    output_path = _save_image_bytes(image_bytes, "generated")
    record_image_result(user_id, output_path, prompt_text, source="generated")
    session_context = get_session_context(user_id) or session_context
    assistant_row_id = None
    if db and user_id:
        assistant_row_id = _safe_add_message(
            db,
            user_id=user_id,
            role="assistant",
            content="Image generated.",
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_path=output_path,
            file_type="image/png",
        )

    suggestion_context = _build_suggestion_context(
        user_id=user_id,
        db=db,
        session_context=session_context,
        user_message=prompt_text,
        assistant_message="Image generated.",
        task_type="image_generation",
        extra_context="User asked to generate an image.",
    )
    with open(output_path, "rb") as handle:
        sent_message = await _reply_with_suggestions(
            message,
            "Image generated.",
            llm_client,
            history_text=suggestion_context,
            photo=handle,
            caption="Image generated."
        )
    _safe_update_message_source(db, assistant_row_id, sent_message)


async def _handle_image_edit(
    message,
    image_client,
    llm_client,
    prompt_text: str,
    *,
    user_id: str | None,
    photo,
    session_context: dict | None,
    source_meta: dict[str, str | None] | None = None,
    settings,
    telegram_app,
) -> None:
    if image_client is None:
        await message.reply_text("Gemini image model is not configured. Please set GEMINI_API_KEY.")
        return
    prompt_text = prompt_text.strip()
    if not prompt_text:
        await message.reply_text("Please describe the edit or analysis you want.")
        return
    reply_note = _build_reply_context_note(message, user_id)
    llm_prompt_text = _inject_reply_context(prompt_text, reply_note)

    reference_path = None
    if photo is not None:
        photo_size = getattr(photo, "file_size", None)
        if photo_size and photo_size > GEMINI_FLASH_MAX_INLINE_IMAGE_BYTES:
            await message.reply_text(
                _file_too_large_message(
                    "Image", GEMINI_FLASH_MAX_INLINE_IMAGE_BYTES)
            )
            return
        try:
            reference_path = await _download_photo_to_temp(photo, message)
        except Exception as exc:  # pragma: no cover - network/IO errors
            logger.error("Photo download failed: %s", exc)
            await message.reply_text("Photo download failed. Please try again.")
            return
        record_image_result(
            user_id,
            reference_path,
            prompt_text or None,
            source="uploaded",
        )
    else:
        reference_path = session_context.get(
            "last_image_path") if session_context else None
        if not reference_path or not Path(reference_path).exists():
            await message.reply_text("Please send the image you want to edit.")
            return

    validation_error = await _validate_image_for_gemini(
        llm_client,
        prompt_text,
        reference_path,
        getattr(photo, "mime_type", None) if photo is not None else None,
    )
    if validation_error:
        await message.reply_text(validation_error)
        return

    db = get_database() if user_id else None
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=prompt_text,
            timestamp=_timestamp_from_message(message),
            file_path=reference_path,
            file_type=mimetypes.guess_type(reference_path)[
                0] if reference_path else None,
            **_user_source_kwargs(message, source_meta),
        )

    if await _is_analysis_request_with_llm(llm_client, llm_prompt_text):
        streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
        show_thoughts = bool(getattr(settings, "show_thoughts", True))
        if streaming_enabled and hasattr(llm_client, "stream_with_image"):
            async def _stream_runner() -> None:
                try:
                    live_message = await message.reply_text(
                        "⏳ Analyzing image...",
                        parse_mode=ParseMode.HTML,
                    )
                    manager = MessageManager(
                        bot=telegram_app.bot,
                        chat_id=str(message.chat_id),
                        message_id=live_message.message_id,
                    )
                    handler = AsyncStreamHandler(
                        manager,
                        show_thoughts=show_thoughts,
                        response_label="Decision",
                    )

                    async def _recover() -> str:
                        return await asyncio.to_thread(
                            image_client.analyze_image,
                            llm_prompt_text,
                            reference_path,
                        )

                    result = await handler.stream_chunks(
                        llm_client.stream_with_image(
                            llm_prompt_text,
                            reference_path,
                            include_thoughts=show_thoughts,
                        ),
                        recovery=_recover,
                    )
                    response_text = result.response_text.strip()
                    if response_text and result.completed and db and user_id:
                        _safe_add_message(
                            db,
                            user_id=user_id,
                            role="assistant",
                            content=response_text,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            message=live_message,
                        )
                    if response_text and result.completed:
                        suggestion_context = _build_suggestion_context(
                            user_id=user_id,
                            db=db,
                            session_context=session_context,
                            user_message=prompt_text,
                            assistant_message=response_text,
                            task_type="image_analysis",
                            extra_context="User asked to analyze an image.",
                        )
                        reply_markup = await _build_suggestion_markup(llm_client, suggestion_context)
                        await manager.finalize(
                            handler.format_message(include_thoughts=False),
                            reply_markup=reply_markup,
                        )
                    else:
                        await manager.finalize(handler.format_message(include_thoughts=False))
                    if not response_text and not handler.thought_text:
                        await message.reply_text("No analysis result.")
                except Exception as exc:  # pragma: no cover - network/telegram errors
                    logger.error("Image analysis stream failed: %s", exc)
                    await message.reply_text("Image analysis failed. Please try again.")

            telegram_app.create_task(_stream_runner())
            return
        try:
            response_text = await asyncio.to_thread(
                image_client.analyze_image,
                llm_prompt_text,
                reference_path,
            )
        except Exception as exc:  # pragma: no cover - network/proxy errors
            logger.error("Image analysis failed: %s", exc)
            await message.reply_text("Image analysis failed. Please try again.")
            return
        if response_text:
            assistant_row_id = None
            if db and user_id:
                assistant_row_id = _safe_add_message(
                    db,
                    user_id=user_id,
                    role="assistant",
                    content=response_text,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

            suggestion_context = _build_suggestion_context(
                user_id=user_id,
                db=db,
                session_context=session_context,
                user_message=prompt_text,
                assistant_message=response_text,
                task_type="image_analysis",
                extra_context="User asked to analyze an image.",
            )
            sent_message = await _reply_with_suggestions(
                message,
                response_text,
                llm_client,
                history_text=suggestion_context,
            )
            _safe_update_message_source(db, assistant_row_id, sent_message)
        else:
            await message.reply_text("No analysis result.")
        return

    async def _finalize_edit(image_bytes: bytes) -> None:
        output_path = _save_image_bytes(image_bytes, "edited")
        record_image_result(user_id, output_path, prompt_text, source="edited")
        session_context = get_session_context(user_id) or session_context
        response_text = "Image updated."
        if llm_client is not None:
            summary_prompt = (
                "Write a short, friendly confirmation sentence in the user's language describing the image edit. "
                f"User request: {prompt_text}"
            )
            try:
                generated = await asyncio.to_thread(llm_client.generate_text, summary_prompt)
                if generated:
                    response_text = generated.strip()
            except Exception as exc:  # pragma: no cover - network/proxy errors
                logger.warning("Image edit summary failed: %s", exc)
        assistant_row_id = None
        if db and user_id:
            assistant_row_id = _safe_add_message(
                db,
                user_id=user_id,
                role="assistant",
                content=response_text,
                timestamp=datetime.now(timezone.utc).isoformat(),
                file_path=output_path,
                file_type="image/png",
            )

        suggestion_context = _build_suggestion_context(
            user_id=user_id,
            db=db,
            session_context=session_context,
            user_message=prompt_text,
            assistant_message=response_text,
            task_type="image_edit",
            extra_context="User asked to edit an image.",
        )
        with open(output_path, "rb") as handle:
            sent_message = await _reply_with_suggestions(
                message,
                response_text,
                llm_client,
                history_text=suggestion_context,
                photo=handle,
                caption=response_text,
            )
        _safe_update_message_source(db, assistant_row_id, sent_message)

    streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
    show_thoughts = bool(getattr(settings, "show_thoughts", True))
    if streaming_enabled and show_thoughts and hasattr(image_client, "stream_edit_image"):
        async def _stream_runner() -> None:
            try:
                live_message = await message.reply_text(
                    "⏳ Editing image...",
                    parse_mode=ParseMode.HTML,
                )
                manager = MessageManager(
                    bot=telegram_app.bot,
                    chat_id=str(message.chat_id),
                    message_id=live_message.message_id,
                )
                handler = None
                image_bytes = None
                last_error: Exception | None = None
                for attempt in range(1, _STREAM_RETRY_ATTEMPTS + 1):
                    handler = AsyncStreamHandler(
                        manager,
                        show_thoughts=show_thoughts,
                        response_label="Decision",
                    )
                    stream_state = await image_client.stream_edit_image(
                        llm_prompt_text,
                        reference_path,
                        include_thoughts=show_thoughts,
                    )
                    result = await handler.stream_chunks(stream_state.chunks)
                    image_bytes = stream_state.get_image_bytes()
                    if result.error is None and image_bytes:
                        break
                    last_error = result.error or ValueError(
                        "No image bytes received from stream.")
                    if attempt < _STREAM_RETRY_ATTEMPTS:
                        await manager.update("⏳ Retrying...", force=True)
                        await asyncio.sleep(_retry_delay(attempt, _STREAM_RETRY_BASE_DELAY, _STREAM_RETRY_MAX_DELAY))
                if handler is None:
                    raise RuntimeError("Stream did not initialize.")
                if not image_bytes or (last_error is not None and image_bytes is None):
                    raise last_error or ValueError(
                        "No image bytes received from stream.")
                await _finalize_edit(image_bytes)
                await manager.finalize(handler.format_message(include_thoughts=False))
            except Exception as exc:  # pragma: no cover - network/telegram errors
                logger.error("Image edit stream failed: %s", exc)
                try:
                    image_bytes = await _retry_image_bytes(
                        lambda: asyncio.to_thread(
                            image_client.edit_image,
                            llm_prompt_text,
                            reference_path,
                        ),
                        label="Image edit",
                        message=message,
                        attempts=_STREAM_RETRY_ATTEMPTS,
                        base_delay=_STREAM_RETRY_BASE_DELAY,
                        max_delay=_STREAM_RETRY_MAX_DELAY,
                    )
                except Exception as fallback_exc:
                    logger.error("Image edit failed: %s", fallback_exc)
                    await _safe_reply_text(
                        message,
                        "Image edit failed. Please try again.",
                        label="Image edit failure reply",
                    )
                    return
                await _finalize_edit(image_bytes)

        telegram_app.create_task(_stream_runner())
        return

    try:
        image_bytes = await _retry_image_bytes(
            lambda: asyncio.to_thread(
                image_client.edit_image,
                llm_prompt_text,
                reference_path,
            ),
            label="Image edit",
            message=message,
            attempts=_STREAM_RETRY_ATTEMPTS,
            base_delay=_STREAM_RETRY_BASE_DELAY,
            max_delay=_STREAM_RETRY_MAX_DELAY,
        )
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.error("Image edit failed: %s", exc)
        await _safe_reply_text(
            message,
            "Image edit failed. Please try again.",
            label="Image edit failure reply",
        )
        return
    await _finalize_edit(image_bytes)


async def _handle_document_request(
    message,
    context,
    caption: str,
    document,
    user_id: str | None,
    session_context: dict | None,
    source_meta: dict[str, str | None] | None = None,
) -> None:
    caption_text = caption.strip()
    if not caption_text:
        await message.reply_text("Please add a caption describing what to do with the file.")
        return
    llm_client = context.application.bot_data.get("llm_client")
    if llm_client is None:
        await message.reply_text("Gemini is not configured. Please set GEMINI_API_KEY.")
        return
    reply_note = _build_reply_context_note(message, user_id)
    llm_prompt_text = _inject_reply_context(caption_text, reply_note)
    document_size = getattr(document, "file_size", None)
    if document_size and document_size > GEMINI_FLASH_MAX_DOCUMENT_BYTES:
        await message.reply_text(
            _file_too_large_message(
                "Document", GEMINI_FLASH_MAX_DOCUMENT_BYTES)
        )
        return
    temp_path = None
    db = get_database() if user_id else None
    try:
        temp_path = await _download_document_to_temp(document, message)
    except Exception as exc:  # pragma: no cover - network/IO errors
        logger.error("Document download failed: %s", exc)
        await message.reply_text("Document download failed. Please try again.")
        return
    if document_size is None and temp_path:
        try:
            downloaded_size = Path(temp_path).stat().st_size
        except Exception as exc:
            logger.warning("Document size check failed: %s", exc)
            downloaded_size = None
        if downloaded_size and downloaded_size > GEMINI_FLASH_MAX_DOCUMENT_BYTES:
            await message.reply_text(
                _file_too_large_message(
                    "Document", GEMINI_FLASH_MAX_DOCUMENT_BYTES)
            )
            return

    mime_type = getattr(document, "mime_type", None)
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=caption_text,
            timestamp=_timestamp_from_message(message),
            file_path=temp_path,
            file_type=mime_type,
            **_user_source_kwargs(message, source_meta),
        )
    result = await asyncio.to_thread(
        llm_client.generate_with_file,
        llm_prompt_text,
        temp_path,
        mime_type,
    )
    if result.error:
        await message.reply_text(result.error)
        return
    if not result.text:
        await message.reply_text("No response generated for the document.")
        return
    assistant_row_id = None
    if db and user_id:
        assistant_row_id = _safe_add_message(
            db,
            user_id=user_id,
            role="assistant",
            content=result.text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    suggestion_context = _build_suggestion_context(
        user_id=user_id,
        db=db,
        session_context=session_context,
        user_message=caption_text,
        assistant_message=result.text,
        task_type="document",
        extra_context="User asked to process a document.",
    )
    reply_markup = await _build_suggestion_markup(llm_client, suggestion_context)
    sent_message = await _send_text_in_chunks(
        message,
        result.text,
        chunk_size=3500,
        reply_markup=reply_markup,
    )
    _safe_update_message_source(db, assistant_row_id, sent_message)


async def _handle_media_request(
    message,
    context,
    caption: str,
    media,
    user_id: str | None,
    session_context: dict | None,
    *,
    source_meta: dict[str, str | None] | None = None,
    media_kind: str,
    default_prompt: str,
) -> None:
    llm_client = context.application.bot_data.get("llm_client")
    if llm_client is None:
        await message.reply_text("Gemini is not configured. Please set GEMINI_API_KEY.")
        return
    caption_text = (caption or "").strip()
    prompt_text = caption_text or default_prompt
    if not prompt_text:
        await message.reply_text("Please add a caption describing what to do with the file.")
        return
    reply_note = _build_reply_context_note(message, user_id)
    llm_prompt_text = _inject_reply_context(prompt_text, reply_note)
    temp_path = None
    db = get_database() if user_id else None
    try:
        temp_path = await _download_media_to_temp(media, message, media_kind)
    except Exception as exc:  # pragma: no cover - network/IO errors
        logger.error("%s download failed: %s", media_kind.capitalize(), exc)
        await message.reply_text("Media download failed. Please try again.")
        return
    mime_type = getattr(media, "mime_type", None)
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=caption_text,
            timestamp=_timestamp_from_message(message),
            file_path=temp_path,
            file_type=mime_type,
            **_user_source_kwargs(message, source_meta),
        )
    result = await asyncio.to_thread(
        llm_client.generate_with_file,
        llm_prompt_text,
        temp_path,
        mime_type,
    )
    if result.error:
        await message.reply_text(result.error)
        return
    if not result.text:
        await message.reply_text("No response generated for the media.")
        return
    assistant_row_id = None
    if db and user_id:
        assistant_row_id = _safe_add_message(
            db,
            user_id=user_id,
            role="assistant",
            content=result.text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    suggestion_context = _build_suggestion_context(
        user_id=user_id,
        db=db,
        session_context=session_context,
        user_message=caption_text or None,
        assistant_message=result.text,
        task_type=media_kind,
        extra_context=f"User asked to process a {media_kind} file.",
    )
    reply_markup = await _build_suggestion_markup(llm_client, suggestion_context)
    sent_message = await _send_text_in_chunks(
        message,
        result.text,
        chunk_size=3500,
        reply_markup=reply_markup,
    )
    _safe_update_message_source(db, assistant_row_id, sent_message)


def _save_image_bytes(image_bytes: bytes, prefix: str) -> str:
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = IMAGE_OUTPUT_DIR / f"{prefix}_{timestamp}.png"
    path.write_bytes(image_bytes)
    return str(path)


async def _download_document_to_temp(document, message) -> str:
    TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    telegram_file = await document.get_file()
    suffix = ""
    if telegram_file.file_path:
        suffix = Path(telegram_file.file_path).suffix
    if not suffix and getattr(document, "file_name", None):
        suffix = Path(document.file_name).suffix
    if not suffix and getattr(document, "mime_type", None):
        suffix = mimetypes.guess_extension(document.mime_type) or ""
    if not suffix:
        suffix = ".bin"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = TEMP_IMAGE_DIR / \
        f"telegram_doc_{message.message_id}_{timestamp}{suffix}"
    await telegram_file.download_to_drive(custom_path=str(path))
    return str(path)


async def _download_photo_to_temp(photo, message) -> str:
    TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    telegram_file = await photo.get_file()
    suffix = ""
    if telegram_file.file_path:
        suffix = Path(telegram_file.file_path).suffix
    if not suffix and getattr(photo, "file_name", None):
        suffix = Path(photo.file_name).suffix
    if not suffix and getattr(photo, "mime_type", None):
        suffix = mimetypes.guess_extension(photo.mime_type) or ""
    if not suffix:
        suffix = ".jpg"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = TEMP_IMAGE_DIR / \
        f"telegram_{message.message_id}_{timestamp}{suffix}"
    await telegram_file.download_to_drive(custom_path=str(path))
    return str(path)


async def _download_media_to_temp(media, message, media_kind: str) -> str:
    TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    telegram_file = await media.get_file()
    suffix = ""
    if telegram_file.file_path:
        suffix = Path(telegram_file.file_path).suffix
    if not suffix and getattr(media, "file_name", None):
        suffix = Path(media.file_name).suffix
    if not suffix and getattr(media, "mime_type", None):
        suffix = mimetypes.guess_extension(media.mime_type) or ""
    if not suffix:
        suffix = ".bin"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = TEMP_IMAGE_DIR / \
        f"telegram_{media_kind}_{message.message_id}_{timestamp}{suffix}"
    await telegram_file.download_to_drive(custom_path=str(path))
    return str(path)


def _extract_media_payload(message) -> tuple[object | None, str, str]:
    if message.voice:
        return message.voice, "audio", _DEFAULT_AUDIO_PROMPT
    if message.audio:
        return message.audio, "audio", _DEFAULT_AUDIO_PROMPT
    if message.video_note:
        return message.video_note, "video", _DEFAULT_VIDEO_PROMPT
    if message.video:
        return message.video, "video", _DEFAULT_VIDEO_PROMPT
    return None, "", ""


def _extract_pc_task(message_text: str) -> tuple[str, bool]:
    cleaned = message_text.strip()
    lowered = cleaned.lower()
    if lowered.startswith("pc:"):
        return cleaned.split(":", 1)[1].strip(), True
    if lowered.startswith("/pc"):
        parts = cleaned.split(maxsplit=1)
        if len(parts) > 1:
            return parts[1].strip(), True
        return "", True
    return cleaned, False


def _is_startup_reply(message_text: str) -> bool:
    normalized = message_text.strip().lower()
    return normalized in {"job_search", "job search"}


async def _send_text_in_chunks(
    message,
    text: str,
    chunk_size: int = 3500,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> object | None:
    text = text.strip()
    if not text:
        return None
    last_message = None
    for start in range(0, len(text), chunk_size):
        chunk = text[start: start + chunk_size]
        is_last = start + chunk_size >= len(text)
        last_message = await message.reply_text(
            _format_telegram_html(chunk),
            reply_markup=reply_markup if is_last else None,
            parse_mode=ParseMode.HTML,
        )
    return last_message


def _format_telegram_html(text: str) -> str:
    if not text:
        return text
    parts = text.split("```")
    if len(parts) == 1:
        return _format_inline_html(text)
    formatted_parts: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            code = _escape_html(part.strip("\n"))
            if code:
                formatted_parts.append(f"<pre>{code}</pre>")
        else:
            formatted_parts.append(_format_inline_html(part))
    return "".join(formatted_parts)


def _format_inline_html(text: str) -> str:
    escaped = _escape_html(text)
    lines = escaped.splitlines(keepends=True)
    formatted: list[str] = []
    for line in lines:
        newline = "\n" if line.endswith("\n") else ""
        content = line[:-1] if newline else line
        heading = re.match(r"^(#{1,6})\s+(.*)$", content)
        if heading:
            heading_text = heading.group(2).strip()
            heading_text = _apply_inline_markdown(heading_text)
            content = f"<b>{heading_text}</b>" if heading_text else ""
        else:
            content = _apply_inline_markdown(content)
        formatted.append(content + newline)
    return "".join(formatted)


def _apply_inline_markdown(text: str) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    formatted_parts: list[str] = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) >= 2:
            inner = part[1:-1]
            formatted_parts.append(f"<code>{inner}</code>")
            continue
        styled = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", part)
        styled = re.sub(
            r"(?<!\*)\*([^\s][^*]*?[^\s])\*(?!\*)", r"<i>\1</i>", styled)
        formatted_parts.append(styled)
    return "".join(formatted_parts)


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _timestamp_from_message(message) -> str:
    message_time = getattr(message, "date", None)
    if message_time:
        return message_time.astimezone(timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _safe_get_history(db, user_id: str, limit: int):
    try:
        return db.get_recent_messages(user_id, limit)
    except Exception as exc:
        logger.warning("Message history load failed: %s", exc)
        return []


def _safe_add_message(db, **kwargs) -> int | None:
    message = kwargs.pop("message", None)
    if message is not None:
        role = kwargs.get("role")
        sender = getattr(message, "from_user", None)
        if role == "user" and sender is not None and getattr(sender, "is_bot", False):
            message = None
    if message is not None:
        kwargs.setdefault("source", "telegram")
        message_id = getattr(message, "message_id", None)
        if message_id is not None:
            kwargs.setdefault("source_message_id", str(message_id))
        chat_id = getattr(message, "chat_id", None)
        if chat_id is not None:
            kwargs.setdefault("source_chat_id", str(chat_id))
    try:
        return db.add_message(**kwargs)
    except Exception as exc:
        logger.warning("Message persistence failed: %s", exc)
        return None


def _safe_update_message_source(db, message_row_id: int | None, message) -> None:
    if not db or not message_row_id or message is None:
        return
    message_id = getattr(message, "message_id", None)
    if message_id is None:
        return
    chat_id = getattr(message, "chat_id", None)
    try:
        db.update_message_source(
            message_row_id,
            source="telegram",
            source_message_id=str(message_id),
            source_chat_id=str(chat_id) if chat_id is not None else None,
        )
    except Exception as exc:
        logger.warning("Message source update failed: %s", exc)


def _build_chat_prompt(prompt_text: str, history: list) -> str:
    if not history:
        return prompt_text
    lines = [_format_history_line(entry) for entry in reversed(history)]
    history_block = "\n".join(line for line in lines if line)
    if not history_block:
        return prompt_text
    return f"Conversation so far:\n{history_block}\n\nUser: {prompt_text}"


def _format_history_line(entry) -> str:
    content = (entry.content or "").strip()
    if not content and entry.file_path:
        content = "[attachment]"
    file_note = ""
    if entry.file_path:
        file_note = f" [file: {entry.file_path}"
        if entry.file_type:
            file_note += f", type: {entry.file_type}"
        file_note += "]"
    return f"{entry.role}: {content}{file_note}".strip()


def _build_suggestion_context(
    *,
    user_id: str | None,
    db,
    session_context: dict | None,
    user_message: str | None,
    assistant_message: str | None,
    task_type: str,
    extra_context: str | None = None,
) -> str:
    parts = [f"Task type: {task_type}"]
    session_lines = _format_session_context(session_context)
    if session_lines:
        parts.append("Session context:")
        parts.extend(session_lines)
    history_text = _format_recent_history(db, user_id, _CHAT_CONTEXT_LIMIT)
    if history_text:
        parts.append("Recent conversation history:")
        parts.append(history_text)
    if user_message:
        parts.append(f"User: {user_message}")
    if assistant_message:
        parts.append(f"Assistant: {assistant_message}")
    if extra_context:
        parts.append(f"Extra context: {extra_context}")
    return "\n".join(parts)


def _format_session_context(session_context: dict | None) -> list[str]:
    if not session_context:
        return []
    lines: list[str] = []
    for key in ("active_app", "last_action", "last_summary", "last_image_summary", "last_image_source"):
        value = session_context.get(key)
        if value:
            lines.append(f"- {key}: {value}")
    pending_details = session_context.get("pending_details")
    if isinstance(pending_details, dict) and pending_details:
        pending_items = ", ".join(
            f"{key}={value}" for key, value in pending_details.items() if value
        )
        if pending_items:
            lines.append(f"- pending_details: {pending_items}")
    short_term_memory = session_context.get("short_term_memory")
    if isinstance(short_term_memory, list) and short_term_memory:
        joined = " | ".join(str(item) for item in short_term_memory if item)
        if joined:
            lines.append(f"- session_history: {joined}")
    return lines


def _format_recent_history(db, user_id: str | None, limit: int) -> str:
    if not db or not user_id:
        return ""
    history = _safe_get_history(db, user_id, limit)
    if not history:
        return ""
    lines = [_format_history_line(entry) for entry in reversed(history)]
    history_block = "\n".join(line for line in lines if line)
    return history_block.strip()


def _format_sources_text(sources) -> str:
    if not sources:
        return ""
    cleaned: list[tuple[str, str]] = []
    seen = set()
    for source in sources:
        uri = str(getattr(source, "uri", "") or "").strip()
        if not uri or uri in seen:
            continue
        title = str(getattr(source, "title", "") or "").strip() or uri
        cleaned.append((title, uri))
        seen.add(uri)
        if len(cleaned) >= _MAX_SOURCE_COUNT:
            break
    if not cleaned:
        return ""
    lines = ["Sources:"]
    for title, uri in cleaned:
        if title == uri:
            lines.append(f"- {uri}")
        else:
            lines.append(f"- {title} ({uri})")
    return "\n".join(lines)


def _load_chart_font(size: int):
    candidates = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _resolve_chart_type(text: str) -> str:
    normalized = _normalize_text(text)
    if _contains_any(normalized, _CANDLE_TOKENS):
        return "candlestick"
    if _contains_any(normalized, _LINE_TOKENS):
        return "line"
    return "bar"


def _should_show_moving_average(text: str) -> bool:
    normalized = _normalize_text(text)
    return _contains_any(normalized, _MA_TOKENS)


def _is_chart_request(text: str) -> bool:
    normalized = _normalize_text(text)
    return _contains_any(normalized, _CHART_TOKENS)


def _build_chart_data_prompt(message: str) -> str:
    return (
        "You are generating data for a bar chart based on the user request.\n"
        "Return ONLY JSON with this shape:\n"
        "{\n"
        '  "title": "<short title>",\n'
        '  "x_label": "<x axis label>",\n'
        '  "y_label": "<y axis label>",\n'
        '  "items": [\n'
        '    {"label": "<short label>", "value": <integer>}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Use 3-8 items.\n"
        "- Values must be integers between 1 and 500.\n"
        "- Labels must be short (max 12 characters).\n"
        "- If the user asks for fictional data, make it up.\n"
        "- Use the user's language.\n"
        f"User request: {message}\n"
    )


def _parse_chart_payload(text: str) -> dict | None:
    if not text:
        return None
    cleaned = _extract_json_block(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Chart JSON parse failed: %s", cleaned[:200])
        return None
    items = data.get("items") or data.get("data")
    if not isinstance(items, list):
        return None
    normalized_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        value = item.get("value")
        try:
            value = int(float(value))
        except (TypeError, ValueError):
            continue
        if not label or value <= 0:
            continue
        normalized_items.append({"label": label[:12], "value": value})
        if len(normalized_items) >= _CHART_MAX_ITEMS:
            break
    if not normalized_items:
        return None
    return {
        "type": "bar",
        "title": str(data.get("title", "")).strip(),
        "x_label": str(data.get("x_label", "")).strip(),
        "y_label": str(data.get("y_label", "")).strip(),
        "items": normalized_items,
    }


def _extract_symbol(message: str) -> str:
    match = re.search(
        r"hisse\s*adi[:\s]*([A-Za-z0-9-]{2,})", message, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"\b([A-Z][A-Z0-9-]{2,})\b", message)
    if match:
        return match.group(1).strip()
    return "TITAN-AS"


def _build_month_labels(count: int = 12) -> list[str]:
    months = [
        "Ocak",
        "Subat",
        "Mart",
        "Nisan",
        "Mayis",
        "Haziran",
        "Temmuz",
        "Agustos",
        "Eylul",
        "Ekim",
        "Kasim",
        "Aralik",
    ]
    return months[:count]


def _generate_time_series_payload(
    message: str,
    *,
    chart_type: str,
    show_ma: bool,
    points: int = 12,
) -> dict:
    import random

    symbol = _extract_symbol(message)
    labels = _build_month_labels(points)
    title = f"{symbol} Hisse Performansi (1 Yillik)"
    x_label = "Ay"
    y_label = "Fiyat"

    if chart_type == "candlestick":
        series = []
        price = random.randint(80, 160)
        for _ in range(points):
            open_price = price
            change = random.randint(-25, 25)
            close_price = max(10, min(500, open_price + change))
            high = min(500, max(open_price, close_price) +
                       random.randint(5, 30))
            low = max(5, min(open_price, close_price) - random.randint(5, 25))
            series.append(
                {
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close_price,
                }
            )
            price = close_price
        spike_up = max(1, points // 3)
        spike_down = min(points - 2, spike_up + 3)
        series[spike_up]["close"] = min(
            500, series[spike_up]["open"] + random.randint(80, 140))
        series[spike_up]["high"] = min(
            500, series[spike_up]["close"] + random.randint(10, 30))
        series[spike_down]["close"] = max(
            10, series[spike_down]["open"] - random.randint(80, 140))
        series[spike_down]["low"] = max(
            5, series[spike_down]["close"] - random.randint(10, 25))
        return {
            "type": "candlestick",
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
            "labels": labels,
            "points": series,
            "show_ma": show_ma,
            "spikes": {"up": spike_up, "down": spike_down},
        }

    values: list[int] = []
    value = random.randint(80, 160)
    for _ in range(points):
        step = random.randint(-20, 20)
        value = max(10, min(500, value + step))
        values.append(value)
    spike_up = max(1, points // 3)
    spike_down = min(points - 2, spike_up + 4)
    values[spike_up] = min(500, values[spike_up - 1] + random.randint(80, 140))
    values[spike_down] = max(
        10, values[spike_down - 1] - random.randint(80, 140))
    points_payload = [{"label": label, "value": value}
                      for label, value in zip(labels, values)]
    return {
        "type": "line",
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "points": points_payload,
        "show_ma": show_ma,
        "spikes": {"up": spike_up, "down": spike_down},
    }


def _fallback_chart_payload(message: str) -> dict | None:
    match = re.search(r"\(([^)]+)\)", message or "")
    labels: list[str] = []
    if match:
        raw = match.group(1)
        labels = [part.strip() for part in raw.split(",") if part.strip()]
    labels = labels[:_CHART_MAX_ITEMS]
    if not labels:
        return None
    values = [50 + (len(labels) - idx) * 20 for idx in range(len(labels))]
    items = [{"label": label[:12], "value": value}
             for label, value in zip(labels, values)]
    return {
        "type": "bar",
        "title": "Bar chart",
        "x_label": "",
        "y_label": "",
        "items": items,
    }


def _render_bar_chart(payload: dict) -> bytes | None:
    items = payload.get("items") or []
    if not items:
        return None
    labels = [item["label"] for item in items]
    values = [item["value"] for item in items]
    if not labels or not values:
        return None

    width = _CHART_WIDTH
    height = _CHART_HEIGHT
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _load_chart_font(12)
    title_font = _load_chart_font(16)
    axis_font = _load_chart_font(12)

    left = 80
    right = width - 40
    top = 70
    bottom = height - 80
    chart_width = right - left
    chart_height = bottom - top

    title = payload.get("title") or ""
    if title:
        title_box = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_box[2] - title_box[0]
        draw.text(((width - title_w) / 2, 20), title,
                  fill="black", font=title_font)

    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)

    max_value = max(values) if max(values) > 0 else 1
    count = len(values)
    gap = max(8, int(chart_width * 0.04))
    bar_width = max(10, int((chart_width - gap * (count + 1)) / count))
    palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
               "#59a14f", "#edc948", "#b07aa1", "#ff9da7"]

    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = left + gap + idx * (bar_width + gap)
        x1 = x0 + bar_width
        bar_height = int((value / max_value) * chart_height)
        y1 = bottom
        y0 = bottom - bar_height
        color = palette[idx % len(palette)]
        draw.rectangle([x0, y0, x1, y1], fill=color, outline="black")

        value_text = str(value)
        value_box = draw.textbbox((0, 0), value_text, font=font)
        value_w = value_box[2] - value_box[0]
        draw.text((x0 + (bar_width - value_w) / 2, max(top, y0 - 15)),
                  value_text, fill="black", font=font)

        label_box = draw.textbbox((0, 0), label, font=font)
        label_w = label_box[2] - label_box[0]
        draw.text((x0 + (bar_width - label_w) / 2, bottom + 6),
                  label, fill="black", font=font)

    x_label = payload.get("x_label") or ""
    y_label = payload.get("y_label") or ""
    if x_label:
        label_box = draw.textbbox((0, 0), x_label, font=axis_font)
        label_w = label_box[2] - label_box[0]
        draw.text(((width - label_w) / 2, height - 30),
                  x_label, fill="black", font=axis_font)
    if y_label:
        label_box = draw.textbbox((0, 0), y_label, font=axis_font)
        label_h = label_box[3] - label_box[1]
        draw.text((10, top - label_h - 5), y_label,
                  fill="black", font=axis_font)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _render_line_chart(payload: dict) -> bytes | None:
    points = payload.get("points") or []
    if not points:
        return None
    labels = [point["label"] for point in points]
    values = [point["value"] for point in points]
    if not labels or not values:
        return None

    width = _CHART_WIDTH
    height = _CHART_HEIGHT
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _load_chart_font(12)
    title_font = _load_chart_font(16)
    axis_font = _load_chart_font(12)

    left = 80
    right = width - 40
    top = 70
    bottom = height - 80
    chart_width = right - left
    chart_height = bottom - top

    title = payload.get("title") or ""
    if title:
        title_box = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_box[2] - title_box[0]
        draw.text(((width - title_w) / 2, 20), title,
                  fill="black", font=title_font)

    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)

    max_value = max(values) if max(values) > 0 else 1
    min_value = min(values) if min(values) > 0 else 0
    span = max(1, max_value - min_value)

    def _point(idx: int, value: int) -> tuple[int, int]:
        x = left + int((idx / max(1, len(values) - 1)) * chart_width)
        y = bottom - int(((value - min_value) / span) * chart_height)
        return x, y

    line_points = [_point(idx, value) for idx, value in enumerate(values)]
    draw.line(line_points, fill="#4e79a7", width=3)
    for x, y in line_points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3),
                     fill="#4e79a7", outline="white")

    if payload.get("show_ma"):
        window = 3
        ma_points: list[tuple[int, int]] = []
        for idx in range(len(values)):
            if idx + 1 < window:
                continue
            window_values = values[idx + 1 - window: idx + 1]
            avg = sum(window_values) / window
            ma_points.append(_point(idx, int(avg)))
        if len(ma_points) >= 2:
            draw.line(ma_points, fill="#e15759", width=2)

    spikes = payload.get("spikes") or {}
    spike_up = spikes.get("up")
    spike_down = spikes.get("down")
    for label, index, color in (
        ("Yukselis", spike_up, "#59a14f"),
        ("Dusus", spike_down, "#e15759"),
    ):
        if index is None or index < 0 or index >= len(values):
            continue
        x, y = _point(index, values[index])
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=2)
        draw.text((x + 6, max(top, y - 18)), label, fill=color, font=font)

    for idx, label in enumerate(labels):
        x, _ = _point(idx, values[idx])
        label_box = draw.textbbox((0, 0), label, font=font)
        label_w = label_box[2] - label_box[0]
        draw.text((x - label_w / 2, bottom + 6),
                  label, fill="black", font=font)

    x_label = payload.get("x_label") or ""
    y_label = payload.get("y_label") or ""
    if x_label:
        label_box = draw.textbbox((0, 0), x_label, font=axis_font)
        label_w = label_box[2] - label_box[0]
        draw.text(((width - label_w) / 2, height - 30),
                  x_label, fill="black", font=axis_font)
    if y_label:
        label_box = draw.textbbox((0, 0), y_label, font=axis_font)
        label_h = label_box[3] - label_box[1]
        draw.text((10, top - label_h - 5), y_label,
                  fill="black", font=axis_font)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _render_candlestick_chart(payload: dict) -> bytes | None:
    points = payload.get("points") or []
    labels = payload.get("labels") or []
    if not points or not labels:
        return None

    width = _CHART_WIDTH
    height = _CHART_HEIGHT
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _load_chart_font(12)
    title_font = _load_chart_font(16)
    axis_font = _load_chart_font(12)

    left = 80
    right = width - 40
    top = 70
    bottom = height - 80
    chart_width = right - left
    chart_height = bottom - top

    title = payload.get("title") or ""
    if title:
        title_box = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_box[2] - title_box[0]
        draw.text(((width - title_w) / 2, 20), title,
                  fill="black", font=title_font)

    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)

    highs = [point["high"] for point in points]
    lows = [point["low"] for point in points]
    max_value = max(highs) if max(highs) > 0 else 1
    min_value = min(lows) if min(lows) > 0 else 0
    span = max(1, max_value - min_value)

    def _y(value: int) -> int:
        return bottom - int(((value - min_value) / span) * chart_height)

    count = len(points)
    gap = max(6, int(chart_width * 0.03))
    candle_width = max(10, int((chart_width - gap * (count + 1)) / count))

    for idx, (label, point) in enumerate(zip(labels, points)):
        x0 = left + gap + idx * (candle_width + gap)
        x1 = x0 + candle_width
        open_price = point["open"]
        close_price = point["close"]
        high = point["high"]
        low = point["low"]
        color = "#59a14f" if close_price >= open_price else "#e15759"
        draw.line((x0 + candle_width / 2, _y(high), x0 +
                  candle_width / 2, _y(low)), fill="black", width=1)
        y_open = _y(open_price)
        y_close = _y(close_price)
        y_top = min(y_open, y_close)
        y_bottom = max(y_open, y_close)
        draw.rectangle([x0, y_top, x1, y_bottom], fill=color, outline="black")

        label_box = draw.textbbox((0, 0), label, font=font)
        label_w = label_box[2] - label_box[0]
        draw.text((x0 + (candle_width - label_w) / 2, bottom + 6),
                  label, fill="black", font=font)

    if payload.get("show_ma"):
        close_values = [point["close"] for point in points]
        ma_points: list[tuple[int, int]] = []
        window = 3
        for idx in range(len(close_values)):
            if idx + 1 < window:
                continue
            avg = sum(close_values[idx + 1 - window: idx + 1]) / window
            x = left + gap + idx * (candle_width + gap) + candle_width / 2
            ma_points.append((int(x), _y(int(avg))))
        if len(ma_points) >= 2:
            draw.line(ma_points, fill="#4e79a7", width=2)

    spikes = payload.get("spikes") or {}
    for label, index, color in (
        ("Yukselis", spikes.get("up"), "#59a14f"),
        ("Dusus", spikes.get("down"), "#e15759"),
    ):
        if index is None or index < 0 or index >= len(points):
            continue
        x = left + gap + index * (candle_width + gap) + candle_width / 2
        y = _y(points[index]["close"])
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=2)
        draw.text((x + 6, max(top, y - 18)), label, fill=color, font=font)

    x_label = payload.get("x_label") or ""
    y_label = payload.get("y_label") or ""
    if x_label:
        label_box = draw.textbbox((0, 0), x_label, font=axis_font)
        label_w = label_box[2] - label_box[0]
        draw.text(((width - label_w) / 2, height - 30),
                  x_label, fill="black", font=axis_font)
    if y_label:
        label_box = draw.textbbox((0, 0), y_label, font=axis_font)
        label_h = label_box[3] - label_box[1]
        draw.text((10, top - label_h - 5), y_label,
                  fill="black", font=axis_font)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _truncate_callback_data(value: str, max_bytes: int = 64) -> str:
    data = value.strip()
    if not data:
        return ""
    encoded = data.encode("utf-8")
    if len(encoded) <= max_bytes:
        return data
    truncated = encoded[:max_bytes]
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    return ""


def _build_suggestion_callback_data(text: str) -> str:
    prefix_bytes = _SUGGESTION_PREFIX.encode("utf-8")
    remaining = 64 - len(prefix_bytes)
    if remaining <= 0:
        return ""
    truncated = _truncate_callback_data(text, max_bytes=remaining)
    if not truncated:
        return ""
    return f"{_SUGGESTION_PREFIX}{truncated}"


def _strip_suggestion_prefix(value: str) -> str | None:
    if not value.startswith(_SUGGESTION_PREFIX):
        return None
    stripped = value[len(_SUGGESTION_PREFIX):].strip()
    return stripped if stripped else None


async def _build_suggestion_markup(llm_client, history_text: str | None) -> InlineKeyboardMarkup | None:
    if llm_client is None or not history_text:
        return None
    try:
        suggestions = await _retry_async(
            lambda: asyncio.to_thread(
                llm_client.generate_suggestions, history_text),
            label="Suggestion generation",
            attempts=_SUGGESTION_RETRY_ATTEMPTS,
            base_delay=_SUGGESTION_RETRY_BASE_DELAY,
            max_delay=_SUGGESTION_RETRY_MAX_DELAY,
        )
        if not suggestions:
            return None
        keyboard = []
        row = []
        for suggestion in suggestions:
            text = suggestion.strip()
            if not text:
                continue
            callback_data = _build_suggestion_callback_data(text)
            if not callback_data:
                continue
            row.append(InlineKeyboardButton(
                text=text[:30], callback_data=callback_data))
            if len(row) == 2:
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)
        if not keyboard:
            return None
        return InlineKeyboardMarkup(keyboard)
    except Exception as exc:
        logger.warning("Failed to generate suggestions: %s", exc)
        return None


def _augment_session_context_for_image(
    task: str,
    session_context: dict[str, object] | None,
) -> dict[str, object] | None:
    if not session_context:
        return session_context
    last_image_path = session_context.get("last_image_path")
    if not last_image_path or not _mentions_image_reference(task):
        return session_context
    enriched = dict(session_context)
    enriched["intent_hints"] = {
        "image_reference_path": last_image_path,
        "image_reference_summary": session_context.get("last_image_summary"),
        "note": "The user references the last image with a pronoun.",
    }
    return enriched


def _mentions_image_reference(text: str) -> bool:
    normalized = _normalize_text(text)
    return _contains_any(
        normalized,
        ("bunu", "sunu", "su", "this", "that", "o", "gorsel", "resim"),
    )


def _mentions_explicit_image_reference(text: str) -> bool:
    normalized = _normalize_text(text)
    return _contains_any(normalized, _EXPLICIT_IMAGE_TOKENS)


def _is_analysis_request(text: str) -> bool:
    normalized = _normalize_text(text)
    if _contains_any(normalized, _IMAGE_EDIT_TOKENS):
        return False
    return _contains_any(normalized, _ANALYSIS_TOKENS)


async def _is_analysis_request_with_llm(llm_client, text: str) -> bool:
    if llm_client is not None and text.strip():
        intent = await _classify_image_request(llm_client, text)
        if intent == "ANALYZE":
            return True
        if intent == "EDIT":
            return False
    return _is_analysis_request(text)


async def _classify_image_request(llm_client, text: str) -> str | None:
    prompt = _ANALYSIS_CLASSIFIER_PROMPT.format(message=text.strip())
    try:
        response_text = await asyncio.to_thread(llm_client.generate_text, prompt)
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.warning("Image request classification failed: %s", exc)
        return None
    cleaned = _extract_json_block(response_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(
            "Image request classification parse failed: %s", cleaned[:200])
        return None
    intent = str(data.get("intent", "")).strip().upper()
    if intent in {"ANALYZE", "EDIT", "UNKNOWN"}:
        return intent
    return None


def _extract_json_block(text: str) -> str:
    cleaned = (text or "").strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            if "{" in part:
                cleaned = part
                break
    cleaned = cleaned.strip()
    start = cleaned.find("{")
    if start != -1:
        cleaned = cleaned[start:]
    end = cleaned.rfind("}")
    if end != -1:
        cleaned = cleaned[: end + 1]
    return cleaned


def _retry_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
    return delay + random.uniform(0, base_delay / 2)


async def _retry_async(
    action,
    *,
    label: str,
    attempts: int,
    base_delay: float,
    max_delay: float,
    on_retry=None,
):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await action()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            logger.warning("%s failed (attempt %s/%s): %s",
                           label, attempt, attempts, exc)
            if on_retry is not None:
                try:
                    result = on_retry(attempt, attempts, exc)
                    if inspect.isawaitable(result):
                        await result
                except Exception as notify_exc:
                    logger.warning(
                        "%s retry notification failed: %s", label, notify_exc)
            await asyncio.sleep(_retry_delay(attempt, base_delay, max_delay))
    if last_exc:
        raise last_exc
    raise RuntimeError(f"{label} failed without exception.")


async def _safe_reply_text(message, text: str, *, label: str = "Telegram reply") -> None:
    async def _send():
        return await message.reply_text(text)

    try:
        await _retry_async(
            _send,
            label=label,
            attempts=_TELEGRAM_REPLY_RETRY_ATTEMPTS,
            base_delay=_TELEGRAM_REPLY_RETRY_BASE_DELAY,
            max_delay=_TELEGRAM_REPLY_RETRY_MAX_DELAY,
        )
    except Exception as exc:
        logger.warning("%s failed: %s", label, exc)


async def _retry_image_bytes(
    action,
    *,
    label: str,
    message,
    attempts: int,
    base_delay: float,
    max_delay: float,
) -> bytes:
    async def _wrapped_action():
        image_bytes = await action()
        if not image_bytes:
            raise ValueError("No image bytes received.")
        return image_bytes

    async def _notify_retry(attempt: int, total: int, exc: Exception):
        notice = f"{label} failed. Retrying ({attempt + 1}/{total})..."
        await _safe_reply_text(message, notice, label=f"{label} retry notice")

    return await _retry_async(
        _wrapped_action,
        label=label,
        attempts=attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        on_retry=_notify_retry,
    )


def _format_limit_mb(limit_bytes: int) -> str:
    return f"{int(limit_bytes / _BYTES_PER_MB)} MB"


def _file_too_large_message(file_kind: str, max_bytes: int) -> str:
    return f"{file_kind} file is too large for Gemini. Max size is {_format_limit_mb(max_bytes)}."


async def _build_unsupported_image_message(
    llm_client,
    prompt_text: str,
    mime_label: str,
) -> str:
    fallback = (
        f"Sorry, Gemini doesn't support this image type ({mime_label}). "
        "Please send PNG, JPEG, WEBP, HEIC, or HEIF."
    )
    if llm_client is None or not prompt_text:
        return fallback
    prompt = (
        "The user tried to upload an image format that Gemini cannot process.\n"
        f"User request: {prompt_text}\n"
        f"Image type: {mime_label}\n"
        "Reply in the user's language with a short, friendly message that this image format is not supported. "
        "Mention the supported formats: PNG, JPEG, WEBP, HEIC, HEIF."
    )
    try:
        response = await asyncio.to_thread(llm_client.generate_text, prompt)
    except Exception as exc:
        logger.warning("Unsupported image message generation failed: %s", exc)
        return fallback
    cleaned = response.strip()
    return cleaned or fallback


def _resolve_image_mime(image_path: str, hinted_mime: str | None) -> str | None:
    return hinted_mime or mimetypes.guess_type(image_path)[0]


async def _validate_image_for_gemini(
    llm_client,
    prompt_text: str,
    image_path: str,
    mime_type: str | None,
) -> str | None:
    resolved_mime = _resolve_image_mime(image_path, mime_type)
    if resolved_mime not in GEMINI_IMAGE_MIME_TYPES:
        label = resolved_mime or Path(image_path).suffix.lower() or "unknown"
        return await _build_unsupported_image_message(llm_client, prompt_text, label)
    try:
        size_bytes = Path(image_path).stat().st_size
    except Exception as exc:
        logger.warning("Image size check failed: %s", exc)
        return None
    if size_bytes > GEMINI_FLASH_MAX_INLINE_IMAGE_BYTES:
        return _file_too_large_message("Image", GEMINI_FLASH_MAX_INLINE_IMAGE_BYTES)
    return None


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _normalize_text(value: str) -> str:
    normalized = value.lower()
    return normalized.translate(
        {
            ord("\u00e7"): "c",
            ord("\u011f"): "g",
            ord("\u0131"): "i",
            ord("\u00f6"): "o",
            ord("\u015f"): "s",
            ord("\u00fc"): "u",
        }
    )


async def _start_pc_task(
    task: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: str | None = None,
    require_prefix: bool = True,
    silent_fail: bool = False,
) -> bool:
    message = update.message
    if message is None:
        message = update.effective_message
    if message is None and update.callback_query is not None:
        message = update.callback_query.message
    if message is None:
        return False
    task = task.strip()
    if not task:
        if not silent_fail:
            await message.reply_text("Please provide a task after /pc or pc:.")
        return False

    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        if not silent_fail:
            await message.reply_text("Agent is not ready. Please try again later.")
        return False

    pending = agent_context.pending_request
    if pending is not None and pending.category != "startup":
        if not silent_fail:
            await message.reply_text(
                "There is a pending request already. Please reply to it first or send STOP.",
            )
        return False

    llm_client = context.application.bot_data.get("llm_client")
    if llm_client is None:
        if require_prefix and not silent_fail:
            await message.reply_text("Gemini is not configured. Please set GEMINI_API_KEY.")
        return False

    settings = _resolve_settings(context)
    streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
    show_thoughts = bool(getattr(settings, "show_thoughts", True))
    live_message = None
    if streaming_enabled and hasattr(llm_client, "stream_text"):
        live_message = await message.reply_text("⏳ Planning PC actions...", parse_mode=ParseMode.HTML)
    else:
        await message.reply_text(f"Planning PC actions for: {task}")
    session_context = get_session_context(user_id)
    planner_context = _augment_session_context_for_image(task, session_context)

    async def _runner() -> None:
        from app.pc_agent.controller import run_task_with_approval
        from app.pc_agent.llm_planner import plan_pc_actions

        stream_handler = None
        manager = None
        handler = None
        if live_message is not None:
            manager = MessageManager(
                bot=context.application.bot,
                chat_id=str(message.chat_id),
                message_id=live_message.message_id,
            )
            handler = AsyncStreamHandler(
                manager,
                show_thoughts=show_thoughts,
                response_label="Plan",
                response_as_code=True,
            )

            async def _stream_prompt(prompt: str) -> str:
                result = await handler.stream(
                    llm_client,
                    prompt,
                    include_thoughts=show_thoughts,
                )
                await manager.finalize(handler.format_message(include_thoughts=False))
                return result.response_text

            stream_handler = _stream_prompt

        steps = await plan_pc_actions(
            llm_client,
            task,
            session_context=planner_context,
            stream_handler=stream_handler,
        )
        if not steps:
            if require_prefix and not silent_fail:
                await context.application.bot.send_message(
                    chat_id=message.chat_id,
                    text="Could not build a PC action plan. Try rephrasing the task.",
                )
            return
        result = await run_task_with_approval(
            agent_context=agent_context,
            telegram_app=context.application,
            chat_id=str(message.chat_id),
            task=task,
            context={"steps": steps},
            user_id=user_id,
            llm_client=llm_client,
        )
        final_text = f"PC task finished: {result.reason}."

        suggestion_context = _build_suggestion_context(
            user_id=user_id,
            db=get_database() if user_id else None,
            session_context=get_session_context(user_id),
            user_message=task,
            assistant_message=final_text,
            task_type="computer_use",
            extra_context=f"Result: {result.reason}",
        )
        reply_markup = await _build_suggestion_markup(llm_client, suggestion_context)

        await context.application.bot.send_message(
            chat_id=message.chat_id,
            text=final_text,
            reply_markup=reply_markup,
        )

    context.application.create_task(_runner())
    return True
