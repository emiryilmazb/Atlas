import asyncio
from datetime import datetime, timezone
import logging
import mimetypes
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from app.agent.intent_router import RouterDecision, RouterIntent, route_intent
from app.session_manager import (
    clear_session,
    get_session_context,
    record_image_result,
    record_user_message,
    should_reset_context,
)
from app.storage.database import get_database

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
IMAGE_OUTPUT_DIR = ROOT_DIR / "artifacts" / "images"
TEMP_IMAGE_DIR = ROOT_DIR / "artifacts" / "tmp_images"
_ANALYSIS_TOKENS = (
    "analyze",
    "analyse",
    "describe",
    "identify",
    "explain",
    "what is",
    "whats",
    "nedir",
    "tanimla",
    "acikla",
    "incele",
    "ne goruyorsun",
)
_CHAT_CONTEXT_LIMIT = 20

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


def build_main_keyboard() -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton(text=label, callback_data=value)] for label, value in _BUTTONS]
    return InlineKeyboardMarkup(keyboard)


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

    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        return

    pending = agent_context.pending_request
    if pending is None:
        return

    from app.agent.state import HumanReply

    reply = HumanReply(
        question_id=pending.question_id,
        text=str(query.data),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="telegram",
    )
    await agent_context.human_reply_queue.put(reply)


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
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

    if message.text.strip().upper() == "STOP":
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
        has_image=False,
        photo=None,
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
    pending,
) -> None:
    message = update.message
    if message is None:
        return
    llm_client = context.application.bot_data.get("llm_client")
    image_client = context.application.bot_data.get("image_client")
    session_context = get_session_context(user_id)

    decision = await asyncio.to_thread(
        route_intent,
        llm_client,
        message_text,
        has_image=has_image,
        session_context=session_context,
    )
    task_text, has_pc_prefix = _extract_pc_task(message_text)
    if has_pc_prefix:
        decision = RouterDecision(intent=RouterIntent.COMPUTER_USE, reason="forced:pc_prefix")

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
        await _handle_image_generation(message, image_client, message_text, user_id=user_id)
        return
    if decision.intent == RouterIntent.IMAGE_EDIT:
        await _handle_image_edit(
            message,
            image_client,
            message_text,
            user_id=user_id,
            photo=photo,
            session_context=session_context,
        )
        return

    await _handle_chat(message, llm_client, message_text, user_id)


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


async def _handle_chat(message, llm_client, prompt_text: str, user_id: str | None) -> None:
    prompt_text = prompt_text.strip()
    if not prompt_text:
        await message.reply_text("Please send a message.")
        return
    if llm_client is None:
        await message.reply_text("Gemini is not configured. Please set GEMINI_API_KEY.")
        return
    db = get_database() if user_id else None
    history = _safe_get_history(db, user_id, _CHAT_CONTEXT_LIMIT) if db and user_id else []
    timestamp = _timestamp_from_message(message)
    prompt = _build_chat_prompt(prompt_text, history)
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=prompt_text,
            timestamp=timestamp,
        )
    try:
        response_text = await asyncio.to_thread(llm_client.generate_text, prompt)
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.error("Chat response failed: %s", exc)
        await message.reply_text("Chat response failed. Please try again.")
        return
    if not response_text:
        await message.reply_text("No response generated.")
        return
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="assistant",
            content=response_text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    await message.reply_text(response_text)


async def _handle_image_generation(message, image_client, prompt_text: str, user_id: str | None) -> None:
    if image_client is None:
        await message.reply_text("Gemini image model is not configured. Please set GEMINI_API_KEY.")
        return
    prompt_text = prompt_text.strip()
    if not prompt_text:
        await message.reply_text("Please describe the image you want to generate.")
        return
    db = get_database() if user_id else None
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=prompt_text,
            timestamp=_timestamp_from_message(message),
        )
    try:
        image_bytes = await asyncio.to_thread(image_client.generate_image, prompt_text)
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.error("Image generation failed: %s", exc)
        await message.reply_text("Image generation failed. Please try again.")
        return
    if not image_bytes:
        await message.reply_text("Image generation failed. Please try again.")
        return
    output_path = _save_image_bytes(image_bytes, "generated")
    record_image_result(user_id, output_path, prompt_text, source="generated")
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="assistant",
            content="Image generated.",
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_path=output_path,
            file_type="image/png",
        )
    with open(output_path, "rb") as handle:
        await message.reply_photo(photo=handle, caption="Image generated.")


async def _handle_image_edit(
    message,
    image_client,
    prompt_text: str,
    *,
    user_id: str | None,
    photo,
    session_context: dict | None,
) -> None:
    if image_client is None:
        await message.reply_text("Gemini image model is not configured. Please set GEMINI_API_KEY.")
        return
    prompt_text = prompt_text.strip()
    if not prompt_text:
        await message.reply_text("Please describe the edit or analysis you want.")
        return

    reference_path = None
    if photo is not None:
        try:
            reference_path = await _download_photo_to_temp(photo, message)
        except Exception as exc:  # pragma: no cover - network/IO errors
            logger.error("Photo download failed: %s", exc)
            await message.reply_text("Photo download failed. Please try again.")
            return
    else:
        reference_path = session_context.get("last_image_path") if session_context else None
        if not reference_path or not Path(reference_path).exists():
            await message.reply_text("Please send the image you want to edit.")
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
            file_type=mimetypes.guess_type(reference_path)[0] if reference_path else None,
        )

    if _is_analysis_request(prompt_text):
        try:
            response_text = await asyncio.to_thread(
                image_client.analyze_image,
                prompt_text,
                reference_path,
            )
        except Exception as exc:  # pragma: no cover - network/proxy errors
            logger.error("Image analysis failed: %s", exc)
            await message.reply_text("Image analysis failed. Please try again.")
            return
        if response_text:
            if db and user_id:
                _safe_add_message(
                    db,
                    user_id=user_id,
                    role="assistant",
                    content=response_text,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            await message.reply_text(response_text)
        else:
            await message.reply_text("No analysis result.")
        return

    try:
        image_bytes = await asyncio.to_thread(
            image_client.edit_image,
            prompt_text,
            reference_path,
        )
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.error("Image edit failed: %s", exc)
        await message.reply_text("Image edit failed. Please try again.")
        return
    if not image_bytes:
        await message.reply_text("Image edit failed. Please try again.")
        return
    output_path = _save_image_bytes(image_bytes, "edited")
    record_image_result(user_id, output_path, prompt_text, source="edited")
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="assistant",
            content="Image updated.",
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_path=output_path,
            file_type="image/png",
        )
    with open(output_path, "rb") as handle:
        await message.reply_photo(photo=handle, caption="Image updated.")


async def _handle_document_request(
    message,
    context,
    caption: str,
    document,
    user_id: str | None,
) -> None:
    if not caption.strip():
        await message.reply_text("Please add a caption describing what to do with the file.")
        return
    llm_client = context.application.bot_data.get("llm_client")
    if llm_client is None:
        await message.reply_text("Gemini is not configured. Please set GEMINI_API_KEY.")
        return
    temp_path = None
    db = get_database() if user_id else None
    try:
        temp_path = await _download_document_to_temp(document, message)
    except Exception as exc:  # pragma: no cover - network/IO errors
        logger.error("Document download failed: %s", exc)
        await message.reply_text("Document download failed. Please try again.")
        return

    mime_type = getattr(document, "mime_type", None)
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=caption.strip(),
            timestamp=_timestamp_from_message(message),
            file_path=temp_path,
            file_type=mime_type,
        )
    result = await asyncio.to_thread(
        llm_client.generate_with_file,
        caption.strip(),
        temp_path,
        mime_type,
    )
    if result.error:
        await message.reply_text(result.error)
        return
    if not result.text:
        await message.reply_text("No response generated for the document.")
        return
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="assistant",
            content=result.text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    await _send_text_in_chunks(message, result.text, chunk_size=3500)


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
    path = TEMP_IMAGE_DIR / f"telegram_doc_{message.message_id}_{timestamp}{suffix}"
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
    path = TEMP_IMAGE_DIR / f"telegram_{message.message_id}_{timestamp}{suffix}"
    await telegram_file.download_to_drive(custom_path=str(path))
    return str(path)


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


async def _send_text_in_chunks(message, text: str, chunk_size: int = 3500) -> None:
    text = text.strip()
    if not text:
        return
    for start in range(0, len(text), chunk_size):
        await message.reply_text(text[start : start + chunk_size])


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


def _safe_add_message(db, **kwargs) -> None:
    try:
        db.add_message(**kwargs)
    except Exception as exc:
        logger.warning("Message persistence failed: %s", exc)


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


def _is_analysis_request(text: str) -> bool:
    normalized = _normalize_text(text)
    return _contains_any(normalized, _ANALYSIS_TOKENS)


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

    await message.reply_text(f"Planning PC actions for: {task}")
    session_context = get_session_context(user_id)
    planner_context = _augment_session_context_for_image(task, session_context)

    async def _runner() -> None:
        from app.pc_agent.controller import run_task_with_approval
        from app.pc_agent.llm_planner import plan_pc_actions

        steps = await plan_pc_actions(llm_client, task, session_context=planner_context)
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
        await context.application.bot.send_message(
            chat_id=message.chat_id,
            text=f"PC task finished: {result.reason}.",
        )

    context.application.create_task(_runner())
    return True
