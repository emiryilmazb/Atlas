from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from app.config import get_settings

from .handlers import (
    handle_button,
    handle_document_message,
    handle_pc_command,
    handle_photo_message,
    handle_text_message,
    start,
)


async def _post_init(application: Application) -> None:
    settings = get_settings()
    if not settings.telegram_chat_id:
        raise ValueError("TELEGRAM_CHAT_ID is not set")
    await application.bot.send_message(
        chat_id=settings.telegram_chat_id,
        text="Bot initialized successfully",
    )


def build_application() -> Application:
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set")

    application = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .post_init(_post_init)
        .build()
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("pc", handle_pc_command))
    application.add_handler(CallbackQueryHandler(handle_button))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo_message))
    application.add_handler(
        MessageHandler(filters.Document.ALL & ~filters.Document.IMAGE, handle_document_message)
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    return application
