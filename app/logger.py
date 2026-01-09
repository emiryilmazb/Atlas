import logging
from logging.config import dictConfig

from app.config import get_settings


class SuppressTelegramGetUpdatesFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith("httpx"):
            return True

        message = record.getMessage()
        if "/getUpdates" in message:
            return False
        if "/editMessageText" in message:
            return False

        return True


def init_logging() -> None:
    settings = get_settings()

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "filters": ["suppress_telegram_get_updates"],
                    "level": settings.log_level,
                }
            },
            "filters": {
                "suppress_telegram_get_updates": {
                    "()": "app.logger.SuppressTelegramGetUpdatesFilter"
                }
            },
            "root": {"handlers": ["console"], "level": settings.log_level},
        }
    )
