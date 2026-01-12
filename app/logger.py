import logging
import re
from logging.config import dictConfig
from urllib.parse import urlparse

from app.config import get_settings


_HTTPX_REQUEST_RE = re.compile(
    r'HTTP Request:\s+(?P<method>[A-Z]+)\s+(?P<url>\S+)\s+"(?P<status>[^"]+)"'
)


def _shorten_httpx_request_message(message: str) -> str:
    match = _HTTPX_REQUEST_RE.search(message)
    if not match:
        return message

    method = match.group("method")
    url = match.group("url")
    status = match.group("status")
    path = urlparse(url).path
    if path:
        leaf = path.rstrip("/").split("/")[-1]
        short_path = f"/{leaf}" if leaf else path
    else:
        short_path = url

    return f'HTTP Request: {method} {short_path} "{status}"'


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


class ShortenHttpxRequestLogFilter(logging.Filter):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__()
        self._enabled = enabled

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._enabled:
            return True
        if not record.name.startswith("httpx"):
            return True

        message = record.getMessage()
        short_message = _shorten_httpx_request_message(message)
        if short_message != message:
            record.msg = short_message
            record.args = ()

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
                    "filters": [
                        "suppress_telegram_get_updates",
                        "shorten_httpx_request",
                    ],
                    "level": settings.log_level,
                }
            },
            "filters": {
                "suppress_telegram_get_updates": {
                    "()": "app.logger.SuppressTelegramGetUpdatesFilter"
                },
                "shorten_httpx_request": {
                    "()": "app.logger.ShortenHttpxRequestLogFilter",
                    "enabled": settings.httpx_log_path_only,
                },
            },
            "root": {"handlers": ["console"], "level": settings.log_level},
        }
    )
