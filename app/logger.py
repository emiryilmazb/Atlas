import logging
import os
import re
import sys
import time
from logging.config import dictConfig
from urllib.parse import urlparse

from app.config import get_settings


_HTTPX_REQUEST_RE = re.compile(
    r'HTTP Request:\s+(?P<method>[A-Z]+)\s+(?P<url>\S+)\s+"(?P<status>[^"]+)"'
)


def _shorten_httpx_request_message(message: str, shorten_path: bool) -> str:
    match = _HTTPX_REQUEST_RE.search(message)
    if not match:
        if message.startswith("HTTP Request: "):
            return message[len("HTTP Request: ") :]
        return message

    method = match.group("method")
    url = match.group("url")
    status = match.group("status")
    if shorten_path:
        path = urlparse(url).path
        if path:
            leaf = path.rstrip("/").split("/")[-1]
            short_path = f"/{leaf}" if leaf else path
        else:
            short_path = url
        target = short_path
    else:
        target = url

    return f'{method} {target} "{status}"'


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
        if not record.name.startswith("httpx"):
            return True

        message = record.getMessage()
        short_message = _shorten_httpx_request_message(message, self._enabled)
        if short_message != message:
            record.msg = short_message
            record.args = ()

        return True


class RewriteGenaiAfcLogFilter(logging.Filter):
    _PREFIXES = (
        "AFC is enabled with max remote calls:",
        "AFC tools enabled:",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "google_genai.models":
            return True
        message = record.getMessage()
        return not message.startswith(self._PREFIXES)


class PrettyConsoleFormatter(logging.Formatter):
    _LEVEL_ALIASES = {
        "DEBUG": "DBG",
        "INFO": "INF",
        "WARNING": "WRN",
        "ERROR": "ERR",
        "CRITICAL": "CRT",
    }
    _LEVEL_COLORS = {
        "DEBUG": "\x1b[2m",
        "INFO": "\x1b[32m",
        "WARNING": "\x1b[33m",
        "ERROR": "\x1b[31m",
        "CRITICAL": "\x1b[31;1m",
    }

    def __init__(self, use_colors: bool = True, name_width: int = 28) -> None:
        super().__init__()
        self._use_colors = use_colors
        self._name_width = max(8, name_width)
        self._reset = "\x1b[0m"
        self._dim = "\x1b[2m"
        self._name_color = "\x1b[36m"

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        timestamp = self._format_timestamp(record)
        level = self._format_level(record.levelname)
        logger_name = self._format_logger_name(record.name)
        base = self._format_line(timestamp, level, logger_name, message)

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            base = f"{base}\n{record.exc_text}"
        if record.stack_info:
            base = f"{base}\n{self.formatStack(record.stack_info)}"

        return base

    def _format_timestamp(self, record: logging.LogRecord) -> str:
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        timestamp = f"{time_str}.{int(record.msecs):03d}"
        if self._use_colors:
            return f"{self._dim}{timestamp}{self._reset}"
        return timestamp

    def _format_level(self, levelname: str) -> str:
        short = self._LEVEL_ALIASES.get(levelname, levelname[:3].upper())
        tag = f"[{short}]"
        if not self._use_colors:
            return tag
        color = self._LEVEL_COLORS.get(levelname, "")
        if color:
            return f"{color}{tag}{self._reset}"
        return tag

    def _format_logger_name(self, name: str) -> str:
        if len(name) > self._name_width:
            name = f"...{name[-(self._name_width - 3):]}"
        padded = name.ljust(self._name_width)
        if self._use_colors:
            return f"{self._name_color}{padded}{self._reset}"
        return padded

    def _format_line(self, timestamp: str, level: str, logger_name: str, message: str) -> str:
        if self._use_colors:
            sep = f"{self._dim}|{self._reset}"
            return f"{timestamp} {sep} {level} {sep} {logger_name} {sep} {message}"
        return f"{timestamp} | {level} | {logger_name} | {message}"


def init_logging() -> None:
    settings = get_settings()
    style = (settings.log_style or "standard").strip().lower()
    use_colors = settings.log_color and _supports_color()
    formatter_name = "pretty" if style == "pretty" else "standard"

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                },
                "pretty": {
                    "()": "app.logger.PrettyConsoleFormatter",
                    "use_colors": use_colors,
                    "name_width": 28,
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": formatter_name,
                    "filters": [
                        "suppress_telegram_get_updates",
                        "shorten_httpx_request",
                        "rewrite_genai_afc",
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
                "rewrite_genai_afc": {
                    "()": "app.logger.RewriteGenaiAfcLogFilter",
                },
            },
            "root": {"handlers": ["console"], "level": settings.log_level},
        }
    )


def _supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    stream = sys.stderr
    return bool(getattr(stream, "isatty", lambda: False)())
