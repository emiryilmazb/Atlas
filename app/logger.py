from logging.config import dictConfig

from app.config import get_settings


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
                    "level": settings.log_level,
                }
            },
            "root": {"handlers": ["console"], "level": settings.log_level},
        }
    )
