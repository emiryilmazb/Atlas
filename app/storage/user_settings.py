from __future__ import annotations

import threading

from app.config import get_settings
from app.storage.database import get_database

_LOCK = threading.Lock()
_ANON_CACHE: dict[str, bool] = {}


def is_anonymous_mode(user_key: str | None) -> bool:
    if not user_key:
        return False
    with _LOCK:
        cached = _ANON_CACHE.get(user_key)
    if cached is not None:
        return cached
    settings = get_settings()
    default_off = bool(getattr(settings, "anonymity_default_off", True))
    default_value = False if default_off else True
    db = get_database()
    enabled = db.get_anonymous_mode(user_key, default_value)
    with _LOCK:
        _ANON_CACHE[user_key] = enabled
    return enabled


def set_anonymous_mode(user_key: str | None, enabled: bool) -> None:
    if not user_key:
        return
    db = get_database()
    db.set_anonymous_mode(user_key, enabled)
    with _LOCK:
        _ANON_CACHE[user_key] = bool(enabled)


def clear_anonymous_cache(user_key: str | None = None) -> None:
    with _LOCK:
        if user_key:
            _ANON_CACHE.pop(user_key, None)
        else:
            _ANON_CACHE.clear()
