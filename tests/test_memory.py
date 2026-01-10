import numpy as np

from app.memory.embeddings import cosine_similarity, deserialize_embedding, serialize_embedding
from app.memory.long_term import extract_json_payload
from app.storage.database import DatabaseManager
from app.storage import user_settings


def test_embedding_roundtrip() -> None:
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    payload = serialize_embedding(vector)
    restored = deserialize_embedding(payload)
    assert np.allclose(vector, restored)


def test_cosine_similarity() -> None:
    vec_a = np.array([1.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0], dtype=np.float32)
    vec_c = np.array([1.0, 1.0], dtype=np.float32)
    assert cosine_similarity(vec_a, vec_b) == 0.0
    assert round(cosine_similarity(vec_a, vec_c), 6) == 0.707107


def test_extract_json_payload() -> None:
    raw = "Sure.```json\\n{\\\"title\\\": \\\"Hello\\\", \\\"summary\\\": \\\"World\\\"}\\n```"
    payload = extract_json_payload(raw)
    assert payload is not None
    assert payload["title"] == "Hello"
    assert payload["summary"] == "World"


def test_anonymous_mode_toggle(tmp_path, monkeypatch) -> None:
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    monkeypatch.setattr(user_settings, "get_database", lambda: db)
    user_settings.clear_anonymous_cache()
    user_settings.set_anonymous_mode("user-1", True)
    assert user_settings.is_anonymous_mode("user-1") is True
    user_settings.set_anonymous_mode("user-1", False)
    assert user_settings.is_anonymous_mode("user-1") is False
