from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import threading
from typing import Any, Iterable, Mapping

from app.config import get_settings

try:
    import psycopg
    from psycopg.rows import dict_row
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    psycopg = None
    dict_row = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _ConnectionAdapter:
    def __init__(self, connection: Any, reconnect: callable | None = None) -> None:
        self._connection = connection
        self._reconnect = reconnect

    def _prepare_query(self, query: str) -> str:
        return query.replace("?", "%s")

    def _is_closed(self) -> bool:
        try:
            return bool(self._connection.closed)
        except Exception:
            return False

    def _ensure_connection(self) -> None:
        if not self._is_closed():
            return
        if not self._reconnect:
            raise RuntimeError("Database connection is closed.")
        self._connection = self._reconnect()

    def _should_reconnect(self, exc: Exception) -> bool:
        if self._is_closed():
            return True
        if psycopg is None:
            return False
        if isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError)):
            message = str(exc).lower()
            if "connection is closed" in message:
                return True
            if "server closed the connection unexpectedly" in message:
                return True
        return False

    def _execute_with_retry(self, method: str, *args):
        self._ensure_connection()
        try:
            return getattr(self._connection, method)(*args)
        except Exception as exc:
            if self._should_reconnect(exc) and self._reconnect:
                self._connection = self._reconnect()
                return getattr(self._connection, method)(*args)
            raise

    def execute(self, query: str, params: Iterable[Any] | None = None):
        return self._execute_with_retry(
            "execute", self._prepare_query(query), params or ()
        )

    def executemany(self, query: str, params: Iterable[Iterable[Any]]):
        def run():
            if hasattr(self._connection, "executemany"):
                return self._connection.executemany(self._prepare_query(query), params)
            with self._connection.cursor() as cursor:
                return cursor.executemany(self._prepare_query(query), params)

        self._ensure_connection()
        try:
            return run()
        except Exception as exc:
            if self._should_reconnect(exc) and self._reconnect:
                self._connection = self._reconnect()
                return run()
            raise

    def executescript(self, script: str) -> None:
        statements = [stmt.strip() for stmt in script.split(";") if stmt.strip()]
        for statement in statements:
            self._execute_with_retry("execute", statement)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "_ConnectionAdapter":
        self._ensure_connection()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool | None:
        if self._is_closed():
            return False
        if exc_type:
            self._connection.rollback()
            return False
        self._connection.commit()
        return None


@dataclass(frozen=True)
class MessageRow:
    role: str
    content: str
    timestamp: str
    file_path: str | None = None
    file_type: str | None = None
    source: str | None = None
    source_message_id: str | None = None
    source_chat_id: str | None = None


@dataclass(frozen=True)
class MemoryItemRow:
    id: int
    user_key: str
    chat_id: str | None
    kind: str
    title: str | None
    summary: str
    tags: str | None
    embedding: bytes | None
    importance: float
    created_at: str
    last_used_at: str | None


class DatabaseManager:
    def __init__(self, database_url: str, *, connect_timeout: int | None = None) -> None:
        if not database_url:
            raise RuntimeError("DATABASE_URL is required for PostgreSQL support.")
        if psycopg is None:
            raise RuntimeError("psycopg is required for PostgreSQL support.")
        if connect_timeout is None:
            raw_timeout = os.getenv("DB_CONNECT_TIMEOUT") or os.getenv("PGCONNECT_TIMEOUT") or ""
            try:
                connect_timeout = int(raw_timeout)
            except ValueError:
                connect_timeout = None
        self._lock = threading.RLock()
        self._database_url = database_url
        self._connect_timeout = connect_timeout
        self._connection = _ConnectionAdapter(self._open_connection(), self._reconnect)

    def _open_connection(self) -> Any:
        connect_kwargs: dict[str, Any] = {"row_factory": dict_row}
        if self._connect_timeout:
            connect_kwargs["connect_timeout"] = self._connect_timeout
        connection = psycopg.connect(self._database_url, **connect_kwargs)
        connection.execute("SET TIME ZONE 'UTC'")
        connection.commit()
        return connection

    def _reconnect(self) -> Any:
        with self._lock:
            return self._open_connection()

    def initialize(self) -> None:
        schema = self._schema_postgres()
        with self._lock, self._connection:
            self._connection.executescript(schema)
        self._ensure_message_columns()
        self._ensure_memory_tables()

    def _schema_postgres(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS messages (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            file_path TEXT,
            file_type TEXT,
            source TEXT,
            source_message_id TEXT,
            source_chat_id TEXT,
            CHECK (role IN ('user', 'assistant', 'system', 'tool'))
        );
        CREATE INDEX IF NOT EXISTS idx_messages_user_time
            ON messages(user_id, timestamp);

        CREATE TABLE IF NOT EXISTS sessions (
            user_id TEXT PRIMARY KEY,
            active_app TEXT,
            last_action TEXT,
            last_summary TEXT,
            last_image_path TEXT,
            last_image_prompt TEXT,
            last_image_summary TEXT,
            last_image_source TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS session_pending_details (
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (user_id, key),
            FOREIGN KEY (user_id) REFERENCES sessions(user_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS session_short_term_memory (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            user_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES sessions(user_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_session_short_term_user_pos
            ON session_short_term_memory(user_id, position);

        CREATE TABLE IF NOT EXISTS session_images (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            user_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            image_prompt TEXT,
            image_summary TEXT,
            image_source TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES sessions(user_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_session_images_user_pos
            ON session_images(user_id, position);

        CREATE TABLE IF NOT EXISTS personal_facts (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            position INTEGER NOT NULL,
            intent TEXT,
            question TEXT,
            value TEXT NOT NULL,
            source TEXT,
            confirmed_at TEXT,
            country TEXT
        );

        CREATE TABLE IF NOT EXISTS answer_history (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            position INTEGER NOT NULL,
            intent TEXT,
            question TEXT,
            answer TEXT,
            source TEXT,
            submitted_at TEXT,
            pending INTEGER NOT NULL DEFAULT 0,
            category TEXT
        );

        CREATE TABLE IF NOT EXISTS cv_profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            summary TEXT,
            work_type TEXT
        );
        CREATE TABLE IF NOT EXISTS cv_skill (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            profile_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            skill TEXT NOT NULL,
            FOREIGN KEY (profile_id) REFERENCES cv_profile(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS cv_experience (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            profile_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            company TEXT,
            role TEXT,
            duration TEXT,
            FOREIGN KEY (profile_id) REFERENCES cv_profile(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS cv_experience_technology (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            experience_id BIGINT NOT NULL,
            position INTEGER NOT NULL,
            technology TEXT NOT NULL,
            FOREIGN KEY (experience_id) REFERENCES cv_experience(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS cv_experience_responsibility (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            experience_id BIGINT NOT NULL,
            position INTEGER NOT NULL,
            responsibility TEXT NOT NULL,
            FOREIGN KEY (experience_id) REFERENCES cv_experience(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS cv_experience_achievement (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            experience_id BIGINT NOT NULL,
            position INTEGER NOT NULL,
            achievement TEXT NOT NULL,
            FOREIGN KEY (experience_id) REFERENCES cv_experience(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS cv_education (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            profile_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            institution TEXT,
            degree TEXT,
            field TEXT,
            duration TEXT,
            gpa TEXT,
            FOREIGN KEY (profile_id) REFERENCES cv_profile(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS cv_project (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            profile_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            name TEXT,
            role TEXT,
            duration TEXT,
            FOREIGN KEY (profile_id) REFERENCES cv_profile(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS cv_project_technology (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            project_id BIGINT NOT NULL,
            position INTEGER NOT NULL,
            technology TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES cv_project(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS cv_project_description (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            project_id BIGINT NOT NULL,
            position INTEGER NOT NULL,
            description TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES cv_project(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS cv_preference_location (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            profile_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            location TEXT NOT NULL,
            FOREIGN KEY (profile_id) REFERENCES cv_profile(id) ON DELETE CASCADE
        );
        """

    def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        timestamp: str | None = None,
        file_path: str | None = None,
        file_type: str | None = None,
        source: str | None = None,
        source_message_id: str | None = None,
        source_chat_id: str | None = None,
    ) -> int:
        timestamp = timestamp or _utc_now()
        with self._lock, self._connection:
            return self._insert_and_get_id(
                """
                INSERT INTO messages (
                    user_id, role, content, timestamp, file_path, file_type,
                    source, source_message_id, source_chat_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    role,
                    content,
                    timestamp,
                    file_path,
                    file_type,
                    source,
                    source_message_id,
                    source_chat_id,
                ),
            )

    def get_recent_messages(self, user_id: str, limit: int) -> list[MessageRow]:
        rows = self._fetchall(
            """
            SELECT role, content, timestamp, file_path, file_type,
                   source, source_message_id, source_chat_id
            FROM messages
            WHERE user_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return [
            MessageRow(
                role=row["role"],
                content=row["content"],
                timestamp=row["timestamp"],
                file_path=row["file_path"],
                file_type=row["file_type"],
                source=row["source"],
                source_message_id=row["source_message_id"],
                source_chat_id=row["source_chat_id"],
            )
            for row in rows
        ]

    def get_message_by_source_id(
        self,
        user_id: str,
        source: str,
        source_message_id: str,
        source_chat_id: str | None,
    ) -> MessageRow | None:
        if not user_id or not source_message_id:
            return None
        where = "user_id = ? AND source = ? AND source_message_id = ?"
        params: list[Any] = [user_id, source, source_message_id]
        if source_chat_id is not None:
            where += " AND source_chat_id = ?"
            params.append(source_chat_id)
        rows = self._fetchall(
            f"""
            SELECT role, content, timestamp, file_path, file_type,
                   source, source_message_id, source_chat_id
            FROM messages
            WHERE {where}
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            params,
        )
        if not rows:
            return None
        row = rows[0]
        return MessageRow(
            role=row["role"],
            content=row["content"],
            timestamp=row["timestamp"],
            file_path=row["file_path"],
            file_type=row["file_type"],
            source=row["source"],
            source_message_id=row["source_message_id"],
            source_chat_id=row["source_chat_id"],
        )

    def find_message_by_content(
        self,
        user_id: str,
        content: str,
        role: str | None = None,
    ) -> MessageRow | None:
        if not user_id or not content:
            return None
        where = "user_id = ? AND content = ?"
        params: list[Any] = [user_id, content]
        if role:
            where += " AND role = ?"
            params.append(role)
        rows = self._fetchall(
            f"""
            SELECT role, content, timestamp, file_path, file_type,
                   source, source_message_id, source_chat_id
            FROM messages
            WHERE {where}
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            params,
        )
        if not rows:
            return None
        row = rows[0]
        return MessageRow(
            role=row["role"],
            content=row["content"],
            timestamp=row["timestamp"],
            file_path=row["file_path"],
            file_type=row["file_type"],
            source=row["source"],
            source_message_id=row["source_message_id"],
            source_chat_id=row["source_chat_id"],
        )

    def update_message_source(
        self,
        message_id: int,
        *,
        source: str | None,
        source_message_id: str | None,
        source_chat_id: str | None,
    ) -> None:
        if not message_id:
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE messages
                SET source = ?, source_message_id = ?, source_chat_id = ?
                WHERE id = ?
                """,
                (source, source_message_id, source_chat_id, message_id),
            )

    def delete_messages(self, user_id: str) -> list[str]:
        rows = self._fetchall(
            """
            SELECT DISTINCT file_path
            FROM messages
            WHERE user_id = ? AND file_path IS NOT NULL
            """,
            (user_id,),
        )
        file_paths = [row["file_path"] for row in rows if row["file_path"]]
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        return file_paths

    def get_anonymous_mode(self, user_key: str, default: bool = False) -> bool:
        if not user_key:
            return default
        row = self._fetchone(
            "SELECT anonymous_mode FROM user_settings WHERE user_key = ?",
            (user_key,),
        )
        if row is None:
            return default
        return bool(row["anonymous_mode"])

    def set_anonymous_mode(self, user_key: str, enabled: bool) -> None:
        if not user_key:
            return
        updated_at = _utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO user_settings (user_key, anonymous_mode, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_key) DO UPDATE SET
                    anonymous_mode = excluded.anonymous_mode,
                    updated_at = excluded.updated_at
                """,
                (user_key, 1 if enabled else 0, updated_at),
            )

    def get_user_profile(self, user_key: str) -> str | None:
        if not user_key:
            return None
        row = self._fetchone(
            "SELECT profile_json FROM user_profile WHERE user_key = ?",
            (user_key,),
        )
        if row is None:
            return None
        return row["profile_json"]

    def set_user_profile(self, user_key: str, profile_json: str) -> None:
        if not user_key:
            return
        updated_at = _utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO user_profile (user_key, profile_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_key) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
                """,
                (user_key, profile_json, updated_at),
            )

    def add_memory_item(
        self,
        *,
        user_key: str,
        chat_id: str | None,
        kind: str,
        title: str | None,
        summary: str,
        tags: str | None,
        embedding: bytes | None,
        importance: float,
    ) -> int:
        created_at = _utc_now()
        with self._lock, self._connection:
            return self._insert_and_get_id(
                """
                INSERT INTO memory_items (
                    user_key, chat_id, kind, title, summary, tags,
                    embedding, importance, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_key,
                    chat_id,
                    kind,
                    title,
                    summary,
                    tags,
                    embedding,
                    float(importance),
                    created_at,
                ),
            )

    def get_memory_items(self, user_key: str, limit: int = 500) -> list[MemoryItemRow]:
        if not user_key:
            return []
        rows = self._fetchall(
            """
            SELECT id, user_key, chat_id, kind, title, summary, tags,
                   embedding, importance, created_at, last_used_at
            FROM memory_items
            WHERE user_key = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (user_key, limit),
        )
        return [
            MemoryItemRow(
                id=int(row["id"]),
                user_key=row["user_key"],
                chat_id=row["chat_id"],
                kind=row["kind"],
                title=row["title"],
                summary=row["summary"],
                tags=row["tags"],
                embedding=row["embedding"],
                importance=float(row["importance"] or 0.0),
                created_at=row["created_at"],
                last_used_at=row["last_used_at"],
            )
            for row in rows
        ]

    def update_memory_last_used(self, ids: Iterable[int]) -> None:
        id_list = [int(item) for item in ids if item]
        if not id_list:
            return
        updated_at = _utc_now()
        with self._lock, self._connection:
            self._connection.executemany(
                "UPDATE memory_items SET last_used_at = ? WHERE id = ?",
                [(updated_at, item) for item in id_list],
            )

    def delete_memory_item(self, user_key: str, memory_id: int) -> bool:
        if not user_key or not memory_id:
            return False
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM memory_items WHERE user_key = ? AND id = ?",
                (user_key, int(memory_id)),
            )
            return cursor.rowcount > 0

    def load_sessions(self) -> dict[str, dict[str, Any]]:
        sessions = {}
        session_rows = self._fetchall(
            """
            SELECT user_id, active_app, last_action, last_summary, last_image_path,
                   last_image_prompt, last_image_summary, last_image_source
            FROM sessions
            """,
        )
        for row in session_rows:
            sessions[row["user_id"]] = {
                "active_app": row["active_app"],
                "last_action": row["last_action"],
                "last_summary": row["last_summary"],
                "last_image_path": row["last_image_path"],
                "last_image_prompt": row["last_image_prompt"],
                "last_image_summary": row["last_image_summary"],
                "last_image_source": row["last_image_source"],
                "pending_details": {},
                "short_term_memory": [],
                "recent_images": [],
            }

        pending_rows = self._fetchall(
            "SELECT user_id, key, value FROM session_pending_details",
        )
        for row in pending_rows:
            payload = sessions.get(row["user_id"])
            if payload is None:
                continue
            payload["pending_details"][row["key"]] = row["value"]

        memory_rows = self._fetchall(
            """
            SELECT user_id, message
            FROM session_short_term_memory
            ORDER BY position ASC, id ASC
            """,
        )
        for row in memory_rows:
            payload = sessions.get(row["user_id"])
            if payload is None:
                continue
            payload["short_term_memory"].append(row["message"])

        image_rows = self._fetchall(
            """
            SELECT user_id, image_path, image_prompt, image_summary, image_source
            FROM session_images
            ORDER BY position ASC, id ASC
            """,
        )
        for row in image_rows:
            payload = sessions.get(row["user_id"])
            if payload is None:
                continue
            payload["recent_images"].append(
                {
                    "path": row["image_path"],
                    "prompt": row["image_prompt"],
                    "summary": row["image_summary"],
                    "source": row["image_source"],
                }
            )
        return sessions

    def save_session_state(
        self,
        user_id: str,
        *,
        active_app: str | None,
        last_action: str | None,
        pending_details: dict[str, Any],
        short_term_memory: list[str],
        last_summary: str | None,
        last_image_path: str | None,
        last_image_prompt: str | None,
        last_image_summary: str | None,
        last_image_source: str | None,
        recent_images: list[dict[str, Any]],
    ) -> None:
        updated_at = _utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO sessions (
                    user_id, active_app, last_action, last_summary, last_image_path,
                    last_image_prompt, last_image_summary, last_image_source, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    active_app = excluded.active_app,
                    last_action = excluded.last_action,
                    last_summary = excluded.last_summary,
                    last_image_path = excluded.last_image_path,
                    last_image_prompt = excluded.last_image_prompt,
                    last_image_summary = excluded.last_image_summary,
                    last_image_source = excluded.last_image_source,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    active_app,
                    last_action,
                    last_summary,
                    last_image_path,
                    last_image_prompt,
                    last_image_summary,
                    last_image_source,
                    updated_at,
                ),
            )
            self._connection.execute(
                "DELETE FROM session_pending_details WHERE user_id = ?",
                (user_id,),
            )
            if pending_details:
                self._connection.executemany(
                    """
                    INSERT INTO session_pending_details (user_id, key, value)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (user_id, str(key), str(value))
                        for key, value in pending_details.items()
                    ],
                )
            self._connection.execute(
                "DELETE FROM session_short_term_memory WHERE user_id = ?",
                (user_id,),
            )
            for position, message in enumerate(short_term_memory):
                self._connection.execute(
                    """
                    INSERT INTO session_short_term_memory (user_id, position, message, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, position, message, updated_at),
                )
            self._connection.execute(
                "DELETE FROM session_images WHERE user_id = ?",
                (user_id,),
            )
            for position, image in enumerate(recent_images):
                image_path = image.get("path") if isinstance(image, dict) else None
                if not image_path:
                    continue
                self._connection.execute(
                    """
                    INSERT INTO session_images (
                        user_id, position, image_path, image_prompt,
                        image_summary, image_source, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        position,
                        image_path,
                        image.get("prompt") if isinstance(image, dict) else None,
                        image.get("summary") if isinstance(image, dict) else None,
                        image.get("source") if isinstance(image, dict) else None,
                        updated_at,
                    ),
                )

    def delete_session(self, user_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def replace_cv_profile(self, profile: dict[str, Any]) -> None:
        summary = str(profile.get("summary", "") or "")
        preferences = profile.get("preferences") or {}
        work_type = str(preferences.get("work_type", "") or "")
        skills = profile.get("skills", []) or []
        experience = profile.get("experience", []) or []
        education = profile.get("education", []) or []
        projects = profile.get("projects", []) or []
        locations = preferences.get("locations", []) or []

        with self._lock, self._connection:
            self._clear_cv_profile()
            self._connection.execute(
                "INSERT INTO cv_profile (id, summary, work_type) VALUES (1, ?, ?)",
                (summary, work_type),
            )
            for position, skill in enumerate(skills):
                self._connection.execute(
                    """
                    INSERT INTO cv_skill (profile_id, position, skill)
                    VALUES (1, ?, ?)
                    """,
                    (position, str(skill)),
                )
            for position, exp in enumerate(experience):
                exp_id = self._insert_and_get_id(
                    """
                    INSERT INTO cv_experience (profile_id, position, company, role, duration)
                    VALUES (1, ?, ?, ?, ?)
                    """,
                    (
                        position,
                        exp.get("company"),
                        exp.get("role"),
                        exp.get("duration"),
                    ),
                )
                for index, item in enumerate(exp.get("technologies", []) or []):
                    self._connection.execute(
                        """
                        INSERT INTO cv_experience_technology (experience_id, position, technology)
                        VALUES (?, ?, ?)
                        """,
                        (exp_id, index, str(item)),
                    )
                for index, item in enumerate(exp.get("responsibilities", []) or []):
                    self._connection.execute(
                        """
                        INSERT INTO cv_experience_responsibility (experience_id, position, responsibility)
                        VALUES (?, ?, ?)
                        """,
                        (exp_id, index, str(item)),
                    )
                for index, item in enumerate(exp.get("achievements", []) or []):
                    self._connection.execute(
                        """
                        INSERT INTO cv_experience_achievement (experience_id, position, achievement)
                        VALUES (?, ?, ?)
                        """,
                        (exp_id, index, str(item)),
                    )
            for position, edu in enumerate(education):
                self._connection.execute(
                    """
                    INSERT INTO cv_education (profile_id, position, institution, degree, field, duration, gpa)
                    VALUES (1, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        position,
                        edu.get("institution"),
                        edu.get("degree"),
                        edu.get("field"),
                        edu.get("duration"),
                        edu.get("gpa"),
                    ),
                )
            for position, project in enumerate(projects):
                project_id = self._insert_and_get_id(
                    """
                    INSERT INTO cv_project (profile_id, position, name, role, duration)
                    VALUES (1, ?, ?, ?, ?)
                    """,
                    (
                        position,
                        project.get("name"),
                        project.get("role"),
                        project.get("duration"),
                    ),
                )
                for index, item in enumerate(project.get("technologies", []) or []):
                    self._connection.execute(
                        """
                        INSERT INTO cv_project_technology (project_id, position, technology)
                        VALUES (?, ?, ?)
                        """,
                        (project_id, index, str(item)),
                    )
                for index, item in enumerate(project.get("description", []) or []):
                    self._connection.execute(
                        """
                        INSERT INTO cv_project_description (project_id, position, description)
                        VALUES (?, ?, ?)
                        """,
                        (project_id, index, str(item)),
                    )
            for position, location in enumerate(locations):
                self._connection.execute(
                    """
                    INSERT INTO cv_preference_location (profile_id, position, location)
                    VALUES (1, ?, ?)
                    """,
                    (position, str(location)),
                )

    def load_cv_profile(self) -> dict[str, Any]:
        row = self._fetchone("SELECT summary, work_type FROM cv_profile WHERE id = 1")
        if row is None:
            return {}
        profile = {
            "summary": row["summary"] or "",
            "skills": [],
            "experience": [],
            "education": [],
            "projects": [],
            "preferences": {"locations": [], "work_type": row["work_type"] or ""},
        }
        skills = self._fetchall(
            "SELECT skill FROM cv_skill WHERE profile_id = 1 ORDER BY position ASC, id ASC",
        )
        profile["skills"] = [row["skill"] for row in skills]

        experiences = self._fetchall(
            """
            SELECT id, company, role, duration
            FROM cv_experience
            WHERE profile_id = 1
            ORDER BY position ASC, id ASC
            """,
        )
        tech_rows = self._fetchall(
            """
            SELECT experience_id, technology
            FROM cv_experience_technology
            ORDER BY position ASC, id ASC
            """,
        )
        resp_rows = self._fetchall(
            """
            SELECT experience_id, responsibility
            FROM cv_experience_responsibility
            ORDER BY position ASC, id ASC
            """,
        )
        ach_rows = self._fetchall(
            """
            SELECT experience_id, achievement
            FROM cv_experience_achievement
            ORDER BY position ASC, id ASC
            """,
        )
        tech_map = self._group_rows(tech_rows, "experience_id", "technology")
        resp_map = self._group_rows(resp_rows, "experience_id", "responsibility")
        ach_map = self._group_rows(ach_rows, "experience_id", "achievement")
        for exp in experiences:
            exp_id = exp["id"]
            profile["experience"].append(
                {
                    "company": exp["company"],
                    "role": exp["role"],
                    "duration": exp["duration"],
                    "technologies": tech_map.get(exp_id, []),
                    "responsibilities": resp_map.get(exp_id, []),
                    "achievements": ach_map.get(exp_id, []),
                }
            )

        education_rows = self._fetchall(
            """
            SELECT institution, degree, field, duration, gpa
            FROM cv_education
            WHERE profile_id = 1
            ORDER BY position ASC, id ASC
            """,
        )
        profile["education"] = [
            {
                "institution": row["institution"],
                "degree": row["degree"],
                "field": row["field"],
                "duration": row["duration"],
                "gpa": row["gpa"],
            }
            for row in education_rows
        ]

        project_rows = self._fetchall(
            """
            SELECT id, name, role, duration
            FROM cv_project
            WHERE profile_id = 1
            ORDER BY position ASC, id ASC
            """,
        )
        project_tech_rows = self._fetchall(
            """
            SELECT project_id, technology
            FROM cv_project_technology
            ORDER BY position ASC, id ASC
            """,
        )
        project_desc_rows = self._fetchall(
            """
            SELECT project_id, description
            FROM cv_project_description
            ORDER BY position ASC, id ASC
            """,
        )
        project_tech_map = self._group_rows(project_tech_rows, "project_id", "technology")
        project_desc_map = self._group_rows(project_desc_rows, "project_id", "description")
        for project in project_rows:
            project_id = project["id"]
            profile["projects"].append(
                {
                    "name": project["name"],
                    "role": project["role"],
                    "duration": project["duration"],
                    "technologies": project_tech_map.get(project_id, []),
                    "description": project_desc_map.get(project_id, []),
                }
            )

        location_rows = self._fetchall(
            """
            SELECT location
            FROM cv_preference_location
            WHERE profile_id = 1
            ORDER BY position ASC, id ASC
            """,
        )
        profile["preferences"]["locations"] = [row["location"] for row in location_rows]
        return profile

    def add_personal_fact(
        self,
        *,
        value: str,
        intent: str | None = None,
        question: str | None = None,
        source: str | None = None,
        confirmed_at: str | None = None,
        country: str | None = None,
        position: int | None = None,
    ) -> int:
        confirmed_at = confirmed_at or _utc_now()
        position = position if position is not None else self._next_position("personal_facts")
        with self._lock, self._connection:
            return self._insert_and_get_id(
                """
                INSERT INTO personal_facts
                    (position, intent, question, value, source, confirmed_at, country)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (position, intent, question, value, source, confirmed_at, country),
            )

    def load_personal_facts(self) -> list[dict[str, Any]]:
        rows = self._fetchall(
            """
            SELECT intent, question, value, source, confirmed_at, country
            FROM personal_facts
            ORDER BY position ASC, id ASC
            """,
        )
        return [dict(row) for row in rows]

    def add_answer_history(
        self,
        *,
        intent: str | None,
        question: str | None,
        answer: str | None,
        source: str | None,
        submitted_at: str | None,
        pending: bool = False,
        category: str | None = None,
        position: int | None = None,
    ) -> int:
        submitted_at = submitted_at or _utc_now()
        position = position if position is not None else self._next_position("answer_history")
        with self._lock, self._connection:
            return self._insert_and_get_id(
                """
                INSERT INTO answer_history
                    (position, intent, question, answer, source, submitted_at, pending, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position,
                    intent,
                    question,
                    answer,
                    source,
                    submitted_at,
                    1 if pending else 0,
                    category,
                ),
            )

    def load_answer_history(self) -> list[dict[str, Any]]:
        rows = self._fetchall(
            """
            SELECT intent, question, answer, source, submitted_at, pending, category
            FROM answer_history
            ORDER BY position ASC, id ASC
            """,
        )
        return [dict(row) for row in rows]

    def has_gmail_notification(self, user_key: str, message_id: str) -> bool:
        if not user_key or not message_id:
            return False
        row = self._fetchone(
            """
            SELECT 1 FROM gmail_notifications
            WHERE user_key = ? AND message_id = ?
            LIMIT 1
            """,
            (user_key, message_id),
        )
        return row is not None

    def mark_gmail_notified(self, user_key: str, message_id: str, label: str | None = None) -> None:
        if not user_key or not message_id:
            return
        notified_at = _utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO gmail_notifications (user_key, message_id, label, notified_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_key, message_id) DO UPDATE SET
                    label = excluded.label,
                    notified_at = excluded.notified_at
                """,
                (user_key, message_id, label, notified_at),
            )

    def count_table(self, table: str) -> int:
        row = self._fetchone(f"SELECT COUNT(*) AS count FROM {table}")
        if row is None:
            return 0
        return int(row["count"])

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _clear_cv_profile(self) -> None:
        self._connection.execute("DELETE FROM cv_skill")
        self._connection.execute("DELETE FROM cv_experience_technology")
        self._connection.execute("DELETE FROM cv_experience_responsibility")
        self._connection.execute("DELETE FROM cv_experience_achievement")
        self._connection.execute("DELETE FROM cv_experience")
        self._connection.execute("DELETE FROM cv_education")
        self._connection.execute("DELETE FROM cv_project_technology")
        self._connection.execute("DELETE FROM cv_project_description")
        self._connection.execute("DELETE FROM cv_project")
        self._connection.execute("DELETE FROM cv_preference_location")
        self._connection.execute("DELETE FROM cv_profile")

    def _next_position(self, table: str) -> int:
        row = self._fetchone(f"SELECT COALESCE(MAX(position), -1) AS max_pos FROM {table}")
        if row is None:
            return 0
        return int(row["max_pos"]) + 1

    def _insert_and_get_id(self, query: str, params: Iterable[Any]) -> int:
        cursor = self._connection.execute(f"{query} RETURNING id", params)
        row = cursor.fetchone()
        if row is None:
            return 0
        if isinstance(row, Mapping):
            value = row.get("id")
            if value is None and row:
                value = next(iter(row.values()))
            return int(value or 0)
        return int(row[0])

    def _fetchall(self, query: str, params: Iterable[Any] | None = None) -> list[Mapping[str, Any]]:
        with self._lock:
            cursor = self._connection.execute(query, params or ())
            return cursor.fetchall()

    def _fetchone(self, query: str, params: Iterable[Any] | None = None) -> Mapping[str, Any] | None:
        with self._lock:
            cursor = self._connection.execute(query, params or ())
            return cursor.fetchone()

    def _ensure_message_columns(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "ALTER TABLE messages ADD COLUMN IF NOT EXISTS source TEXT"
            )
            self._connection.execute(
                "ALTER TABLE messages ADD COLUMN IF NOT EXISTS source_message_id TEXT"
            )
            self._connection.execute(
                "ALTER TABLE messages ADD COLUMN IF NOT EXISTS source_chat_id TEXT"
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_source
                ON messages(user_id, source, source_chat_id, source_message_id)
                """
            )

    def _ensure_memory_tables(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    user_key TEXT NOT NULL,
                    chat_id TEXT,
                    kind TEXT NOT NULL DEFAULT 'summary',
                    title TEXT,
                    summary TEXT NOT NULL,
                    tags TEXT,
                    embedding BYTEA,
                    importance DOUBLE PRECISION DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profile (
                    user_key TEXT PRIMARY KEY,
                    profile_json TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_key TEXT PRIMARY KEY,
                    anonymous_mode INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_events (
                    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    user_key TEXT,
                    event_type TEXT,
                    payload TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS gmail_notifications (
                    user_key TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    label TEXT,
                    notified_at TEXT NOT NULL,
                    PRIMARY KEY (user_key, message_id)
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_items_user_created
                ON memory_items(user_key, created_at)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_items_user_last_used
                ON memory_items(user_key, last_used_at)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_gmail_notifications_user_time
                ON gmail_notifications(user_key, notified_at)
                """
            )

    @staticmethod
    def _group_rows(
        rows: list[Mapping[str, Any]], key_field: str, value_field: str
    ) -> dict[int, list[str]]:
        grouped: dict[int, list[str]] = {}
        for row in rows:
            key = int(row[key_field])
            grouped.setdefault(key, []).append(row[value_field])
        return grouped


_DB_INSTANCE: DatabaseManager | None = None
_DB_LOCK = threading.Lock()


def get_database(database_url: str | None = None) -> DatabaseManager:
    global _DB_INSTANCE
    with _DB_LOCK:
        if _DB_INSTANCE is None:
            settings = get_settings()
            database_url = database_url or getattr(settings, "database_url", "") or ""
            if not database_url:
                raise RuntimeError("DATABASE_URL must be set to use PostgreSQL.")
            _DB_INSTANCE = DatabaseManager(database_url=database_url)
            _DB_INSTANCE.initialize()
        return _DB_INSTANCE
