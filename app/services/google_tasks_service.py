from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from googleapiclient.discovery import build

from app.services.gmail_auth_service import GmailAuthConfig, GmailAuthService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskList:
    tasklist_id: str
    title: str


@dataclass(frozen=True)
class TaskItem:
    task_id: str
    title: str
    notes: str | None
    status: str | None
    due: str | None


class TasksService:
    def __init__(self, auth_service: GmailAuthService) -> None:
        self._auth = auth_service
        self._service = None
        self._tasklist_cache: dict[str, str] = {}
        self._default_tasklist_id: str | None = None

    def ensure_credentials(self) -> None:
        self._auth.get_credentials()

    def reset(self) -> None:
        self._service = None
        self._tasklist_cache = {}
        self._default_tasklist_id = None

    def list_tasklists(self, *, max_results: int = 25) -> list[TaskList]:
        service = self._get_service()
        response = service.tasklists().list(
            maxResults=max(1, min(25, max_results)),
        ).execute()
        items = response.get("items", []) or []
        tasklists = [_normalize_tasklist(item) for item in items if isinstance(item, dict)]
        self._cache_tasklists(tasklists)
        return tasklists

    def list_tasks(
        self,
        *,
        tasklist_id: str | None = None,
        tasklist_title: str | None = None,
        show_completed: bool = False,
        due_min: str | None = None,
        due_max: str | None = None,
        max_results: int = 20,
    ) -> list[TaskItem]:
        resolved = self.resolve_tasklist_id(tasklist_id, tasklist_title)
        if not resolved:
            raise ValueError("Task list not found.")
        service = self._get_service()
        params = {
            "tasklist": resolved,
            "maxResults": max(1, min(50, max_results)),
            "showCompleted": bool(show_completed),
        }
        if due_min:
            params["dueMin"] = due_min
        if due_max:
            params["dueMax"] = due_max
        response = service.tasks().list(**params).execute()
        items = response.get("items", []) or []
        return [_normalize_task(item) for item in items if isinstance(item, dict)]

    def create_task(
        self,
        *,
        title: str,
        notes: str | None = None,
        due: str | None = None,
        tasklist_id: str | None = None,
        tasklist_title: str | None = None,
    ) -> TaskItem:
        resolved = self.resolve_tasklist_id(tasklist_id, tasklist_title)
        if not resolved:
            raise ValueError("Task list not found.")
        body: dict = {"title": title}
        if notes:
            body["notes"] = notes
        if due:
            body["due"] = due
        service = self._get_service()
        created = service.tasks().insert(tasklist=resolved, body=body).execute()
        return _normalize_task(created)

    def update_task(
        self,
        *,
        task_id: str,
        title: str | None = None,
        notes: str | None = None,
        due: str | None = None,
        status: str | None = None,
        completed: str | None = None,
        tasklist_id: str | None = None,
        tasklist_title: str | None = None,
    ) -> TaskItem:
        if not task_id:
            raise ValueError("Missing task id.")
        resolved = self.resolve_tasklist_id(tasklist_id, tasklist_title)
        if not resolved:
            raise ValueError("Task list not found.")
        body: dict = {}
        if title is not None:
            body["title"] = title
        if notes is not None:
            body["notes"] = notes
        if due is not None:
            body["due"] = due
        if status is not None:
            body["status"] = status
        if completed is not None:
            body["completed"] = completed
        service = self._get_service()
        if not body:
            existing = service.tasks().get(tasklist=resolved, task=task_id).execute()
            return _normalize_task(existing)
        updated = service.tasks().patch(tasklist=resolved, task=task_id, body=body).execute()
        return _normalize_task(updated)

    def complete_task(
        self,
        *,
        task_id: str,
        tasklist_id: str | None = None,
        tasklist_title: str | None = None,
    ) -> TaskItem:
        completed_at = datetime.now(timezone.utc).isoformat()
        return self.update_task(
            task_id=task_id,
            status="completed",
            completed=completed_at,
            tasklist_id=tasklist_id,
            tasklist_title=tasklist_title,
        )

    def delete_task(
        self,
        *,
        task_id: str,
        tasklist_id: str | None = None,
        tasklist_title: str | None = None,
    ) -> None:
        if not task_id:
            return
        resolved = self.resolve_tasklist_id(tasklist_id, tasklist_title)
        if not resolved:
            raise ValueError("Task list not found.")
        service = self._get_service()
        service.tasks().delete(tasklist=resolved, task=task_id).execute()

    def resolve_tasklist_id(self, tasklist_id: str | None, tasklist_title: str | None) -> str | None:
        if tasklist_id:
            return tasklist_id
        if tasklist_title:
            self._ensure_tasklist_cache()
            normalized = tasklist_title.strip().lower()
            if normalized in self._tasklist_cache:
                return self._tasklist_cache[normalized]
        if self._default_tasklist_id:
            return self._default_tasklist_id
        tasklists = self.list_tasklists()
        if not tasklists:
            return None
        preferred = next((item for item in tasklists if item.title.lower() == "my tasks"), None)
        selected = preferred or tasklists[0]
        self._default_tasklist_id = selected.tasklist_id
        return self._default_tasklist_id

    def _ensure_tasklist_cache(self) -> None:
        if self._tasklist_cache:
            return
        self.list_tasklists()

    def _cache_tasklists(self, tasklists: list[TaskList]) -> None:
        for item in tasklists:
            key = item.title.strip().lower()
            if key and key not in self._tasklist_cache:
                self._tasklist_cache[key] = item.tasklist_id

    def _get_service(self):
        if self._service is None:
            creds = self._auth.get_credentials()
            self._service = build("tasks", "v1", credentials=creds, cache_discovery=False)
        return self._service


def build_tasks_service(settings) -> TasksService:
    config = GmailAuthConfig(
        credentials_path=getattr(settings, "gmail_credentials_path", ""),
        token_path=getattr(settings, "gmail_token_path", None),
        scopes=getattr(settings, "gmail_scopes", None),
        oauth_flow=getattr(settings, "gmail_oauth_flow", "local"),
        open_browser=bool(getattr(settings, "gmail_oauth_open_browser", True)),
    )
    auth_service = GmailAuthService(config)
    return TasksService(auth_service)


def _normalize_tasklist(item: dict) -> TaskList:
    return TaskList(
        tasklist_id=str(item.get("id", "")),
        title=str(item.get("title", "")),
    )


def _normalize_task(item: dict) -> TaskItem:
    return TaskItem(
        task_id=str(item.get("id", "")),
        title=str(item.get("title", "")),
        notes=item.get("notes"),
        status=item.get("status"),
        due=item.get("due"),
    )
