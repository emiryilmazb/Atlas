from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import logging

from googleapiclient.discovery import build

from app.services.gmail_auth_service import GmailAuthConfig, GmailAuthService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalendarEvent:
    event_id: str
    summary: str
    start: str | None
    end: str | None
    html_link: str | None
    status: str | None
    location: str | None


class CalendarService:
    def __init__(self, auth_service: GmailAuthService, calendar_id: str = "primary") -> None:
        self._auth = auth_service
        self._calendar_id = calendar_id or "primary"
        self._service = None

    @property
    def calendar_id(self) -> str:
        return self._calendar_id

    def ensure_credentials(self) -> None:
        self._auth.get_credentials()

    def reset(self) -> None:
        self._service = None

    def list_events(
        self,
        *,
        time_min: str | None = None,
        time_max: str | None = None,
        query: str | None = None,
        max_results: int = 10,
        calendar_id: str | None = None,
        include_cancelled: bool = False,
    ) -> list[CalendarEvent]:
        service = self._get_service()
        params = {
            "calendarId": calendar_id or self._calendar_id,
            "singleEvents": True,
            "orderBy": "startTime",
            "maxResults": max(1, min(25, max_results)),
            "showDeleted": bool(include_cancelled),
        }
        if time_min:
            params["timeMin"] = time_min
        if time_max:
            params["timeMax"] = time_max
        if query:
            params["q"] = query
        response = service.events().list(**params).execute()
        items = response.get("items", []) or []
        return [_normalize_event(item) for item in items if isinstance(item, dict)]

    def create_event(
        self,
        *,
        summary: str,
        start_time: str | None = None,
        end_time: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        description: str | None = None,
        location: str | None = None,
        attendees: list[str] | None = None,
        time_zone: str | None = None,
        calendar_id: str | None = None,
    ) -> CalendarEvent:
        body = _build_event_body(
            summary=summary,
            start_time=start_time,
            end_time=end_time,
            start_date=start_date,
            end_date=end_date,
            description=description,
            location=location,
            attendees=attendees,
            time_zone=time_zone,
        )
        service = self._get_service()
        created = service.events().insert(
            calendarId=calendar_id or self._calendar_id,
            body=body,
        ).execute()
        return _normalize_event(created)

    def update_event(
        self,
        *,
        event_id: str,
        summary: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        description: str | None = None,
        location: str | None = None,
        attendees: list[str] | None = None,
        time_zone: str | None = None,
        calendar_id: str | None = None,
    ) -> CalendarEvent:
        if not event_id:
            raise ValueError("Missing calendar event id.")
        body = _build_event_patch(
            summary=summary,
            start_time=start_time,
            end_time=end_time,
            start_date=start_date,
            end_date=end_date,
            description=description,
            location=location,
            attendees=attendees,
            time_zone=time_zone,
        )
        service = self._get_service()
        calendar_id_value = calendar_id or self._calendar_id
        if not body:
            existing = service.events().get(
                calendarId=calendar_id_value,
                eventId=event_id,
            ).execute()
            return _normalize_event(existing)
        updated = service.events().patch(
            calendarId=calendar_id_value,
            eventId=event_id,
            body=body,
        ).execute()
        return _normalize_event(updated)

    def delete_event(self, *, event_id: str, calendar_id: str | None = None) -> None:
        if not event_id:
            return
        service = self._get_service()
        service.events().delete(
            calendarId=calendar_id or self._calendar_id,
            eventId=event_id,
        ).execute()

    def _get_service(self):
        if self._service is None:
            creds = self._auth.get_credentials()
            self._service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        return self._service


def build_calendar_service(settings) -> CalendarService:
    config = GmailAuthConfig(
        credentials_path=getattr(settings, "gmail_credentials_path", ""),
        token_path=getattr(settings, "gmail_token_path", None),
        scopes=getattr(settings, "gmail_scopes", None),
        oauth_flow=getattr(settings, "gmail_oauth_flow", "local"),
        open_browser=bool(getattr(settings, "gmail_oauth_open_browser", True)),
    )
    auth_service = GmailAuthService(config)
    calendar_id = getattr(settings, "calendar_id", "") or "primary"
    return CalendarService(auth_service, calendar_id=calendar_id)


def _normalize_event(event: dict) -> CalendarEvent:
    start = event.get("start", {}) if isinstance(event, dict) else {}
    end = event.get("end", {}) if isinstance(event, dict) else {}
    return CalendarEvent(
        event_id=str(event.get("id", "")),
        summary=str(event.get("summary", "")),
        start=start.get("dateTime") or start.get("date"),
        end=end.get("dateTime") or end.get("date"),
        html_link=event.get("htmlLink"),
        status=event.get("status"),
        location=event.get("location"),
    )


def _build_event_body(
    *,
    summary: str,
    start_time: str | None,
    end_time: str | None,
    start_date: str | None,
    end_date: str | None,
    description: str | None,
    location: str | None,
    attendees: list[str] | None,
    time_zone: str | None,
) -> dict:
    body: dict = {"summary": summary}
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    attendee_items = _normalize_attendees(attendees)
    if attendee_items:
        body["attendees"] = attendee_items
    start_block, end_block = _build_event_time_blocks(
        start_time=start_time,
        end_time=end_time,
        start_date=start_date,
        end_date=end_date,
        time_zone=time_zone,
    )
    body["start"] = start_block
    body["end"] = end_block
    return body


def _build_event_patch(
    *,
    summary: str | None,
    start_time: str | None,
    end_time: str | None,
    start_date: str | None,
    end_date: str | None,
    description: str | None,
    location: str | None,
    attendees: list[str] | None,
    time_zone: str | None,
) -> dict:
    body: dict = {}
    if summary:
        body["summary"] = summary
    if description is not None:
        body["description"] = description
    if location is not None:
        body["location"] = location
    attendee_items = _normalize_attendees(attendees)
    if attendee_items is not None:
        body["attendees"] = attendee_items
    if start_time or end_time or start_date or end_date:
        start_block, end_block = _build_event_time_blocks(
            start_time=start_time,
            end_time=end_time,
            start_date=start_date,
            end_date=end_date,
            time_zone=time_zone,
        )
        body["start"] = start_block
        body["end"] = end_block
    return body


def _build_event_time_blocks(
    *,
    start_time: str | None,
    end_time: str | None,
    start_date: str | None,
    end_date: str | None,
    time_zone: str | None,
) -> tuple[dict, dict]:
    if start_date:
        resolved_end = end_date or _next_date(start_date)
        return (
            {"date": start_date},
            {"date": resolved_end},
        )
    if start_time or end_time:
        if not start_time or not end_time:
            raise ValueError("Start and end times are required for timed events.")
        start_block: dict = {"dateTime": start_time}
        end_block: dict = {"dateTime": end_time}
        if time_zone:
            start_block["timeZone"] = time_zone
            end_block["timeZone"] = time_zone
        return start_block, end_block
    raise ValueError("Missing start_time or start_date for the event.")


def _normalize_attendees(attendees: list[str] | None) -> list[dict] | None:
    if attendees is None:
        return None
    items = []
    for email in attendees:
        cleaned = str(email or "").strip()
        if cleaned:
            items.append({"email": cleaned})
    return items


def _next_date(value: str) -> str:
    try:
        return (date.fromisoformat(value) + timedelta(days=1)).isoformat()
    except ValueError:
        logger.warning("Invalid date value: %s", value)
        return value
