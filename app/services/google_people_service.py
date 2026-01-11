from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

from googleapiclient.discovery import build

from app.services.gmail_auth_service import GmailAuthConfig, GmailAuthService

logger = logging.getLogger(__name__)

_DEFAULT_FIELDS = (
    "names",
    "emailAddresses",
    "phoneNumbers",
    "birthdays",
    "organizations",
    "urls",
    "addresses",
)


@dataclass(frozen=True)
class ContactRecord:
    resource_name: str
    display_name: str
    emails: list[str]
    phones: list[str]
    birthdays: list[str]
    organizations: list[str]
    urls: list[str]
    addresses: list[str]


class PeopleService:
    def __init__(self, auth_service: GmailAuthService) -> None:
        self._auth = auth_service
        self._service = None

    def ensure_credentials(self) -> None:
        self._auth.get_credentials()

    def reset(self) -> None:
        self._service = None

    def search_contacts(
        self,
        *,
        query: str,
        max_results: int = 10,
        fields: Iterable[str] | None = None,
    ) -> list[ContactRecord]:
        if not query:
            return []
        service = self._get_service()
        read_mask = _normalize_fields(fields)
        response = (
            service.people()
            .searchContacts(
                query=query,
                readMask=",".join(read_mask),
                pageSize=max(1, min(25, max_results)),
            )
            .execute()
        )
        results = response.get("results", []) or []
        contacts = []
        for item in results:
            person = item.get("person")
            if isinstance(person, dict):
                contacts.append(_normalize_contact(person))
        return contacts

    def get_contact(
        self,
        *,
        resource_name: str,
        fields: Iterable[str] | None = None,
    ) -> ContactRecord:
        if not resource_name:
            raise ValueError("Missing contact resource name.")
        service = self._get_service()
        read_mask = _normalize_fields(fields)
        person = (
            service.people()
            .get(
                resourceName=resource_name,
                personFields=",".join(read_mask),
            )
            .execute()
        )
        return _normalize_contact(person or {})

    def list_connections(
        self,
        *,
        max_results: int = 10,
        page_token: str | None = None,
        fields: Iterable[str] | None = None,
    ) -> tuple[list[ContactRecord], str | None]:
        service = self._get_service()
        read_mask = _normalize_fields(fields)
        response = (
            service.people()
            .connections()
            .list(
                resourceName="people/me",
                pageSize=max(1, min(50, max_results)),
                pageToken=page_token or None,
                personFields=",".join(read_mask),
            )
            .execute()
        )
        connections = response.get("connections", []) or []
        contacts = [_normalize_contact(item) for item in connections if isinstance(item, dict)]
        next_token = response.get("nextPageToken")
        return contacts, next_token

    def _get_service(self):
        if self._service is None:
            creds = self._auth.get_credentials()
            self._service = build("people", "v1", credentials=creds, cache_discovery=False)
        return self._service


def build_people_service(settings) -> PeopleService:
    config = GmailAuthConfig(
        credentials_path=getattr(settings, "gmail_credentials_path", ""),
        token_path=getattr(settings, "gmail_token_path", None),
        scopes=getattr(settings, "gmail_scopes", None),
        oauth_flow=getattr(settings, "gmail_oauth_flow", "local"),
        open_browser=bool(getattr(settings, "gmail_oauth_open_browser", True)),
    )
    auth_service = GmailAuthService(config)
    return PeopleService(auth_service)


def _normalize_fields(fields: Iterable[str] | None) -> list[str]:
    if not fields:
        return list(_DEFAULT_FIELDS)
    cleaned = []
    for field in fields:
        item = str(field or "").strip()
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned or list(_DEFAULT_FIELDS)


def _normalize_contact(person: dict) -> ContactRecord:
    names = _pluck_values(person.get("names"), "displayName")
    emails = _pluck_values(person.get("emailAddresses"), "value")
    phones = _pluck_values(person.get("phoneNumbers"), "value")
    birthdays = _format_birthdays(person.get("birthdays"))
    organizations = _pluck_values(person.get("organizations"), "name")
    urls = _pluck_values(person.get("urls"), "value")
    addresses = _format_addresses(person.get("addresses"))
    display_name = names[0] if names else ""
    return ContactRecord(
        resource_name=str(person.get("resourceName") or ""),
        display_name=display_name,
        emails=emails,
        phones=phones,
        birthdays=birthdays,
        organizations=organizations,
        urls=urls,
        addresses=addresses,
    )


def _pluck_values(items, key: str) -> list[str]:
    values = []
    if not isinstance(items, list):
        return values
    for item in items:
        if not isinstance(item, dict):
            continue
        value = str(item.get(key) or "").strip()
        if value:
            values.append(value)
    return values


def _format_birthdays(items) -> list[str]:
    values = []
    if not isinstance(items, list):
        return values
    for item in items:
        date_info = item.get("date") if isinstance(item, dict) else None
        if not isinstance(date_info, dict):
            continue
        year = date_info.get("year")
        month = date_info.get("month")
        day = date_info.get("day")
        if month and day:
            if year:
                values.append(f"{year:04d}-{month:02d}-{day:02d}")
            else:
                values.append(f"{month:02d}-{day:02d}")
    return values


def _format_addresses(items) -> list[str]:
    values = []
    if not isinstance(items, list):
        return values
    for item in items:
        if not isinstance(item, dict):
            continue
        formatted = str(item.get("formattedValue") or "").strip()
        if formatted:
            values.append(formatted)
            continue
        parts = []
        for key in ("streetAddress", "city", "region", "postalCode", "country"):
            value = str(item.get(key) or "").strip()
            if value:
                parts.append(value)
        if parts:
            values.append(", ".join(parts))
    return values
