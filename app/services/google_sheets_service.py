from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.services.gmail_auth_service import GmailAuthConfig, GmailAuthService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SheetInfo:
    spreadsheet_id: str
    title: str
    url: str


class SheetsService:
    def __init__(self, auth_service: GmailAuthService) -> None:
        self._auth = auth_service
        self._service = None
        self._sheet_title_cache: dict[str, list[str]] = {}

    def ensure_credentials(self) -> None:
        self._auth.get_credentials()

    def reset(self) -> None:
        self._service = None
        self._sheet_title_cache = {}

    def create_spreadsheet(
        self,
        *,
        title: str,
        sheet_title: str | None = None,
    ) -> SheetInfo:
        if not title:
            raise ValueError("Missing spreadsheet title.")
        body: dict[str, Any] = {"properties": {"title": title}}
        if sheet_title:
            body["sheets"] = [{"properties": {"title": sheet_title}}]
        try:
            response = (
                self._get_service()
                .spreadsheets()
                .create(body=body)
                .execute()
            )
        except HttpError as exc:
            raise _sheets_http_error(exc) from exc
        spreadsheet_id = str(response.get("spreadsheetId") or "")
        sheet_title_value = str(response.get("properties", {}).get("title") or title)
        url = str(response.get("spreadsheetUrl") or "")
        if spreadsheet_id:
            self._sheet_title_cache.pop(spreadsheet_id, None)
        return SheetInfo(
            spreadsheet_id=spreadsheet_id,
            title=sheet_title_value,
            url=url,
        )

    def append_row(
        self,
        *,
        spreadsheet_id: str,
        sheet_name: str | None = None,
        row_values: list[Any] | None = None,
        row_data: dict[str, Any] | None = None,
        headers: list[str] | None = None,
        value_input_option: str = "USER_ENTERED",
    ) -> dict[str, Any]:
        if not spreadsheet_id:
            raise ValueError("Missing spreadsheet id.")
        if row_values is None and row_data is None:
            raise ValueError("Provide row_values or row_data.")
        sheet_title = self._resolve_sheet_title(spreadsheet_id, sheet_name)
        if row_data is not None:
            resolved_headers = self._ensure_headers(
                spreadsheet_id,
                sheet_title,
                row_data=row_data,
                headers=headers,
            )
            values = [_map_row_to_headers(row_data, resolved_headers)]
        else:
            if headers:
                self._ensure_headers(
                    spreadsheet_id,
                    sheet_title,
                    row_data=None,
                    headers=headers,
                )
            values = [list(row_values or [])]
        try:
            response = (
                self._get_service()
                .spreadsheets()
                .values()
                .append(
                    spreadsheetId=spreadsheet_id,
                    range=sheet_title,
                    valueInputOption=value_input_option,
                    insertDataOption="INSERT_ROWS",
                    body={"values": values},
                )
                .execute()
            )
        except HttpError as exc:
            raise _sheets_http_error(exc) from exc
        updates = response.get("updates", {}) if isinstance(response, dict) else {}
        return {
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": sheet_title,
            "updated_range": updates.get("updatedRange"),
            "updated_rows": updates.get("updatedRows"),
            "updated_columns": updates.get("updatedColumns"),
        }

    def read_range(
        self,
        *,
        spreadsheet_id: str,
        range_a1: str,
    ) -> dict[str, Any]:
        if not spreadsheet_id:
            raise ValueError("Missing spreadsheet id.")
        if not range_a1:
            raise ValueError("Missing range.")
        try:
            response = (
                self._get_service()
                .spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=range_a1)
                .execute()
            )
        except HttpError as exc:
            raise _sheets_http_error(exc) from exc
        return {
            "spreadsheet_id": spreadsheet_id,
            "range": response.get("range"),
            "values": response.get("values", []),
        }

    def update_range(
        self,
        *,
        spreadsheet_id: str,
        range_a1: str,
        values: list[list[Any]],
        value_input_option: str = "USER_ENTERED",
    ) -> dict[str, Any]:
        if not spreadsheet_id:
            raise ValueError("Missing spreadsheet id.")
        if not range_a1:
            raise ValueError("Missing range.")
        if values is None:
            raise ValueError("Missing values to update.")
        try:
            response = (
                self._get_service()
                .spreadsheets()
                .values()
                .update(
                    spreadsheetId=spreadsheet_id,
                    range=range_a1,
                    valueInputOption=value_input_option,
                    body={"values": values},
                )
                .execute()
            )
        except HttpError as exc:
            raise _sheets_http_error(exc) from exc
        return {
            "spreadsheet_id": spreadsheet_id,
            "updated_range": response.get("updatedRange"),
            "updated_rows": response.get("updatedRows"),
            "updated_columns": response.get("updatedColumns"),
            "updated_cells": response.get("updatedCells"),
        }

    def _get_service(self):
        if self._service is None:
            creds = self._auth.get_credentials()
            self._service = build("sheets", "v4", credentials=creds, cache_discovery=False)
        return self._service

    def _resolve_sheet_title(self, spreadsheet_id: str, sheet_name: str | None) -> str:
        titles = self._get_sheet_titles(spreadsheet_id)
        if sheet_name:
            normalized = sheet_name.strip().lower()
            for title in titles:
                if title.lower() == normalized:
                    return title
            raise ValueError(
                f"Spreadsheet tab not found: {sheet_name}. Available: {', '.join(titles)}"
            )
        if not titles:
            raise ValueError("Spreadsheet has no sheets.")
        return titles[0]

    def _get_sheet_titles(self, spreadsheet_id: str) -> list[str]:
        cached = self._sheet_title_cache.get(spreadsheet_id)
        if cached is not None:
            return cached
        try:
            response = (
                self._get_service()
                .spreadsheets()
                .get(spreadsheetId=spreadsheet_id, fields="sheets.properties.title")
                .execute()
            )
        except HttpError as exc:
            raise _sheets_http_error(exc) from exc
        titles: list[str] = []
        for sheet in response.get("sheets", []) or []:
            props = sheet.get("properties", {}) if isinstance(sheet, dict) else {}
            title = str(props.get("title") or "").strip()
            if title:
                titles.append(title)
        self._sheet_title_cache[spreadsheet_id] = titles
        return titles

    def _ensure_headers(
        self,
        spreadsheet_id: str,
        sheet_title: str,
        *,
        row_data: dict[str, Any] | None,
        headers: list[str] | None,
    ) -> list[str]:
        existing = self._read_header_row(spreadsheet_id, sheet_title)
        desired = list(existing)
        if not desired:
            if headers:
                desired = [str(item).strip() for item in headers if str(item).strip()]
            elif row_data:
                desired = [str(key).strip() for key in row_data.keys() if str(key).strip()]
        if row_data:
            normalized_map = {str(key).strip().lower(): str(key).strip() for key in row_data.keys()}
            for key_lower, raw_key in normalized_map.items():
                if not key_lower:
                    continue
                if not any(key_lower == title.lower() for title in desired):
                    desired.append(raw_key)
        if desired and desired != existing:
            range_a1 = f"{sheet_title}!1:1"
            try:
                (
                    self._get_service()
                    .spreadsheets()
                    .values()
                    .update(
                        spreadsheetId=spreadsheet_id,
                        range=range_a1,
                        valueInputOption="RAW",
                        body={"values": [desired]},
                    )
                    .execute()
                )
            except HttpError as exc:
                raise _sheets_http_error(exc) from exc
        return desired

    def _read_header_row(self, spreadsheet_id: str, sheet_title: str) -> list[str]:
        range_a1 = f"{sheet_title}!1:1"
        try:
            response = (
                self._get_service()
                .spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=range_a1)
                .execute()
            )
        except HttpError as exc:
            raise _sheets_http_error(exc) from exc
        values = response.get("values", []) if isinstance(response, dict) else []
        if not values:
            return []
        header_row = values[0] if isinstance(values[0], list) else []
        return [str(item).strip() for item in header_row if str(item).strip()]


def build_sheets_service(settings) -> SheetsService:
    config = GmailAuthConfig(
        credentials_path=getattr(settings, "gmail_credentials_path", ""),
        token_path=getattr(settings, "gmail_token_path", None),
        scopes=getattr(settings, "gmail_scopes", None),
        oauth_flow=getattr(settings, "gmail_oauth_flow", "local"),
        open_browser=bool(getattr(settings, "gmail_oauth_open_browser", True)),
    )
    auth_service = GmailAuthService(config)
    return SheetsService(auth_service)


def _map_row_to_headers(row_data: dict[str, Any], headers: list[str]) -> list[Any]:
    normalized = {str(key).strip().lower(): value for key, value in row_data.items()}
    row_values: list[Any] = []
    for header in headers:
        key = str(header).strip().lower()
        row_values.append(normalized.get(key, ""))
    return row_values


def _sheets_http_error(exc: HttpError) -> Exception:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "resp", None), "status", None)
    message = str(exc).lower()
    if status == 404:
        return ValueError("Spreadsheet not found.")
    if status == 403:
        if "quota" in message or "rate limit" in message:
            return ValueError("Sheets API quota exceeded. Please retry later or reduce requests.")
        return ValueError("Permission denied for this spreadsheet.")
    if status == 400:
        return ValueError("Invalid range or sheet name. Please verify the A1 range.")
    if status == 429:
        return ValueError("Sheets API quota exceeded. Please retry later.")
    return ValueError(f"Sheets request failed (status {status}).")
