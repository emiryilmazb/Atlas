from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.services.gmail_auth_service import GmailAuthConfig, GmailAuthService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YouTubeVideo:
    video_id: str
    title: str
    description: str
    published_at: str
    channel_title: str
    url: str


@dataclass(frozen=True)
class YouTubePlaylist:
    playlist_id: str
    title: str
    description: str


@dataclass(frozen=True)
class YouTubePlaylistItem:
    playlist_item_id: str
    video_id: str
    title: str
    description: str
    published_at: str


class YouTubeService:
    def __init__(self, auth_service: GmailAuthService) -> None:
        self._auth = auth_service
        self._service = None
        self._watch_later_id: str | None = None

    def ensure_credentials(self) -> None:
        self._auth.get_credentials()

    def reset(self) -> None:
        self._service = None
        self._watch_later_id = None

    def search_videos(
        self,
        *,
        query: str,
        max_results: int = 5,
    ) -> list[YouTubeVideo]:
        if not query:
            raise ValueError("Missing YouTube search query.")
        service = self._get_service()
        try:
            response = (
                service.search()
                .list(
                    q=query,
                    part="snippet",
                    type="video",
                    maxResults=max(1, min(5, max_results)),
                    order="relevance",
                    safeSearch="moderate",
                )
                .execute()
            )
        except HttpError as exc:
            raise _youtube_http_error(exc) from exc
        items = response.get("items", []) or []
        return [_normalize_search_item(item) for item in items if isinstance(item, dict)]

    def list_video_details(self, *, video_ids: list[str]) -> list[YouTubeVideo]:
        cleaned = [vid for vid in (video_ids or []) if vid]
        if not cleaned:
            return []
        service = self._get_service()
        try:
            response = (
                service.videos()
                .list(
                    part="snippet",
                    id=",".join(cleaned[:50]),
                    maxResults=min(50, len(cleaned)),
                )
                .execute()
            )
        except HttpError as exc:
            raise _youtube_http_error(exc) from exc
        items = response.get("items", []) or []
        return [_normalize_video_item(item) for item in items if isinstance(item, dict)]

    def list_playlists(self, *, max_results: int = 25) -> list[YouTubePlaylist]:
        service = self._get_service()
        try:
            response = (
                service.playlists()
                .list(
                    part="snippet",
                    mine=True,
                    maxResults=max(1, min(50, max_results)),
                )
                .execute()
            )
        except HttpError as exc:
            raise _youtube_http_error(exc) from exc
        items = response.get("items", []) or []
        return [_normalize_playlist_item(item) for item in items if isinstance(item, dict)]

    def list_playlist_items(
        self,
        *,
        playlist_id: str,
        max_results: int = 10,
    ) -> list[YouTubePlaylistItem]:
        if not playlist_id:
            raise ValueError("Missing playlist id.")
        service = self._get_service()
        try:
            response = (
                service.playlistItems()
                .list(
                    part="snippet,contentDetails",
                    playlistId=playlist_id,
                    maxResults=max(1, min(50, max_results)),
                )
                .execute()
            )
        except HttpError as exc:
            raise _youtube_http_error(exc) from exc
        items = response.get("items", []) or []
        return [_normalize_playlist_item_row(item) for item in items if isinstance(item, dict)]

    def add_to_playlist(self, *, playlist_id: str, video_id: str) -> YouTubePlaylistItem:
        if not playlist_id:
            raise ValueError("Missing playlist id.")
        if not video_id:
            raise ValueError("Missing video id.")
        body = {
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {
                    "kind": "youtube#video",
                    "videoId": video_id,
                },
            }
        }
        service = self._get_service()
        try:
            response = service.playlistItems().insert(part="snippet", body=body).execute()
        except HttpError as exc:
            raise _youtube_http_error(exc) from exc
        return _normalize_playlist_item_row(response)

    def add_to_watch_later(self, *, video_id: str) -> YouTubePlaylistItem:
        playlist_id = self._get_watch_later_id()
        return self.add_to_playlist(playlist_id=playlist_id, video_id=video_id)

    def remove_from_playlist(self, *, playlist_item_id: str) -> None:
        if not playlist_item_id:
            raise ValueError("Missing playlist item id.")
        service = self._get_service()
        try:
            service.playlistItems().delete(id=playlist_item_id).execute()
        except HttpError as exc:
            raise _youtube_http_error(exc) from exc

    def fetch_transcript(self, *, video_id: str) -> dict[str, Any]:
        if not video_id:
            raise ValueError("Missing video id.")
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ModuleNotFoundError:
            return {
                "video_id": video_id,
                "transcript": "",
                "available": False,
                "error": "Transcript library is not installed.",
            }
        try:
            items = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as exc:
            return {
                "video_id": video_id,
                "transcript": "",
                "available": False,
                "error": str(exc),
            }
        text = " ".join(item.get("text", "") for item in items if isinstance(item, dict)).strip()
        return {
            "video_id": video_id,
            "transcript": text,
            "available": bool(text),
            "error": None,
        }

    def _get_watch_later_id(self) -> str:
        if self._watch_later_id:
            return self._watch_later_id
        service = self._get_service()
        try:
            response = service.channels().list(part="contentDetails", mine=True).execute()
        except HttpError as exc:
            raise _youtube_http_error(exc) from exc
        items = response.get("items", []) or []
        if not items:
            raise ValueError("Watch Later playlist could not be resolved.")
        content_details = items[0].get("contentDetails", {}) if isinstance(items[0], dict) else {}
        related = content_details.get("relatedPlaylists", {}) if isinstance(content_details, dict) else {}
        watch_later_id = str(related.get("watchLater") or "").strip()
        if not watch_later_id:
            raise ValueError("Watch Later playlist could not be resolved.")
        self._watch_later_id = watch_later_id
        return watch_later_id

    def _get_service(self):
        if self._service is None:
            creds = self._auth.get_credentials()
            self._service = build("youtube", "v3", credentials=creds, cache_discovery=False)
        return self._service


def build_youtube_service(settings) -> YouTubeService:
    config = GmailAuthConfig(
        credentials_path=getattr(settings, "gmail_credentials_path", ""),
        token_path=getattr(settings, "gmail_token_path", None),
        scopes=getattr(settings, "gmail_scopes", None),
        oauth_flow=getattr(settings, "gmail_oauth_flow", "local"),
        open_browser=bool(getattr(settings, "gmail_oauth_open_browser", True)),
    )
    auth_service = GmailAuthService(config)
    return YouTubeService(auth_service)


def _normalize_search_item(item: dict) -> YouTubeVideo:
    snippet = item.get("snippet", {}) if isinstance(item, dict) else {}
    video_id = ""
    id_block = item.get("id")
    if isinstance(id_block, dict):
        video_id = str(id_block.get("videoId") or "")
    title = str(snippet.get("title") or "")
    description = str(snippet.get("description") or "")
    published_at = str(snippet.get("publishedAt") or "")
    channel_title = str(snippet.get("channelTitle") or "")
    url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""
    return YouTubeVideo(
        video_id=video_id,
        title=title,
        description=description,
        published_at=published_at,
        channel_title=channel_title,
        url=url,
    )


def _normalize_video_item(item: dict) -> YouTubeVideo:
    snippet = item.get("snippet", {}) if isinstance(item, dict) else {}
    video_id = str(item.get("id") or "")
    title = str(snippet.get("title") or "")
    description = str(snippet.get("description") or "")
    published_at = str(snippet.get("publishedAt") or "")
    channel_title = str(snippet.get("channelTitle") or "")
    url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""
    return YouTubeVideo(
        video_id=video_id,
        title=title,
        description=description,
        published_at=published_at,
        channel_title=channel_title,
        url=url,
    )


def _normalize_playlist_item(item: dict) -> YouTubePlaylist:
    snippet = item.get("snippet", {}) if isinstance(item, dict) else {}
    playlist_id = str(item.get("id") or "")
    title = str(snippet.get("title") or "")
    description = str(snippet.get("description") or "")
    return YouTubePlaylist(
        playlist_id=playlist_id,
        title=title,
        description=description,
    )


def _normalize_playlist_item_row(item: dict) -> YouTubePlaylistItem:
    snippet = item.get("snippet", {}) if isinstance(item, dict) else {}
    content = item.get("contentDetails", {}) if isinstance(item, dict) else {}
    playlist_item_id = str(item.get("id") or "")
    video_id = str(content.get("videoId") or snippet.get("resourceId", {}).get("videoId") or "")
    title = str(snippet.get("title") or "")
    description = str(snippet.get("description") or "")
    published_at = str(snippet.get("publishedAt") or "")
    return YouTubePlaylistItem(
        playlist_item_id=playlist_item_id,
        video_id=video_id,
        title=title,
        description=description,
        published_at=published_at,
    )


def _youtube_http_error(exc: HttpError) -> Exception:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "resp", None), "status", None)
    message = str(exc).lower()
    if status == 404:
        return ValueError("YouTube resource not found.")
    if status == 403:
        if "quota" in message or "rate limit" in message:
            return ValueError("YouTube API quota exceeded. Please retry later.")
        return ValueError("Permission denied for this YouTube resource.")
    if status == 400:
        return ValueError("Invalid YouTube request. Please verify the parameters.")
    if status == 429:
        return ValueError("YouTube API quota exceeded. Please retry later.")
    return ValueError(f"YouTube request failed (status {status}).")
