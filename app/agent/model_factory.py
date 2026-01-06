from __future__ import annotations

from typing import Optional

from app.agent.llm_client import (
    GeminiImageClient,
    LLMClient,
    build_image_client,
    build_llm_client,
)


class ModelFactory:
    def __init__(
        self,
        api_key: str,
        text_model: str,
        image_model: str,
        *,
        enable_google_search: bool = True,
        enable_code_execution: bool = True,
    ) -> None:
        self._api_key = api_key
        self._text_model = text_model
        self._image_model = image_model
        self._enable_google_search = enable_google_search
        self._enable_code_execution = enable_code_execution
        self._text_client: Optional[LLMClient] = None
        self._image_client: Optional[GeminiImageClient] = None

    def get_text_client(self) -> Optional[LLMClient]:
        if self._text_client is None:
            self._text_client = build_llm_client(
                self._api_key,
                self._text_model,
                enable_google_search=self._enable_google_search,
                enable_code_execution=self._enable_code_execution,
            )
        return self._text_client

    def get_image_client(self) -> Optional[GeminiImageClient]:
        if self._image_client is None:
            self._image_client = build_image_client(
                self._api_key,
                self._image_model,
                enable_google_search=self._enable_google_search,
            )
        return self._image_client
