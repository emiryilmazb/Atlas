from dataclasses import dataclass
import logging
import mimetypes
from pathlib import Path
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from typing import AsyncIterator, Callable, Optional

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    def generate_text(self, prompt: str) -> str:
        raise NotImplementedError

    def generate_with_file(self, prompt: str, file_path: str, mime_type: str | None = None) -> "FileGenerationResult":
        raise NotImplementedError

    def generate_suggestions(self, history_text: str) -> list[str]:
        raise NotImplementedError

    async def stream_text(self, prompt: str, include_thoughts: bool = False) -> "AsyncIterator[StreamChunk]":
        raise NotImplementedError

    async def stream_with_image(
        self,
        prompt: str,
        image_path: str,
        mime_type: str | None = None,
        include_thoughts: bool = False,
    ) -> "AsyncIterator[StreamChunk]":
        raise NotImplementedError


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str
    enable_google_search: bool = False
    enable_code_execution: bool = False


@dataclass(frozen=True)
class FileGenerationResult:
    text: str
    error: str | None = None


@dataclass(frozen=True)
class StreamChunk:
    text: str
    is_thought: bool = False


@dataclass(frozen=True)
class ImageStreamState:
    chunks: AsyncIterator[StreamChunk]
    get_image_bytes: Callable[[], bytes | None]


@dataclass(frozen=True)
class SourceAttribution:
    title: str
    uri: str


@dataclass(frozen=True)
class TextStreamState:
    chunks: AsyncIterator[StreamChunk]
    get_sources: Callable[[], list[SourceAttribution]]


def _build_tool_list(*, enable_google_search: bool, enable_code_execution: bool):
    if not enable_google_search and not enable_code_execution:
        return []
    from google.genai import types

    tools: list[types.Tool] = []
    if enable_google_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    if enable_code_execution:
        tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
    return tools


def _build_generate_content_config(
    *,
    enable_google_search: bool,
    enable_code_execution: bool,
    include_thoughts: bool = False,
    response_modalities=None,
):
    from google.genai import types

    tools = _build_tool_list(
        enable_google_search=enable_google_search,
        enable_code_execution=enable_code_execution,
    )
    if not tools and not include_thoughts and response_modalities is None:
        return None
    config = types.GenerateContentConfig()
    if tools:
        config.tools = tools
    if include_thoughts:
        config.thinking_config = types.ThinkingConfig(include_thoughts=True)
    if response_modalities is not None:
        config.response_modalities = list(response_modalities)
    return config


def _collect_sources_from_response(response) -> list[SourceAttribution]:
    sources: list[SourceAttribution] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        citation_meta = getattr(candidate, "citation_metadata", None)
        citations = getattr(citation_meta, "citations", None) or []
        for citation in citations:
            uri = getattr(citation, "uri", None)
            if not uri:
                continue
            title = getattr(citation, "title", None) or uri
            sources.append(SourceAttribution(title=str(title), uri=str(uri)))
        grounding = getattr(candidate, "grounding_metadata", None)
        chunks = getattr(grounding, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if not web:
                continue
            uri = getattr(web, "uri", None)
            if not uri:
                continue
            title = getattr(web, "title", None) or getattr(web, "domain", None) or uri
            sources.append(SourceAttribution(title=str(title), uri=str(uri)))
    return sources


def _merge_sources(
    target: dict[str, SourceAttribution],
    new_sources: list[SourceAttribution],
) -> None:
    for source in new_sources:
        uri = source.uri.strip()
        if not uri or uri in target:
            continue
        target[uri] = source


class GeminiClient(LLMClient):
    def __init__(self, config: GeminiConfig) -> None:
        try:
            from google import genai
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise LLMClientError("google-genai is not installed") from exc

        self._client = genai.Client(api_key=config.api_key)
        self._model = config.model
        self._enable_google_search = config.enable_google_search
        self._enable_code_execution = config.enable_code_execution

    def build_generate_config(self, include_thoughts: bool = False, response_modalities=None):
        return _build_generate_content_config(
            enable_google_search=self._enable_google_search,
            enable_code_execution=self._enable_code_execution,
            include_thoughts=include_thoughts,
            response_modalities=response_modalities,
        )

    def generate_text(self, prompt: str) -> str:
        config = self.build_generate_config()
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        if getattr(response, "text", None):
            return response.text
        return ""

    def generate_text_with_sources(self, prompt: str) -> tuple[str, list[SourceAttribution]]:
        if not prompt:
            return "", []
        config = self.build_generate_config()
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        text = getattr(response, "text", None) or ""
        sources: dict[str, SourceAttribution] = {}
        _merge_sources(sources, _collect_sources_from_response(response))
        return text, list(sources.values())

    async def stream_text(self, prompt: str, include_thoughts: bool = False) -> AsyncIterator[StreamChunk]:
        if not prompt:
            return
        config = self.build_generate_config(include_thoughts=include_thoughts)
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=prompt,
            config=config,
        )
        async for chunk in stream:
            parts = getattr(chunk, "parts", None) or []
            if parts:
                for part in parts:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        yield StreamChunk(text=text, is_thought=bool(getattr(part, "thought", False)))
                continue
            text = getattr(chunk, "text", None)
            if isinstance(text, str) and text:
                yield StreamChunk(text=text, is_thought=False)

    async def stream_text_with_sources(
        self,
        prompt: str,
        include_thoughts: bool = False,
    ) -> TextStreamState:
        if not prompt:
            return TextStreamState(chunks=_empty_stream(), get_sources=lambda: [])
        config = self.build_generate_config(include_thoughts=include_thoughts)
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=prompt,
            config=config,
        )
        return _build_text_stream_state(stream)

    async def stream_with_image(
        self,
        prompt: str,
        image_path: str,
        mime_type: str | None = None,
        include_thoughts: bool = False,
    ) -> AsyncIterator[StreamChunk]:
        if not prompt or not image_path:
            return
        from google.genai import types

        with open(image_path, "rb") as handle:
            image_data = handle.read()
        resolved_mime = mime_type or mimetypes.guess_type(image_path)[0] or "image/png"
        config = self.build_generate_config(include_thoughts=include_thoughts)
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type=resolved_mime),
            ],
            config=config,
        )
        async for chunk in stream:
            parts = getattr(chunk, "parts", None) or []
            if parts:
                for part in parts:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        yield StreamChunk(text=text, is_thought=bool(getattr(part, "thought", False)))
                continue
            text = getattr(chunk, "text", None)
            if isinstance(text, str) and text:
                yield StreamChunk(text=text, is_thought=False)

    def generate_with_file(self, prompt: str, file_path: str, mime_type: str | None = None) -> FileGenerationResult:
        if not prompt or not file_path:
            return FileGenerationResult("", "Missing prompt or file.")
        if getattr(self._client, "vertexai", False):
            logger.warning(
                "File upload is not supported for Vertex AI client.")
            return FileGenerationResult("", "File upload is not supported for this client.")
        from google.genai import types

        normalized_mime = mime_type or mimetypes.guess_type(file_path)[0]
        if _is_gemini_flash_model(self._model):
            size_error = _flash_document_size_error(file_path, normalized_mime)
            if size_error:
                return FileGenerationResult("", size_error)
            if not _is_supported_flash_file_type(file_path, normalized_mime):
                return FileGenerationResult(
                    "",
                    self._build_unsupported_file_message(prompt, file_path, normalized_mime),
                )
        if _should_inline_text(file_path, normalized_mime):
            text_payload = _extract_text_fallback(file_path, normalized_mime)
            if not text_payload:
                return FileGenerationResult("", _unreadable_file_message(file_path, normalized_mime))
            response_text = self.generate_text(
                _merge_prompt_and_text(prompt, text_payload))
            if response_text:
                return FileGenerationResult(response_text)
            return FileGenerationResult("", "No response generated for the document.")
        try:
            upload_config = types.UploadFileConfig(
                mime_type=normalized_mime) if normalized_mime else None
            file_obj = self._client.files.upload(
                file=file_path, config=upload_config)
            file_obj = _wait_for_file_active(self._client, file_obj)
            if not file_obj or not getattr(file_obj, "uri", None):
                return FileGenerationResult("", "File upload failed.")
            part = types.Part.from_uri(
                file_uri=file_obj.uri,
                mime_type=file_obj.mime_type or normalized_mime,
            )
            try:
                config = self.build_generate_config()
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[prompt, part],
                    config=config,
                )
            finally:
                _safe_delete_file(self._client, file_obj)
            if getattr(response, "text", None):
                return FileGenerationResult(response.text)
            return FileGenerationResult("", "No response generated for the document.")
        except Exception as exc:  # pragma: no cover - network/proxy errors
            if _is_unsupported_mime_error(exc):
                text_payload = _extract_text_fallback(
                    file_path, normalized_mime)
                if text_payload:
                    response_text = self.generate_text(
                        _merge_prompt_and_text(prompt, text_payload))
                    if response_text:
                        return FileGenerationResult(response_text)
                    return FileGenerationResult("", "No response generated for the document.")
                return FileGenerationResult("", _unreadable_file_message(file_path, normalized_mime))
            logger.warning("File upload failed: %s", exc)
            return FileGenerationResult("", "Sorry, I couldn't process this document right now.")

    def _build_unsupported_file_message(
        self,
        user_prompt: str,
        file_path: str,
        mime_type: str | None,
    ) -> str:
        label = mime_type or Path(file_path).suffix.lower() or "unknown"
        fallback = _unsupported_file_type_message(label)
        if not user_prompt:
            return fallback
        prompt = (
            "The user tried to upload a file that Gemini cannot process.\n"
            f"User request: {user_prompt}\n"
            f"File type: {label}\n"
            "Reply in the user's language with a short, friendly message that this file type is not supported. "
            "If it is a document, suggest PDF or plain text (or exporting to PDF). "
            "If it is audio/video, suggest a common supported format (e.g., mp3, wav, mp4, webm)."
        )
        try:
            response = self.generate_text(prompt)
        except Exception as exc:
            logger.warning("Unsupported file message generation failed: %s", exc)
            return fallback
        cleaned = response.strip()
        return cleaned or fallback

    def generate_suggestions(self, history_text: str) -> list[str]:
        prompt = (
            "You are generating Telegram inline suggestion buttons for a multimodal assistant.\n"
            "Use the recent conversation history and session context to propose the next best steps.\n"
            "Suggestions should be specific, interesting, and directly actionable by the assistant\n"
            "(chat, computer-use, image edit, or image generation).\n"
            "If the task is computer-use, suggest the next step inside the active app.\n"
            "If the task is image-related, suggest visual edits or creative variations.\n"
            "Keep the user's language.\n"
            "Aim for variety: action steps, creative twists, and practical follow-ups.\n"
            "Avoid generic or unrelated suggestions.\n\n"
            f"Context:\n{history_text}\n\n"
            "Return exactly 8 suggestions as a pipe-separated list. "
            "Each suggestion should be 2-7 words, start with a relevant emoji, "
            "and avoid quotes or numbering."
        )
        try:
            response = self.generate_text(prompt)
            if not response:
                return []
            suggestions = [s.strip() for s in response.split("|") if s.strip()]
            return suggestions[:8]
        except Exception as exc:
            logger.warning("Suggestion generation failed: %s", exc)
            return []


def _extract_image_bytes(response) -> bytes | None:
    if response is None:
        return None
    generated_images = getattr(response, "generated_images", None)
    if generated_images:
        for generated in generated_images:
            image = getattr(generated, "image", None)
            if image and getattr(image, "image_bytes", None):
                return image.image_bytes
    images = getattr(response, "images", None)
    if images:
        for image in images:
            if image and getattr(image, "image_bytes", None):
                return image.image_bytes
    parts = getattr(response, "parts", None)
    if parts:
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            data = getattr(inline_data, "data", None) if inline_data else None
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
    return None


def _merge_prompt_and_text(prompt: str, text_payload: str, max_chars: int = 120000) -> str:
    if not text_payload:
        return prompt
    if len(text_payload) > max_chars:
        text_payload = text_payload[:max_chars]
        return f"{prompt}\n\nDocument (truncated):\n{text_payload}"
    return f"{prompt}\n\nDocument:\n{text_payload}"


def _is_unsupported_mime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "unsupported mime type" in message or "invalid_argument" in message


def _should_inline_text(file_path: str, mime_type: str | None) -> bool:
    extension = Path(file_path).suffix.lower()
    if extension in _INLINE_ONLY_EXTENSIONS:
        return True
    if mime_type in _INLINE_ONLY_MIME_TYPES:
        return True
    if mime_type in _RTF_MIME_TYPES:
        return True
    return False


def _extract_text_fallback(file_path: str, mime_type: str | None) -> str | None:
    extension = Path(file_path).suffix.lower()
    if extension in _TEXT_EXTENSIONS or (mime_type and mime_type.startswith("text/")):
        return _read_text_file(file_path)
    if extension == ".docx" or mime_type == _DOCX_MIME:
        return _extract_docx_text(file_path)
    if extension == ".pptx" or mime_type == _PPTX_MIME:
        return _extract_pptx_text(file_path)
    if extension == ".xlsx" or mime_type == _XLSX_MIME:
        return _extract_xlsx_text(file_path)
    if extension == ".odt" or mime_type == _ODT_MIME:
        return _extract_odt_text(file_path)
    if extension == ".rtf" or mime_type in _RTF_MIME_TYPES:
        return _extract_rtf_text(file_path)
    if extension == ".doc" or mime_type == _DOC_MIME:
        return None
    return None


def _read_text_file(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Text file read failed: %s", exc)
        return None


def _extract_docx_text(path: str) -> str | None:
    return _extract_zip_xml_text(
        path,
        member_prefix="word/",
        member_suffix=".xml",
        localnames={"t"},
    )


def _extract_pptx_text(path: str) -> str | None:
    return _extract_zip_xml_text(
        path,
        member_prefix="ppt/slides/",
        member_suffix=".xml",
        localnames={"t"},
    )


def _extract_xlsx_text(path: str) -> str | None:
    text = _extract_zip_xml_text(
        path,
        member_exact="xl/sharedStrings.xml",
        localnames={"t"},
    )
    if text:
        return text
    return _extract_zip_xml_text(
        path,
        member_prefix="xl/worksheets/",
        member_suffix=".xml",
        localnames={"t", "v"},
    )


def _extract_odt_text(path: str) -> str | None:
    return _extract_zip_xml_text(
        path,
        member_exact="content.xml",
        localnames={"p", "span"},
    )


def _extract_zip_xml_text(
    path: str,
    localnames: set[str],
    member_prefix: str | None = None,
    member_suffix: str | None = None,
    member_exact: str | None = None,
) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            selected: list[str] = []
            if member_exact:
                if member_exact not in members:
                    return None
                selected = [member_exact]
            else:
                for name in members:
                    if member_prefix and not name.startswith(member_prefix):
                        continue
                    if member_suffix and not name.endswith(member_suffix):
                        continue
                    if name.endswith(".xml"):
                        selected.append(name)
            if not selected:
                return None
            texts: list[str] = []
            for name in selected:
                try:
                    xml_bytes = archive.read(name)
                except Exception:
                    continue
                try:
                    root = ET.fromstring(xml_bytes)
                except ET.ParseError:
                    continue
                texts.extend(_collect_xml_text(root, localnames))
            joined = "\n".join(texts).strip()
            return joined or None
    except Exception as exc:
        logger.warning("Archive read failed: %s", exc)
        return None


def _collect_xml_text(root: ET.Element, localnames: set[str]) -> list[str]:
    texts: list[str] = []
    for node in root.iter():
        text = node.text
        if not text or not text.strip():
            continue
        localname = node.tag.split("}")[-1].lower()
        if localname in localnames:
            texts.append(text.strip())
    return texts


def _extract_rtf_text(path: str) -> str | None:
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.warning("RTF read failed: %s", exc)
        return None

    def _rtf_hex(match: re.Match[str]) -> str:
        try:
            return bytes.fromhex(match.group(1)).decode("latin-1")
        except Exception:
            return ""
    text = re.sub(r"\\'([0-9a-fA-F]{2})", _rtf_hex, raw)
    text = re.sub(r"\\par[d]?\s?", "\n", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = text.replace("{", "").replace("}", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    return cleaned or None


def _unreadable_file_message(file_path: str, mime_type: str | None) -> str:
    label = mime_type or Path(file_path).suffix.lower() or "unknown"
    return (
        f"Sorry, I couldn't read this file type ({label}). "
        "Please try PDF or plain text, or export the document to PDF."
    )


def _unsupported_file_type_message(label: str) -> str:
    return (
        f"Sorry, Gemini doesn't support this file type ({label}). "
        "Please try PDF or plain text, or export the document to PDF."
    )


_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_ODT_MIME = "application/vnd.oasis.opendocument.text"
_DOC_MIME = "application/msword"
_RTF_MIME_TYPES = {"application/rtf", "text/rtf"}

_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".log",
}
_INLINE_ONLY_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".odt", ".rtf"}
_INLINE_ONLY_MIME_TYPES = {
    _DOCX_MIME,
    _PPTX_MIME,
    _XLSX_MIME,
    _ODT_MIME,
}

_MB = 1024 * 1024

GEMINI_FLASH_MAX_DOCUMENT_BYTES = 50 * _MB
GEMINI_FLASH_MAX_INLINE_IMAGE_BYTES = 7 * _MB
GEMINI_IMAGE_PREVIEW_MAX_INLINE_IMAGE_BYTES = 7 * _MB

GEMINI_IMAGE_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
    }
)
GEMINI_FLASH_DOCUMENT_MIME_TYPES = frozenset(
    {
        "application/pdf",
        "text/plain",
    }
)
GEMINI_FLASH_VIDEO_MIME_TYPES = frozenset(
    {
        "video/x-flv",
        "video/quicktime",
        "video/mpeg",
        "video/mpegs",
        "video/mpg",
        "video/mp4",
        "video/webm",
        "video/wmv",
        "video/3gpp",
    }
)
GEMINI_FLASH_AUDIO_MIME_TYPES = frozenset(
    {
        "audio/x-aac",
        "audio/flac",
        "audio/mp3",
        "audio/m4a",
        "audio/mpeg",
        "audio/mpga",
        "audio/mp4",
        "audio/ogg",
        "audio/pcm",
        "audio/wav",
        "audio/webm",
    }
)


def _is_gemini_flash_model(model: str) -> bool:
    return model.startswith("gemini-3-flash")


def _file_size_bytes(file_path: str) -> int | None:
    try:
        return Path(file_path).stat().st_size
    except Exception as exc:
        logger.warning("File size check failed: %s", exc)
        return None


def _format_limit_mb(limit_bytes: int) -> str:
    return f"{int(limit_bytes / _MB)} MB"


def _file_too_large_message(file_kind: str, max_bytes: int) -> str:
    return f"{file_kind} file is too large for Gemini. Max size is {_format_limit_mb(max_bytes)}."


def _should_enforce_flash_document_limit(file_path: str, mime_type: str | None) -> bool:
    if _should_inline_text(file_path, mime_type):
        return True
    return mime_type in GEMINI_FLASH_DOCUMENT_MIME_TYPES


def _flash_document_size_error(file_path: str, mime_type: str | None) -> str | None:
    if not _should_enforce_flash_document_limit(file_path, mime_type):
        return None
    size_bytes = _file_size_bytes(file_path)
    if size_bytes is None:
        return None
    if size_bytes > GEMINI_FLASH_MAX_DOCUMENT_BYTES:
        return _file_too_large_message("Document", GEMINI_FLASH_MAX_DOCUMENT_BYTES)
    return None


def _is_supported_flash_file_type(file_path: str, mime_type: str | None) -> bool:
    if _should_inline_text(file_path, mime_type):
        return True
    if not mime_type:
        return False
    if mime_type in GEMINI_FLASH_DOCUMENT_MIME_TYPES:
        return True
    if mime_type in GEMINI_FLASH_VIDEO_MIME_TYPES:
        return True
    if mime_type in GEMINI_FLASH_AUDIO_MIME_TYPES:
        return True
    return False


def _wait_for_file_active(client, file_obj, timeout_seconds: int = 30, poll_interval: float = 0.8):
    if not file_obj or not getattr(file_obj, "name", None):
        return file_obj
    from google.genai import types

    state = getattr(file_obj, "state", None)
    if state in {None, types.FileState.ACTIVE, "ACTIVE"}:
        return file_obj
    start = time.monotonic()
    while time.monotonic() - start < timeout_seconds:
        if state in {types.FileState.ACTIVE, "ACTIVE"}:
            return file_obj
        if state in {types.FileState.FAILED, "FAILED"}:
            return file_obj
        time.sleep(poll_interval)
        try:
            file_obj = client.files.get(name=file_obj.name)
        except Exception:
            return file_obj
        state = getattr(file_obj, "state", None)
    return file_obj


def _safe_delete_file(client, file_obj) -> None:
    name = getattr(file_obj, "name", None)
    if not name:
        return
    try:
        client.files.delete(name=name)
    except Exception as exc:
        logger.warning("File cleanup failed: %s", exc)


class GeminiImageClient:
    def __init__(self, config: GeminiConfig) -> None:
        try:
            from google import genai
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise LLMClientError("google-genai is not installed") from exc

        self._client = genai.Client(api_key=config.api_key)
        self._model = config.model
        self._enable_google_search = config.enable_google_search

    def build_generate_config(self, include_thoughts: bool = False, response_modalities=None):
        return _build_generate_content_config(
            enable_google_search=self._enable_google_search,
            enable_code_execution=False,
            include_thoughts=include_thoughts,
            response_modalities=response_modalities,
        )

    def generate_image(self, prompt: str) -> bytes | None:
        if not prompt:
            return None
        if not getattr(self._client, "vertexai", False):
            return self._generate_image_via_content(prompt)
        try:
            response = self._client.models.generate_images(
                model=self._model, prompt=prompt)
            image_bytes = _extract_image_bytes(response)
            if image_bytes:
                return image_bytes
        except Exception as exc:  # pragma: no cover - network/proxy errors
            logger.warning("generate_images failed, falling back: %s", exc)
        return self._generate_image_via_content(prompt)

    def edit_image(self, prompt: str, image_path: str) -> bytes | None:
        if not prompt or not image_path:
            return None
        from google.genai import types

        if not getattr(self._client, "vertexai", False):
            return self._edit_image_via_content(prompt, image_path)
        try:
            raw_image = types.RawReferenceImage(
                reference_id=1,
                reference_image=types.Image.from_file(location=image_path),
            )
            response = self._client.models.edit_image(
                model=self._model,
                prompt=prompt,
                reference_images=[raw_image],
            )
            image_bytes = _extract_image_bytes(response)
            if image_bytes:
                return image_bytes
        except Exception as exc:  # pragma: no cover - network/proxy errors
            logger.warning("edit_image failed, falling back: %s", exc)
        return self._edit_image_via_content(prompt, image_path)

    def analyze_image(self, prompt: str, image_path: str) -> str:
        if not prompt or not image_path:
            return ""
        from google.genai import types

        with open(image_path, "rb") as handle:
            image_data = handle.read()
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        config = self.build_generate_config()
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type=mime_type),
            ],
            config=config,
        )
        if getattr(response, "text", None):
            return response.text
        return ""

    async def stream_generate_image(self, prompt: str, include_thoughts: bool = False) -> ImageStreamState:
        if not prompt:
            return ImageStreamState(chunks=_empty_stream(), get_image_bytes=lambda: None)
        from google.genai import types

        modalities = [types.Modality.IMAGE]
        if include_thoughts:
            modalities.append(types.Modality.TEXT)
        config = self.build_generate_config(
            include_thoughts=include_thoughts,
            response_modalities=modalities,
        )
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=prompt,
            config=config,
        )
        return _build_image_stream_state(stream)

    async def stream_edit_image(
        self,
        prompt: str,
        image_path: str,
        include_thoughts: bool = False,
    ) -> ImageStreamState:
        if not prompt or not image_path:
            return ImageStreamState(chunks=_empty_stream(), get_image_bytes=lambda: None)
        from google.genai import types

        with open(image_path, "rb") as handle:
            image_data = handle.read()
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        modalities = [types.Modality.IMAGE]
        if include_thoughts:
            modalities.append(types.Modality.TEXT)
        config = self.build_generate_config(
            include_thoughts=include_thoughts,
            response_modalities=modalities,
        )
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type=mime_type),
            ],
            config=config,
        )
        return _build_image_stream_state(stream)

    def _generate_image_via_content(self, prompt: str) -> bytes | None:
        from google.genai import types

        config = self.build_generate_config(
            response_modalities=[types.Modality.IMAGE],
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        return _extract_image_bytes(response)

    def _edit_image_via_content(self, prompt: str, image_path: str) -> bytes | None:
        from google.genai import types

        with open(image_path, "rb") as handle:
            image_data = handle.read()
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        config = self.build_generate_config(
            response_modalities=[types.Modality.IMAGE],
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type=mime_type),
            ],
            config=config,
        )
        return _extract_image_bytes(response)


class NoOpLLMClient(LLMClient):
    def generate_text(self, prompt: str) -> str:
        logger.warning("LLM disabled; returning empty response.")
        return ""

    def generate_with_file(self, prompt: str, file_path: str, mime_type: str | None = None) -> FileGenerationResult:
        logger.warning("LLM disabled; returning empty response.")
        return FileGenerationResult("", "LLM is not configured.")

    async def stream_text(self, prompt: str, include_thoughts: bool = False) -> AsyncIterator[StreamChunk]:
        if False:  # pragma: no cover - keeps this as an async generator
            yield StreamChunk(text="", is_thought=False)

    async def stream_with_image(
        self,
        prompt: str,
        image_path: str,
        mime_type: str | None = None,
        include_thoughts: bool = False,
    ) -> AsyncIterator[StreamChunk]:
        if False:  # pragma: no cover - keeps this as an async generator
            yield StreamChunk(text="", is_thought=False)


async def _empty_stream() -> AsyncIterator[StreamChunk]:
    if False:  # pragma: no cover - keeps this as an async generator
        yield StreamChunk(text="", is_thought=False)


def _build_text_stream_state(stream) -> TextStreamState:
    sources: dict[str, SourceAttribution] = {}

    async def _iter() -> AsyncIterator[StreamChunk]:
        async for chunk in stream:
            _merge_sources(sources, _collect_sources_from_response(chunk))
            parts = getattr(chunk, "parts", None) or []
            if parts:
                for part in parts:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        yield StreamChunk(text=text, is_thought=bool(getattr(part, "thought", False)))
                continue
            text = getattr(chunk, "text", None)
            if isinstance(text, str) and text:
                yield StreamChunk(text=text, is_thought=False)

    return TextStreamState(chunks=_iter(), get_sources=lambda: list(sources.values()))


def _build_image_stream_state(stream) -> ImageStreamState:
    image_bytes: bytes | None = None

    async def _iter() -> AsyncIterator[StreamChunk]:
        nonlocal image_bytes
        async for chunk in stream:
            new_bytes = _extract_image_bytes(chunk)
            if new_bytes:
                image_bytes = new_bytes
            parts = getattr(chunk, "parts", None) or []
            if parts:
                for part in parts:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        yield StreamChunk(text=text, is_thought=bool(getattr(part, "thought", False)))
                continue
            text = getattr(chunk, "text", None)
            if isinstance(text, str) and text:
                yield StreamChunk(text=text, is_thought=False)

    return ImageStreamState(chunks=_iter(), get_image_bytes=lambda: image_bytes)


def build_llm_client(
    api_key: str,
    model: str,
    *,
    enable_google_search: bool = False,
    enable_code_execution: bool = False,
) -> Optional[LLMClient]:
    if not api_key:
        return None
    try:
        return GeminiClient(
            GeminiConfig(
                api_key=api_key,
                model=model,
                enable_google_search=enable_google_search,
                enable_code_execution=enable_code_execution,
            )
        )
    except LLMClientError as exc:
        logger.error("LLM client unavailable: %s", exc)
        return None


def build_image_client(
    api_key: str,
    model: str,
    *,
    enable_google_search: bool = False,
) -> Optional[GeminiImageClient]:
    if not api_key:
        return None
    try:
        return GeminiImageClient(
            GeminiConfig(
                api_key=api_key,
                model=model,
                enable_google_search=enable_google_search,
                enable_code_execution=False,
            )
        )
    except LLMClientError as exc:
        logger.error("Image client unavailable: %s", exc)
        return None
