from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import logging
from pathlib import Path
from typing import Any
import ctypes

import pyautogui
from PIL import Image
from pywinauto import Desktop

# Prevent circular import by importing capture_screen only when needed or defining it before use
# Since capture_screen is defined in this file, we don't need to import it from itself.
# We will just remove the circular import line.

logger = logging.getLogger(__name__)

SCREENSHOT_DIR = Path(__file__).resolve(
).parents[2] / "artifacts" / "screenshots"


@dataclass
class Screenshot:
    path: str
    metadata: dict[str, Any]


def capture_screen() -> Screenshot:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = SCREENSHOT_DIR / f"screenshot_{timestamp}.png"
    image = pyautogui.screenshot()
    image.save(path)
    logger.info("Screenshot captured: %s", path)
    return Screenshot(path=str(path), metadata={"captured_at": timestamp})


def _vision_locate(prompt: str, llm_client: Any) -> tuple[int, int] | None:
    if not llm_client:
        return None
    screenshot = capture_screen()

    try:
        with open(screenshot.path, "rb") as img_file:
            image_data = img_file.read()
        image_size = Image.open(io.BytesIO(image_data)).size
        screen_size = pyautogui.size()
        virtual_left = ctypes.windll.user32.GetSystemMetrics(76)
        virtual_top = ctypes.windll.user32.GetSystemMetrics(77)
        virtual_width = ctypes.windll.user32.GetSystemMetrics(78)
        virtual_height = ctypes.windll.user32.GetSystemMetrics(79)
        logger.info(
            "Vision coordinate context: image=%sx%s screen=%sx%s virtual=%sx%s offset=(%s,%s)",
            image_size[0],
            image_size[1],
            screen_size[0],
            screen_size[1],
            virtual_width,
            virtual_height,
            virtual_left,
            virtual_top,
        )

        from google import genai
        from google.genai import types

        if hasattr(llm_client, "_client"):
            client = llm_client._client
        else:
            return None

        config = None
        build_config = getattr(llm_client, "build_generate_config", None)
        if callable(build_config):
            config = build_config()
        if config is None:
            config = types.GenerateContentConfig()
        if hasattr(config, "response_mime_type"):
            config.response_mime_type = "application/json"
        if hasattr(config, "tools"):
            config.tools = []
        response = client.models.generate_content(
            model=llm_client._model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type="image/png"),
            ],
            config=config,
        )

        import json

        text = getattr(response, "text", "") or ""
        cleaned = _extract_json(text)
        if not cleaned:
            return None
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Vision JSON parse failed: %s", cleaned[:200])
            return None

        if not result.get("found"):
            return None
        x = result.get("x")
        y = result.get("y")
        if x is None or y is None:
            return None
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            return None
        if image_size[0] and image_size[1] and virtual_width and virtual_height:
            if image_size != (virtual_width, virtual_height):
                scale_x = virtual_width / image_size[0]
                scale_y = virtual_height / image_size[1]
                logger.info(
                    "Scaling vision coordinates from %sx%s to %sx%s",
                    image_size[0],
                    image_size[1],
                    virtual_width,
                    virtual_height,
                )
                x *= scale_x
                y *= scale_y
            if virtual_left or virtual_top:
                logger.info(
                    "Adjusting vision coordinates by virtual screen offset (%s, %s)",
                    virtual_left,
                    virtual_top,
                )
                x += virtual_left
                y += virtual_top
        elif image_size != screen_size and image_size[0] and image_size[1]:
            scale_x = screen_size[0] / image_size[0]
            scale_y = screen_size[1] / image_size[1]
            logger.info(
                "Scaling vision coordinates from %sx%s to %sx%s",
                image_size[0],
                image_size[1],
                screen_size[0],
                screen_size[1],
            )
            x *= scale_x
            y *= scale_y
        if virtual_width and virtual_height:
            x = min(max(x, virtual_left), virtual_left + virtual_width - 1)
            y = min(max(y, virtual_top), virtual_top + virtual_height - 1)
        x_int = int(round(x))
        y_int = int(round(y))
        return x_int, y_int
    except Exception as exc:
        logger.error("Vision analysis failed: %s", exc)
        return None


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            if "{" in part:
                cleaned = part
                break
    cleaned = cleaned.strip()
    brace_index = cleaned.find("{")
    if brace_index != -1:
        cleaned = cleaned[brace_index:]
    end_index = cleaned.rfind("}")
    if end_index != -1:
        cleaned = cleaned[: end_index + 1]
    return cleaned


def locate_ui_point(instruction: str, llm_client: Any) -> tuple[int, int] | None:
    if not instruction:
        return None
    prompt = f"""
You are helping control a Windows desktop UI.
Find the UI element described here: "{instruction}".
Return ONLY a JSON object with this structure:
{{
  "found": true,
  "x": <integer_x_coordinate>,
  "y": <integer_y_coordinate>,
  "confidence": <float_0_to_1>
}}
If the element is not visible, return {{ "found": false }}.
""".strip()
    return _vision_locate(prompt, llm_client)


def locate_text_center(label: str, llm_client: Any, context: str | None = None) -> tuple[int, int] | None:
    if not label:
        return None
    extra = f"Context: {context}." if context else ""
    prompt = f"""
You are helping control a Windows desktop UI. {extra}
Find the text label exactly matching "{label}".
Return the center point of the text itself (not the avatar/icon, not blank space above).
Return ONLY a JSON object with this structure:
{{
  "found": true,
  "x": <integer_x_coordinate>,
  "y": <integer_y_coordinate>,
  "confidence": <float_0_to_1>
}}
If the element is not visible, return {{ "found": false }}.
""".strip()
    return _vision_locate(prompt, llm_client)


def locate_whatsapp_search_result(contact_name: str, llm_client: Any) -> tuple[int, int] | None:
    if not contact_name:
        return None
    prompt = f"""
You are automating WhatsApp Desktop.
Look at the left panel where the search results are listed (below the search bar).
Find the contact row for "{contact_name}".
The result might look like "{contact_name}" or "{contact_name} *" or similar.
Ignore the narrow navigation bar on the far left (with icons like chat bubbles, phone, gears).
Do not return the search input field or the typed query in the search bar.
Do not click anything in the right chat panel.
Focus ONLY on the list of people/chats.
Return the coordinates for the CENTER of the clickable row area for "{contact_name}".
It must be a safe point to click to open that chat (avoid clicking the avatar image if possible, aim for the text or empty space in the row).
Return ONLY a JSON object with this structure:
{{
  "found": true,
  "x": <integer_x_coordinate>,
  "y": <integer_y_coordinate>,
  "confidence": <float_0_to_1>
}}
If the contact is not visible in the list, return {{ "found": false }}.
""".strip()
    return _vision_locate(prompt, llm_client)


def find_spotify_first_result(llm_client: Any) -> dict[str, Any] | None:
    prompt = """
You are looking at the Spotify desktop app search results.
Find the first playable result row in the results list (not the search bar).
Prefer the first row under the "Songs" section.
If "Songs" is not visible, use the first visible result row under "Top result" or the first visible result row.
Return ONLY a JSON object with this structure:
{
  "found": true,
  "x": <integer_x_coordinate>,
  "y": <integer_y_coordinate>,
  "confidence": <float_0_to_1>
}
If the element is not visible, return { "found": false }.
""".strip()
    point = _vision_locate(prompt, llm_client)
    if not point:
        return None
    x_int, y_int = point
    logger.info(
        "UI element matched via Gemini Vision (Spotify first result) at (%s, %s)",
        x_int,
        y_int,
    )
    return {
        "name": "spotify_first_result",
        "control_type": "vision_spotify_result",
        "rectangle": {
            "left": x_int - 5,
            "top": y_int - 5,
            "right": x_int + 5,
            "bottom": y_int + 5,
        },
        "center_point": (x_int, y_int),
        "handle": None,
    }


def find_spotify_pause_button(llm_client: Any) -> dict[str, Any] | None:
    prompt = """
You are looking at the Spotify desktop app.
Find the Pause button (the icon with two vertical bars) in the player controls, usually bottom left.
Return ONLY a JSON object with this structure:
{
  "found": true,
  "x": <integer_x_coordinate>,
  "y": <integer_y_coordinate>,
  "confidence": <float_0_to_1>
}
If the Pause button is not visible, return { "found": false }.
""".strip()
    point = _vision_locate(prompt, llm_client)
    if not point:
        return None
    x_int, y_int = point
    logger.info(
        "UI element matched via Gemini Vision (Spotify pause button) at (%s, %s)",
        x_int,
        y_int,
    )
    return {
        "name": "spotify_pause_button",
        "control_type": "vision_spotify_pause",
        "rectangle": {
            "left": x_int - 5,
            "top": y_int - 5,
            "right": x_int + 5,
            "bottom": y_int + 5,
        },
        "center_point": (x_int, y_int),
        "handle": None,
    }


def find_spotify_row_play_button(llm_client: Any) -> dict[str, Any] | None:
    prompt = """
You are looking at the Spotify desktop app search results.
Find the Play button that appears when hovering the first song row in the Songs list.
The button is a small green circle with a triangle (play icon) near the row.
Return ONLY a JSON object with this structure:
{
  "found": true,
  "x": <integer_x_coordinate>,
  "y": <integer_y_coordinate>,
  "confidence": <float_0_to_1>
}
If the Play button is not visible, return { "found": false }.
""".strip()
    point = _vision_locate(prompt, llm_client)
    if not point:
        return None
    x_int, y_int = point
    logger.info(
        "UI element matched via Gemini Vision (Spotify row play button) at (%s, %s)",
        x_int,
        y_int,
    )
    return {
        "name": "spotify_row_play_button",
        "control_type": "vision_spotify_row_play",
        "rectangle": {
            "left": x_int - 5,
            "top": y_int - 5,
            "right": x_int + 5,
            "bottom": y_int + 5,
        },
        "center_point": (x_int, y_int),
        "handle": None,
    }


def find_spotify_top_result_play_button(llm_client: Any) -> dict[str, Any] | None:
    prompt = """
You are looking at the Spotify desktop app search results.
Find the large green Play button shown on the "Top result" card.
The button is a green circle
with a triangle icon, typically on the right side of the top result card.
Return ONLY a JSON object with this structure:
{
  "found": true,
  "x": <integer_x_coordinate>,
  "y": <integer_y_coordinate>,
  "confidence": <float_0_to_1>
}
If the Play button is not visible, return { "found": false }.
""".strip()
    point = _vision_locate(prompt, llm_client)
    if not point:
        return None
    x_int, y_int = point
    logger.info(
        "UI element matched via Gemini Vision (Spotify top result play button) at (%s, %s)",
        x_int,
        y_int,
    )
    return {
        "name": "spotify_top_result_play_button",
        "control_type": "vision_spotify_top_play",
        "rectangle": {
            "left": x_int - 5,
            "top": y_int - 5,
            "right": x_int + 5,
            "bottom": y_int + 5,
        },
        "center_point": (x_int, y_int),
        "handle": None,
    }


def find_spotify_player_play_button(llm_client: Any) -> dict[str, Any] | None:
    prompt = """
You are looking at the Spotify desktop app player controls.
Find the Play button in the bottom center control bar (triangle icon).
Return ONLY a JSON object with this structure:
{
  "found": true,
  "x": <integer_x_coordinate>,
  "y": <integer_y_coordinate>,
  "confidence": <float_0_to_1>
}
If the Play button is not visible, return { "found": false }.
""".strip()
    point = _vision_locate(prompt, llm_client)
    if not point:
        return None
    x_int, y_int = point
    logger.info(
        "UI element matched via Gemini Vision (Spotify player play button) at (%s, %s)",
        x_int,
        y_int,
    )
    return {
        "name": "spotify_player_play_button",
        "control_type": "vision_spotify_player_play",
        "rectangle": {
            "left": x_int - 5,
            "top": y_int - 5,
            "right": x_int + 5,
            "bottom": y_int + 5,
        },
        "center_point": (x_int, y_int),
        "handle": None,
    }


def find_ui_element(
    query: str,
    llm_client: Any = None,
    use_vision: bool = True,
) -> dict[str, Any] | None:
    query_lower = query.lower()
    desktop = Desktop(backend="uia")

    def _scan_window(window) -> dict[str, Any] | None:
        try:
            for control in window.descendants():
                name = (control.window_text() or "").strip()
                if query_lower in name.lower():
                    rect = control.rectangle()
                    logger.info("UI element matched via Pywinauto: %s", name)
                    return {
                        "name": name,
                        "control_type": control.friendly_class_name(),
                        "rectangle": {
                            "left": rect.left,
                            "top": rect.top,
                            "right": rect.right,
                            "bottom": rect.bottom,
                        },
                        "handle": control,
                    }
        except Exception:
            return None
        return None

    # 1. Try active window first to avoid matching background apps.
    active_window = None
    try:
        active_window = desktop.get_active()
    except Exception:
        active_window = None

    if active_window:
        match = _scan_window(active_window)
        if match:
            return match

    # 2. If no LLM fallback, scan other windows to increase coverage.
    if llm_client is None:
        for window in desktop.windows():
            if active_window is not None:
                try:
                    if window.handle == active_window.handle:
                        continue
                except Exception:
                    pass
            match = _scan_window(window)
            if match:
                return match

    # 3. Try Gemini Vision (Slow, resilient fallback)
    if llm_client and use_vision:
        logger.info(
            "Element not found via Pywinauto in active window. Attempting Gemini Vision analysis for: %s",
            query,
        )
        prompt = f"""
I need to click on a UI element that matches the text or description: "{query}".
Look at this screenshot and provide the coordinates of the center of this element.
Return ONLY a JSON object with this structure:
{{
  "found": true,
  "x": <integer_x_coordinate>,
  "y": <integer_y_coordinate>,
  "confidence": <float_0_to_1>
}}
If the element is not visible, return {{ "found": false }}.
""".strip()
        point = _vision_locate(prompt, llm_client)
        if point:
            x_int, y_int = point
            logger.info(
                "UI element matched via Gemini Vision: %s at (%s, %s)",
                query,
                x_int,
                y_int,
            )
            return {
                "name": query,
                "control_type": "vision_detected",
                "rectangle": {
                    "left": x_int - 5,
                    "top": y_int - 5,
                    "right": x_int + 5,
                    "bottom": y_int + 5,
                },
                "center_point": (x_int, y_int),
                "handle": None,  # No handle available for vision detection
            }

    return None
