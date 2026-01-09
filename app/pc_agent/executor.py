from __future__ import annotations

import logging
import os
import re
import time
import tkinter as tk

from dataclasses import dataclass

from typing import Any

import pyautogui
from pywinauto import Desktop
from playwright.sync_api import sync_playwright

from app.agent.state import AgentContext
from app.pc_agent.planner import PlannedAction
from app.pc_agent.safety import is_high_risk
from app.pc_agent.vision import (
    find_spotify_first_result,
    find_spotify_pause_button,
    find_spotify_player_play_button,
    find_spotify_row_play_button,
    find_spotify_top_result_play_button,
    find_ui_element,
    locate_ui_point,
    locate_text_center,
    locate_whatsapp_search_result,
)

logger = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    pass


COMMON_APP_SHORTCUTS: dict[str, list[dict[str, Any]]] = {
    "spotify": [
        {"keys": ["space"], "keywords": ("play", "pause", "resume", "toggle")},
        {"keys": ["ctrl", "right"], "keywords": ("next", "skip", "forward")},
        {"keys": ["ctrl", "left"], "keywords": ("previous", "back")},
        {"keys": ["ctrl", "l"], "keywords": ("search", "find")},
    ],
    "browser": [
        {"keys": ["ctrl", "l"], "keywords": ("address", "search", "url", "find")},
        {"keys": ["ctrl", "t"], "keywords": ("new tab", "open tab")},
        {"keys": ["ctrl", "w"], "keywords": ("close tab", "close this tab")},
        {"keys": ["ctrl", "r"], "keywords": ("refresh", "reload")},
        {"keys": ["alt", "left"], "keywords": ("back", "previous page")},
        {"keys": ["alt", "right"], "keywords": ("forward", "next page")},
    ],
    "whatsapp": [
        {"keys": ["ctrl", "f"], "keywords": ("search", "find")},
    ],
}
COMMON_APP_SHORTCUTS["*"] = [
    {"keys": ["ctrl", "f"], "keywords": ("find", "search")},
    {"keys": ["ctrl", "l"], "keywords": ("address", "url")},
]

MENU_PATH_SEPARATORS = (">", "->", "\u00bb")
URL_PATTERN = re.compile(r"(https?://[^\s]+)", re.IGNORECASE)


@dataclass
class PCSession:
    playwright: object
    browser: object
    page: object


class WebBrowserProvider:
    def __init__(self, agent_context: AgentContext) -> None:
        self._agent_context = agent_context

    def navigate(self, url: str) -> None:
        session = _ensure_browser(self._agent_context)
        session.page.goto(str(url), wait_until="domcontentloaded")

    def click_text(self, text: str, timeout_seconds: float) -> bool:
        session = _ensure_browser(self._agent_context)
        remaining = max(0.5, timeout_seconds)
        try:
            session.page.get_by_text(text).first.click(timeout=int(remaining * 1000))
            return True
        except Exception:
            pass
        try:
            session.page.get_by_role("button", name=text).first.click(timeout=int(remaining * 1000))
            return True
        except Exception:
            pass
        try:
            selector = f"input[type=submit][value=\"{text}\"]"
            session.page.locator(selector).first.click(timeout=int(remaining * 1000))
            return True
        except Exception:
            return False

    def type_text(self, value: str, selector: str | None = None, label: str | None = None) -> bool:
        if not selector and not label:
            return False
        session = _ensure_browser(self._agent_context)
        try:
            if selector:
                session.page.locator(selector).first.fill(value)
            else:
                session.page.get_by_label(label).first.fill(value)
            return True
        except Exception:
            return False

    def fill_form(self, fields: list[dict[str, Any]]) -> None:
        session = _ensure_browser(self._agent_context)
        for field in fields:
            label = field.get("label")
            value = field.get("value")
            if label is None or value is None:
                raise ExecutionError("fill_form field missing label/value")
            session.page.get_by_label(str(label)).first.fill(str(value))

    def upload_file(self, selector: str, path: str) -> None:
        session = _ensure_browser(self._agent_context)
        session.page.set_input_files(selector, path)

    def type_search(self, value: str, press_enter: bool = True) -> bool:
        session = _ensure_browser(self._agent_context)
        selectors = [
            "textarea[name='q']",
            "input[name='q']",
            "input[aria-label*='Search']",
            "textarea[aria-label*='Search']",
            "input[type='search']",
        ]
        for selector in selectors:
            try:
                locator = session.page.locator(selector).first
                locator.fill(value)
                if press_enter:
                    session.page.keyboard.press("Enter")
                return True
            except Exception:
                continue
        return False


class DesktopProvider:
    def __init__(self, agent_context: AgentContext) -> None:
        self._agent_context = agent_context

    def click_text(
        self,
        text: str,
        llm_client: Any,
        timeout_seconds: float,
        instruction: str | None,
    ) -> None:
        _click_text(
            self._agent_context,
            text,
            llm_client=llm_client,
            timeout_seconds=timeout_seconds,
            instruction=instruction,
            prefer_web=False,
        )

    def vision_click(self, instruction: str, llm_client: Any) -> None:
        point = locate_ui_point(str(instruction), llm_client)
        if not point:
            raise ExecutionError(f"Unable to locate target via vision: {instruction}")
        pyautogui.click(point[0], point[1])

    def mouse_click(self, x: int, y: int) -> None:
        pyautogui.click(x, y)


def execute_action(
    action: PlannedAction,
    agent_context: AgentContext,
    llm_client: Any = None,
    allow_run_executable: bool = False,
) -> None:
    if agent_context.stop_requested:
        raise ExecutionError("Execution halted due to STOP request.")

    if is_high_risk(action.name):
        logger.warning(
            "High risk action requires explicit confirmation: %s", action.name)

    logger.info("Executing action: %s", action.description)
    _dispatch_action(action, agent_context, llm_client)


def _dispatch_action(action: PlannedAction, agent_context: AgentContext, llm_client: Any = None) -> None:
    if action.name == "open_browser":
        if _use_system_browser(agent_context):
            browser_name = str(action.payload.get("name") or agent_context.pc_state.get("browser_name") or "Chrome")
            window_title = action.payload.get("window_title") or agent_context.pc_state.get(
                "browser_window_title")
            _open_app_from_start_menu(
                browser_name,
                window_title=str(window_title) if window_title else None,
            )
            agent_context.pc_state["active_app_hint"] = "browser"
            agent_context.pc_state["browser_mode"] = "system"
            return
        _ensure_browser(agent_context)
        agent_context.pc_state["browser_mode"] = "playwright"
        return
    if action.name == "navigate":
        url = _resolve_navigation_url(action.payload.get("url"), agent_context)
        if not url:
            raise ExecutionError("navigate action requires url")
        web_provider = WebBrowserProvider(agent_context)
        if _should_use_web_provider(action, agent_context, url=url):
            web_provider.navigate(str(url))
            agent_context.pc_state["browser_mode"] = "playwright"
            agent_context.pc_state["active_app_hint"] = "browser"
            return
        _navigate_system_browser(str(url), agent_context)
        agent_context.pc_state["browser_mode"] = "system"
        agent_context.pc_state["active_app_hint"] = "browser"
        return
    if action.name == "click_text":
        text = action.payload.get("text")
        if not text:
            raise ExecutionError("click_text requires text")
        timeout_seconds = float(action.payload.get("timeout_seconds", 10))
        instruction = action.payload.get("instruction") or action.description
        if _is_search_step(action.description, instruction) and _is_whatsapp_active_window(agent_context):
            _focus_whatsapp_search(agent_context, llm_client)
            return
        if _is_chat_selection_step(action.description, instruction) and _is_whatsapp_active_window(agent_context):
            _select_whatsapp_chat_result(text, agent_context, llm_client)
            return
        if "search" in text.lower():
            agent_context.pc_state["search_context"] = True
        description = str(action.description or "").lower()
        if _is_play_first_result_request(text, description) or _is_song_selection_request(
            agent_context, text, description
        ):
            if _is_spotify_active_window(agent_context):
                if _click_spotify_first_result(llm_client):
                    return
                raise ExecutionError(
                    "Unable to play first search result in Spotify.")
        if _maybe_use_shortcut(action, agent_context):
            return
        web_provider = WebBrowserProvider(agent_context)
        if _should_use_web_provider(action, agent_context) and web_provider.click_text(
            str(text),
            timeout_seconds=timeout_seconds,
        ):
            return
        DesktopProvider(agent_context).click_text(
            str(text),
            llm_client=llm_client,
            timeout_seconds=timeout_seconds,
            instruction=instruction,
        )
        return
    if action.name == "vision_click":
        instruction = action.payload.get("instruction") or action.description
        if not instruction:
            raise ExecutionError("vision_click requires instruction")
        if not llm_client:
            raise ExecutionError("vision_click requires llm_client")
        if _is_search_step(action.description, instruction) and _is_whatsapp_active_window(agent_context):
            _focus_whatsapp_search(agent_context, llm_client)
            return
        if _maybe_use_shortcut(action, agent_context):
            return
        DesktopProvider(agent_context).vision_click(str(instruction), llm_client)
        return
    if action.name == "type_text":
        value = action.payload.get("value")
        if value is None:
            raise ExecutionError("type_text requires value")
        press_enter = action.payload.get("press_enter")
        selector = action.payload.get("selector")
        label = action.payload.get("label")
        web_provider = WebBrowserProvider(agent_context)
        if (selector or label) and _should_use_web_provider(action, agent_context):
            if web_provider.type_text(str(value), selector=selector, label=label):
                return
        if _maybe_type_web_search(action, agent_context, web_provider, str(value), press_enter):
            return
        instruction = action.payload.get(
            "instruction") or action.payload.get("target")
        if not instruction:
            instruction = action.description or None
        if not instruction and label:
            instruction = f"click the input labeled '{label}'"
        is_whatsapp = _is_whatsapp_active_window(agent_context)
        is_search = _is_search_step(action.description, instruction)
        is_message = _is_message_step(action.description, instruction)
        if is_search and is_whatsapp:
            _focus_whatsapp_search(agent_context, llm_client)
        if instruction and llm_client and not (is_whatsapp and (is_search or is_message)):
            point = locate_ui_point(str(instruction), llm_client)
            if not point:
                raise ExecutionError(
                    f"Unable to locate input via vision: {instruction}")
            pyautogui.click(point[0], point[1])
            time.sleep(0.2)
        if is_message and is_whatsapp:
            _focus_whatsapp_window()
            _focus_whatsapp_message_input(llm_client)
        if agent_context.pc_state.pop("force_search_focus", False):
            try:
                logger.info(
                    "Applying Ctrl+L before typing to focus search.")
                pyautogui.hotkey("ctrl", "l")
                time.sleep(0.2)
                agent_context.pc_state["search_context"] = True
            except Exception:
                pass
        if is_search and is_whatsapp:
            _replace_input_text(str(value))
        elif is_message and is_whatsapp:
            _replace_input_text(str(value))
        else:
            pyautogui.typewrite(str(value), interval=0.02)
        if is_search and is_whatsapp:
            _click_whatsapp_search_result(
                str(value), agent_context, llm_client)
        if is_message and is_whatsapp:
            should_send = _should_press_enter_for_send(
                action.description, instruction)
            if press_enter is None:
                press_enter = should_send
            elif press_enter is False and should_send:
                press_enter = True
            if press_enter:
                pyautogui.press("enter")
        if not (is_whatsapp and is_message):
            if press_enter is None and _should_press_enter_for_send(
                action.description, instruction
            ):
                press_enter = True
            if press_enter:
                pyautogui.press("enter")
        if agent_context.pc_state.pop("search_context", False):
            agent_context.pc_state["last_typed_text"] = str(value)
            agent_context.pc_state["pending_play_after_enter"] = True
        return
    if action.name == "vision_type":
        value = action.payload.get("value")
        instruction = action.payload.get("instruction") or action.description
        press_enter = action.payload.get("press_enter")
        if value is None:
            raise ExecutionError("vision_type requires value")
        if not instruction:
            raise ExecutionError("vision_type requires instruction")
        if not llm_client:
            raise ExecutionError("vision_type requires llm_client")
        is_whatsapp = _is_whatsapp_active_window(agent_context)
        is_search = _is_search_step(action.description, instruction)
        is_message = _is_message_step(action.description, instruction)
        if is_search and is_whatsapp:
            _focus_whatsapp_search(agent_context, llm_client)
        if not (is_whatsapp and (is_search or is_message)):
            point = locate_ui_point(str(instruction), llm_client)
            if not point:
                raise ExecutionError(
                    f"Unable to locate input via vision: {instruction}")
            pyautogui.click(point[0], point[1])
            time.sleep(0.2)
        if is_message and is_whatsapp:
            _focus_whatsapp_window()
            _focus_whatsapp_message_input(llm_client)
        if is_search and is_whatsapp:
            _replace_input_text(str(value))
        elif is_message and is_whatsapp:
            _replace_input_text(str(value))
        else:
            pyautogui.typewrite(str(value), interval=0.02)
        if is_search and is_whatsapp:
            _click_whatsapp_search_result(
                str(value), agent_context, llm_client)
        if is_message and is_whatsapp:
            should_send = _should_press_enter_for_send(
                action.description, instruction)
            if press_enter is None:
                press_enter = should_send
            elif press_enter is False and should_send:
                press_enter = True
        if press_enter is None:
            press_enter = _should_press_enter_for_send(
                action.description, instruction)
        if press_enter:
            pyautogui.press("enter")
        return
    if action.name == "scroll":
        direction = str(action.payload.get("direction", "down")).lower()
        amount = int(action.payload.get("amount", 600))
        x = action.payload.get("x")
        y = action.payload.get("y")
        if x is not None and y is not None:
            pyautogui.moveTo(x, y, duration=0.1)
        if direction in {"down", "up"}:
            delta = -amount if direction == "down" else amount
            pyautogui.scroll(delta)
            return
        if direction in {"left", "right"}:
            delta = amount if direction == "right" else -amount
            if hasattr(pyautogui, "hscroll"):
                pyautogui.hscroll(delta)
                return
            raise ExecutionError("Horizontal scrolling is not supported on this platform.")
        raise ExecutionError(f"scroll direction not supported: {direction}")
    if action.name == "drag":
        start_x = action.payload.get("start_x")
        start_y = action.payload.get("start_y")
        end_x = action.payload.get("end_x")
        end_y = action.payload.get("end_y")
        from_instruction = action.payload.get("from_instruction") or action.payload.get("instruction")
        to_instruction = action.payload.get("to_instruction") or action.payload.get("target_instruction")
        duration = float(action.payload.get("duration", 0.4))
        if start_x is None or start_y is None:
            if not from_instruction or not llm_client:
                raise ExecutionError("drag requires start coordinates or from_instruction")
            point = locate_ui_point(str(from_instruction), llm_client)
            if not point:
                raise ExecutionError(
                    f"Unable to locate drag start via vision: {from_instruction}")
            start_x, start_y = point
        if end_x is None or end_y is None:
            if not to_instruction or not llm_client:
                raise ExecutionError("drag requires end coordinates or to_instruction")
            point = locate_ui_point(str(to_instruction), llm_client)
            if not point:
                raise ExecutionError(
                    f"Unable to locate drag end via vision: {to_instruction}")
            end_x, end_y = point
        pyautogui.moveTo(start_x, start_y, duration=0.1)
        pyautogui.dragTo(end_x, end_y, duration=duration)
        return
    if action.name == "wait":
        seconds = float(action.payload.get("seconds", action.payload.get("duration", 1)))
        if seconds > 0:
            time.sleep(seconds)
        return
    if action.name == "fill_form":
        fields = action.payload.get("fields", [])
        if not fields:
            raise ExecutionError("fill_form requires fields payload")
        WebBrowserProvider(agent_context).fill_form(fields)
        return
    if action.name == "upload_file":
        selector = action.payload.get("selector")
        path = action.payload.get("path")
        if not selector or not path:
            raise ExecutionError("upload_file requires selector and path")
        WebBrowserProvider(agent_context).upload_file(selector, path)
        return
    if action.name == "focus_window":
        title = action.payload.get("title")
        if not title:
            raise ExecutionError("focus_window requires title")
        window = Desktop(backend="uia").window(title_re=title)
        window.set_focus()
        if "spotify" in _normalize_text(str(title)):
            agent_context.pc_state["active_app_hint"] = "spotify"
        return
    if action.name == "mouse_move":
        x = action.payload.get("x")
        y = action.payload.get("y")
        if x is None or y is None:
            raise ExecutionError("mouse_move requires x/y")
        pyautogui.moveTo(x, y, duration=0.2)
        return
    if action.name == "mouse_click":
        x = action.payload.get("x")
        y = action.payload.get("y")
        if x is None or y is None:
            raise ExecutionError("mouse_click requires x/y")
        if _maybe_use_shortcut(action, agent_context):
            return
        DesktopProvider(agent_context).mouse_click(x, y)
        return
    if action.name == "keypress":
        key = action.payload.get("key")
        if not key:
            raise ExecutionError("keypress requires key")
        key_value = str(key).lower().replace(" ", "")
        if key_value in {"ctrl+l", "ctrl+f"}:
            agent_context.pc_state["search_context"] = True
        if key_value == "enter":
            pending_query = agent_context.pc_state.get("pending_search_query")
            last_typed = agent_context.pc_state.get("last_typed_text")
            if agent_context.pc_state.get("search_context") and pending_query and not last_typed:
                try:
                    logger.info("Typing pending search query before Enter.")
                    pyautogui.typewrite(str(pending_query), interval=0.02)
                    agent_context.pc_state["last_typed_text"] = str(
                        pending_query)
                    agent_context.pc_state.pop("pending_search_query", None)
                    agent_context.pc_state.pop("search_context", None)
                    agent_context.pc_state["pending_play_after_enter"] = True
                except Exception:
                    pass
        if "+" in key_value:
            keys = [part for part in key_value.split("+") if part]
            if not keys:
                raise ExecutionError("keypress requires key")
            pyautogui.hotkey(*keys)
        else:
            pyautogui.press(key_value)
        if key_value in {"ctrl+c", "ctrl+shift+c"}:
            _capture_handoff_from_clipboard(agent_context)
        if key_value == "enter" and agent_context.pc_state.pop("pending_play_after_enter", False):
            if not agent_context.pc_state.get("allow_auto_play_after_enter", True):
                return
            if _is_spotify_active_window(agent_context):
                if _click_spotify_first_result(llm_client):
                    return
                raise ExecutionError(
                    "Unable to play first search result in Spotify.")
            search_text = str(agent_context.pc_state.get(
                "last_typed_text", "")).strip()
            candidates: list[str] = []
            if search_text:
                candidates.append(search_text)
                lowered = search_text.lower()
                if " by " in lowered:
                    candidates.append(
                        search_text[: lowered.index(" by ")].strip())
                if "-" in search_text:
                    candidates.append(search_text.split("-", 1)[0].strip())
                words = search_text.split()
                if len(words) > 4:
                    candidates.append(" ".join(words[:4]))
            candidates.append("Play")

            seen = set()
            for candidate in candidates:
                candidate = candidate.strip()
                if not candidate or candidate.lower() in seen:
                    continue
                seen.add(candidate.lower())
                try:
                    _click_text(agent_context, candidate,
                                llm_client, timeout_seconds=3)
                    return
                except ExecutionError:
                    continue
            raise ExecutionError(
                "Unable to click search result or Play button after Enter.")
        return
    if action.name == "open_app":
        app_name = action.payload.get("name")
        if not app_name:
            raise ExecutionError("open_app requires name")
        window_title = action.payload.get("window_title")
        timeout_seconds = float(action.payload.get("timeout_seconds", 45))
        _open_app_from_start_menu(
            str(app_name),
            window_title=str(window_title) if window_title else None,
            timeout_seconds=timeout_seconds,
        )
        if "spotify" in _normalize_text(str(app_name)):
            agent_context.pc_state["active_app_hint"] = "spotify"
            time.sleep(0.4)
            _focus_spotify_window()
        if "whatsapp" in _normalize_text(str(app_name)):
            agent_context.pc_state["active_app_hint"] = "whatsapp"
        return
    if action.name == "wait_for_window":
        title = action.payload.get("title")
        if not title:
            raise ExecutionError("wait_for_window requires title")
        timeout_seconds = float(action.payload.get("timeout_seconds", 30))
        focus = bool(action.payload.get("focus", True))
        if not _wait_for_window(title, timeout_seconds=timeout_seconds, focus=focus):
            raise ExecutionError(f"Window not found within timeout: {title}")
        if "spotify" in _normalize_text(str(title)):
            agent_context.pc_state["active_app_hint"] = "spotify"
        return
    if action.name == "run_executable":
        path = action.payload.get("path")
        if not path:
            raise ExecutionError("run_executable requires path")
        if not allow_run_executable:
            raise ExecutionError(
                "run_executable must be executed via explicit user approval gate")
        os.startfile(path)
        return

    raise ExecutionError(f"Unsupported action: {action.name}")


def _should_use_web_provider(
    action: PlannedAction,
    agent_context: AgentContext,
    url: str | None = None,
) -> bool:
    if action.name in {"navigate", "fill_form", "upload_file"}:
        return True
    payload = action.payload or {}
    if url and _looks_like_url(str(url)):
        return True
    if payload.get("selector") or payload.get("label"):
        return True
    combined = " ".join(
        str(value)
        for value in (
            action.description,
            payload.get("text"),
            payload.get("instruction"),
            payload.get("url"),
        )
        if value
    )
    if _contains_web_hint(combined):
        return True
    return agent_context.pc_state.get("active_app_hint") == "browser"


def _resolve_navigation_url(url: str | None, agent_context: AgentContext) -> str | None:
    candidate = str(url).strip() if url else ""
    if candidate.lower() in {"clipboard", "last_copied_url", "last_url"}:
        candidate = ""
    if not candidate:
        handoff = str(agent_context.pc_state.get("handoff_url", "") or "").strip()
        if handoff:
            candidate = handoff
    if not candidate:
        clipboard = _get_clipboard_text()
        candidate = _extract_url_from_text(clipboard) or ""
    if candidate:
        agent_context.pc_state["handoff_url"] = candidate
    return candidate or None


def _capture_handoff_from_clipboard(agent_context: AgentContext) -> None:
    clipboard = _get_clipboard_text()
    if clipboard:
        agent_context.pc_state["last_clipboard_text"] = clipboard
    url = _extract_url_from_text(clipboard)
    if url:
        agent_context.pc_state["handoff_url"] = url


def _get_clipboard_text() -> str:
    try:
        root = tk.Tk()
        root.withdraw()
        text = root.clipboard_get()
        root.update()
        root.destroy()
        return str(text)
    except Exception:
        return ""


def _extract_url_from_text(text: str) -> str | None:
    if not text:
        return None
    match = URL_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _looks_like_url(value: str) -> bool:
    return bool(URL_PATTERN.search(value or ""))


def _contains_web_hint(text: str) -> bool:
    normalized = _normalize_text(text or "")
    if "http" in normalized or "www." in normalized:
        return True
    return any(
        token in normalized
        for token in (
            "linkedin",
            "github",
            "google",
            "gmail",
            "outlook",
            "calendar",
            "drive",
            "docs",
            "forms",
            "browser",
            "website",
            "web page",
            "url",
        )
    )


def _maybe_use_shortcut(action: PlannedAction, agent_context: AgentContext) -> bool:
    if action.name not in {"click_text", "vision_click", "mouse_click"}:
        return False
    if _should_skip_shortcut(action, agent_context):
        return False
    shortcut = _find_shortcut_for_action(action, agent_context)
    if not shortcut:
        return False
    if len(shortcut) == 1:
        pyautogui.press(shortcut[0])
    else:
        pyautogui.hotkey(*shortcut)
    return True


def _should_skip_shortcut(action: PlannedAction, agent_context: AgentContext) -> bool:
    description = str(action.description or "")
    payload = action.payload or {}
    text = str(payload.get("text", "") or "")
    instruction = str(payload.get("instruction", "") or "")
    if _is_whatsapp_active_window(agent_context):
        if _is_search_step(description, instruction) or _is_chat_selection_step(description, instruction):
            return True
    normalized = _normalize_text(f"{description} {text} {instruction}")
    if "result" in normalized:
        return True
    if _is_play_first_result_request(text, normalized):
        return True
    return _is_song_selection_request(agent_context, text, normalized)


def _find_shortcut_for_action(action: PlannedAction, agent_context: AgentContext) -> list[str] | None:
    payload = action.payload or {}
    text = str(payload.get("text", "") or "")
    instruction = str(payload.get("instruction", "") or "")
    description = str(action.description or "")
    normalized = _normalize_text(f"{description} {text} {instruction}")
    if not normalized:
        return None
    app_key = _resolve_app_shortcut_key(agent_context)
    candidates = COMMON_APP_SHORTCUTS.get(app_key, []) + COMMON_APP_SHORTCUTS.get("*", [])
    for rule in candidates:
        if any(keyword in normalized for keyword in rule.get("keywords", ())):
            keys = [str(key) for key in rule.get("keys", []) if key]
            return keys or None
    return None


def _resolve_app_shortcut_key(agent_context: AgentContext) -> str:
    hint = str(agent_context.pc_state.get("active_app_hint") or "").lower()
    if hint:
        return hint
    if _is_spotify_active_window(agent_context):
        return "spotify"
    if _is_whatsapp_active_window(agent_context):
        return "whatsapp"
    if agent_context.pc_state.get("browser_mode") or _use_system_browser(agent_context):
        return "browser"
    return "*"


def _maybe_type_web_search(
    action: PlannedAction,
    agent_context: AgentContext,
    web_provider: WebBrowserProvider,
    value: str,
    press_enter: bool | None,
) -> bool:
    if not _should_use_web_provider(action, agent_context):
        return False
    description = str(action.description or "")
    instruction = str(action.payload.get("instruction", "") or "")
    if not _is_search_step(description, instruction):
        return False
    if press_enter is None:
        press_enter = True
    return web_provider.type_search(value, press_enter=bool(press_enter))


def _should_use_web_dom(agent_context: AgentContext, prefer_web: bool | None) -> bool:
    if prefer_web is False:
        return False
    if _is_spotify_active_window(agent_context) or _is_whatsapp_active_window(agent_context):
        return False
    hint = agent_context.pc_state.get("active_app_hint")
    if hint and hint != "browser":
        return False
    if prefer_web is True:
        return True
    return _is_browser_active_window(agent_context) or hint == "browser"


def _maybe_use_alt_menu_navigation(
    text: str,
    instruction: str | None,
    agent_context: AgentContext,
) -> bool:
    if agent_context.pc_state.get("active_app_hint") == "browser":
        return False
    combined = f"{text} {instruction or ''}".strip()
    tokens = _parse_menu_path(combined)
    if not tokens:
        return False
    try:
        pyautogui.press("alt")
        time.sleep(0.1)
        for token in tokens:
            key = _menu_access_key(token)
            if not key:
                return False
            pyautogui.press(key)
            time.sleep(0.1)
        return True
    except Exception:
        return False


def _parse_menu_path(text: str) -> list[str]:
    if not text:
        return []
    for separator in MENU_PATH_SEPARATORS:
        if separator in text:
            parts = [part.strip() for part in text.split(separator)]
            return [part for part in parts if part]
    normalized = _normalize_text(text)
    if "menu" not in normalized:
        return []
    tokens = []
    for token in ("file", "edit", "view", "insert", "format", "tools", "help"):
        if token in normalized:
            tokens.append(token)
            break
    if tokens:
        return tokens
    return []


def _menu_access_key(token: str) -> str | None:
    token = token.strip()
    if not token:
        return None
    if "&" in token:
        amp = token.index("&")
        if amp + 1 < len(token) and token[amp + 1].isalnum():
            return token[amp + 1].lower()
    for char in token:
        if char.isalnum():
            return char.lower()
    return None


def _ensure_browser(agent_context: AgentContext) -> PCSession:
    session = agent_context.pc_state.get("browser_session")
    if session:
        return session

    playwright = sync_playwright().start()
    user_data_dir = str(agent_context.pc_state.get("browser_user_data_dir") or "").strip()
    executable_path = str(agent_context.pc_state.get(
        "browser_executable_path") or "").strip() or None
    if user_data_dir:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir,
            headless=False,
            executable_path=executable_path,
        )
        page = context.pages[0] if context.pages else context.new_page()
        session = PCSession(playwright=playwright, browser=context, page=page)
    else:
        browser = playwright.chromium.launch(
            headless=False,
            executable_path=executable_path,
        )
        page = browser.new_page()
        session = PCSession(playwright=playwright, browser=browser, page=page)
    agent_context.pc_state["browser_session"] = session
    return session


def _click_text(
    agent_context: AgentContext,
    text: str,
    llm_client: Any = None,
    timeout_seconds: float = 10,
    instruction: str | None = None,
    prefer_web: bool | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    session = agent_context.pc_state.get("browser_session")
    if _should_use_web_dom(agent_context, prefer_web):
        if not session:
            session = _ensure_browser(agent_context)
        remaining = max(0.5, deadline - time.monotonic())
        try:
            session.page.get_by_text(text).first.click(
                timeout=int(remaining * 1000))
            return
        except Exception:
            pass

    while time.monotonic() < deadline:

        element = find_ui_element(
            text, llm_client=llm_client, use_vision=False)
        if element and element.get("handle") is not None:
            element["handle"].click_input()
            return

        time.sleep(0.5)

    if _maybe_use_alt_menu_navigation(text, instruction, agent_context):
        return

    if llm_client:
        if "search" in text.lower():
            try:
                agent_context.pc_state["force_search_focus"] = True
                logger.info(
                    "Trying Ctrl+L to focus search before vision fallback.")
                pyautogui.hotkey("ctrl", "l")
                time.sleep(0.2)
            except Exception:
                pass
        element = find_ui_element(text, llm_client=llm_client, use_vision=True)
        if element and element.get("center_point"):
            x, y = element["center_point"]
            pyautogui.click(x, y)
            return
        if instruction:
            point = locate_ui_point(str(instruction), llm_client)
            if point:
                pyautogui.click(point[0], point[1])
                return

    raise ExecutionError(f"Unable to click text: {text}")


def _use_system_browser(agent_context: AgentContext) -> bool:
    return bool(agent_context.pc_state.get("use_system_browser"))


def _focus_browser_window(agent_context: AgentContext) -> None:
    title = agent_context.pc_state.get("browser_window_title")
    if not title:
        return
    try:
        window = Desktop(backend="uia").window(title_re=str(title))
        if window.exists(timeout=0.5):
            window.set_focus()
            time.sleep(0.1)
    except Exception:
        return


def _is_browser_active_window(agent_context: AgentContext) -> bool:
    try:
        window = Desktop(backend="uia").get_active()
        title = window.window_text() or ""
        title_pattern = str(agent_context.pc_state.get("browser_window_title") or "")
        if title_pattern and re.search(title_pattern, title, re.IGNORECASE):
            return True
        lowered = title.lower()
        return any(token in lowered for token in ("chrome", "edge", "firefox", "brave", "opera"))
    except Exception:
        return False


def _navigate_system_browser(url: str, agent_context: AgentContext) -> None:
    _focus_browser_window(agent_context)
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.1)
    pyautogui.typewrite(url, interval=0.02)
    pyautogui.press("enter")


def _is_spotify_active_window(agent_context: AgentContext) -> bool:
    if agent_context.pc_state.get("active_app_hint") == "spotify":
        return True
    try:
        window = Desktop(backend="uia").get_active()
        title = (window.window_text() or "").lower()
        if "spotify" in title:
            return True
        process_id = window.element_info.process_id
        try:
            import psutil
        except Exception:
            return False
        process_name = psutil.Process(process_id).name().lower()
        return "spotify" in process_name
    except Exception:
        return False


def _is_whatsapp_active_window(agent_context: AgentContext) -> bool:
    if agent_context.pc_state.get("active_app_hint") == "whatsapp":
        return True
    try:
        window = Desktop(backend="uia").get_active()
        title = (window.window_text() or "").lower()
        if "whatsapp" in title:
            return True
        process_id = window.element_info.process_id
        try:
            import psutil
        except Exception:
            return False
        process_name = psutil.Process(process_id).name().lower()
        return "whatsapp" in process_name
    except Exception:
        return False


def _focus_spotify_window() -> None:
    try:
        window = Desktop(backend="uia").window(title_re=".*Spotify.*")
        if window.exists(timeout=0.5):
            window.set_focus()
            return
    except Exception:
        pass
    try:
        import psutil
    except Exception:
        return


def _positions_close(a: tuple[int, int], b: tuple[int, int], threshold: int = 40) -> bool:
    return abs(a[0] - b[0]) <= threshold and abs(a[1] - b[1]) <= threshold


def _hover_spotify_top_result_card(llm_client: Any) -> None:
    if not llm_client:
        return
    top_anchor = find_ui_element(
        "En çok dinlenen sonuç", llm_client=llm_client, use_vision=False
    )
    if not top_anchor:
        top_anchor = find_ui_element(
            "Top result", llm_client=llm_client, use_vision=False)
    if top_anchor and top_anchor.get("rectangle"):
        rect = top_anchor["rectangle"]
        hover_x = rect["left"] + max(100, (rect["right"] - rect["left"]) // 2)
        hover_y = rect["bottom"] + 140
        pyautogui.moveTo(hover_x, hover_y, duration=0.2)
        time.sleep(0.4)


def _hover_spotify_first_song_row(llm_client: Any) -> None:
    if not llm_client:
        return
    songs_anchor = find_ui_element(
        "Şarkılar", llm_client=llm_client, use_vision=False)
    if not songs_anchor:
        songs_anchor = find_ui_element(
            "Songs", llm_client=llm_client, use_vision=False)
    if songs_anchor and songs_anchor.get("rectangle"):
        rect = songs_anchor["rectangle"]
        hover_x = rect["left"] + max(140, (rect["right"] - rect["left"]) // 2)
        hover_y = rect["bottom"] + 70
        pyautogui.moveTo(hover_x, hover_y, duration=0.2)
        time.sleep(0.4)
    try:
        for window in Desktop(backend="uia").windows():
            try:
                process_id = window.element_info.process_id
            except Exception:
                continue
            try:
                process_name = psutil.Process(process_id).name().lower()
            except Exception:
                continue
            if "spotify" in process_name:
                try:
                    window.set_focus()
                except Exception:
                    pass
                return
    except Exception:
        return


def _is_play_first_result_request(text: str, description: str) -> bool:
    text_value = _normalize_text(str(text or "")).strip()
    description_value = _normalize_text(description)
    if "play first search result" in description_value:
        return True
    if "ilk sonuc" in description_value or "ilk sarki" in description_value:
        return True
    if "sarki" in description_value and ("cal" in description_value or "oynat" in description_value or "baslat" in description_value):
        return True
    if "select" in description_value and (
        "song" in description_value or "track" in description_value or "result" in description_value
    ):
        return True
    return text_value in {
        "play first search result",
        "play first result",
        "first search result",
        "first result",
        "ilk sonuc",
        "ilk sarki",
        "sarki cal",
        "sarki oynat",
        "sarki baslat",
    }


def _is_song_selection_request(
    agent_context: AgentContext, text: str, description: str
) -> bool:
    description_value = _normalize_text(description)
    if "select" in description_value and (
        "song" in description_value or "track" in description_value or "result" in description_value
    ):
        return True
    if "sarki" in description_value and (
        "sec" in description_value or "sonuc" in description_value
    ):
        return True
    last_typed = _normalize_text(
        str(agent_context.pc_state.get("last_typed_text", "")))
    pending_query = _normalize_text(
        str(agent_context.pc_state.get("pending_search_query", "")))
    text_value = _normalize_text(str(text or "")).strip()
    if text_value and last_typed and (text_value in last_typed or last_typed in text_value):
        return True
    if text_value and pending_query and (text_value in pending_query or pending_query in text_value):
        return True
    return False


def _click_spotify_first_result(llm_client: Any, timeout_seconds: float = 6) -> bool:
    _focus_spotify_window()
    if _try_spotify_top_result_play(llm_client):
        return True
    if _try_spotify_row_play(llm_client):
        return True
    deadline = time.monotonic() + timeout_seconds
    anchors = [
        "Songs",
        "Top result",
        "Top results",
        "Tracks",
        "Şarkılar",
        "En çok dinlenen sonuç",
    ]
    while time.monotonic() < deadline:
        for anchor in anchors:
            element = find_ui_element(
                anchor, llm_client=llm_client, use_vision=False)
            if element and element.get("rectangle"):
                rect = element["rectangle"]
                x = rect["left"] + max(20, (rect["right"] - rect["left"]) // 4)
                y = rect["bottom"] + 30
                if _ensure_spotify_playback(llm_client, (x, y)):
                    return True
        time.sleep(0.3)

    if llm_client:
        element = find_spotify_first_result(llm_client)
        if element and element.get("center_point"):
            x, y = element["center_point"]
            _, screen_height = pyautogui.size()
            if y < int(screen_height * 0.15):
                logger.info(
                    "Spotify result click ignored due to top-of-screen coordinate: (%s, %s)",
                    x,
                    y,
                )
                return False
            if _ensure_spotify_playback(llm_client, (x, y)):
                return True
    return False


def _try_spotify_top_result_play(llm_client: Any) -> bool:
    if not llm_client:
        return False
    _hover_spotify_top_result_card(llm_client)
    before = find_spotify_top_result_play_button(llm_client)
    if not before or not before.get("center_point"):
        return False
    x, y = before["center_point"]
    pyautogui.click(x, y)
    time.sleep(0.6)
    _hover_spotify_top_result_card(llm_client)
    after = find_spotify_top_result_play_button(llm_client)
    if not after or not after.get("center_point"):
        return True
    return not _positions_close(before["center_point"], after["center_point"])


def _try_spotify_row_play(llm_client: Any) -> bool:
    if not llm_client:
        return False
    _hover_spotify_first_song_row(llm_client)
    before = find_spotify_row_play_button(llm_client)
    if not before or not before.get("center_point"):
        return False
    x, y = before["center_point"]
    pyautogui.click(x, y)
    time.sleep(0.6)
    _hover_spotify_first_song_row(llm_client)
    after = find_spotify_row_play_button(llm_client)
    if not after or not after.get("center_point"):
        return True
    return not _positions_close(before["center_point"], after["center_point"])


def _ensure_spotify_playback(llm_client: Any, click_point: tuple[int, int] | None) -> bool:
    _focus_spotify_window()
    if _try_spotify_top_result_play(llm_client):
        return True
    if _try_spotify_row_play(llm_client):
        return True
    if click_point:
        pyautogui.click(click_point[0], click_point[1])
        time.sleep(0.4)
        pyautogui.doubleClick(click_point[0], click_point[1])
        time.sleep(0.5)
        if _try_spotify_top_result_play(llm_client):
            return True
        if _try_spotify_row_play(llm_client):
            return True
    return False


def _is_spotify_playing(llm_client: Any) -> bool:
    play_element = find_ui_element(
        "Play", llm_client=llm_client, use_vision=False)
    if play_element:
        return False
    play_element = find_ui_element(
        "Çal", llm_client=llm_client, use_vision=False)
    if play_element:
        return False
    if llm_client:
        vision_play = find_spotify_player_play_button(llm_client)
        if vision_play and vision_play.get("center_point"):
            return False
    return True


def _normalize_text(value: str) -> str:
    normalized = value.lower()
    return normalized.translate(
        {
            ord("\u00e7"): "c",
            ord("\u011f"): "g",
            ord("\u0131"): "i",
            ord("\u00f6"): "o",
            ord("\u015f"): "s",
            ord("\u00fc"): "u",
        }
    )


def _is_search_step(description: str | None, instruction: str | None) -> bool:
    combined = f"{description or ''} {instruction or ''}".strip()
    normalized = _normalize_text(combined)
    return any(token in normalized for token in ("search", "ara", "bul", "find"))


def _should_press_enter_for_send(description: str | None, instruction: str | None) -> bool:
    combined = f"{description or ''} {instruction or ''}".strip()
    normalized = _normalize_text(combined)
    return any(token in normalized for token in ("send", "gonder", "yolla", "ileti gonder"))


def _is_message_step(description: str | None, instruction: str | None) -> bool:
    combined = f"{description or ''} {instruction or ''}".strip()
    normalized = _normalize_text(combined)
    return any(token in normalized for token in ("message", "mesaj", "input", "yazi", "type and send"))


def _is_chat_selection_step(description: str | None, instruction: str | None) -> bool:
    combined = f"{description or ''} {instruction or ''}".strip()
    normalized = _normalize_text(combined)
    return any(
        token in normalized
        for token in ("select the chat", "chat result", "contact from results", "select contact")
    )


def _click_whatsapp_search_result(query: str, agent_context: AgentContext, llm_client: Any) -> None:
    if not query:
        raise ExecutionError("Search query is empty for WhatsApp selection.")
    _focus_whatsapp_window()
    if _is_whatsapp_message_ready(llm_client):
        return
    if _try_select_whatsapp_chat_uia(query):
        return
    region = _whatsapp_chat_list_region()
    instruction = (
        f"click the chat result row for '{query}' in WhatsApp search results "
        "within the chats list (left panel), below the search bar and filters"
    )
    point = locate_text_center(
        query,
        llm_client,
        context=(
            "WhatsApp desktop. Use the left chats list. Search results appear below the search bar. "
            "Click the contact name text itself (e.g., 'Bebiss') inside the result row. "
            "The visible text can include extra emoji or symbols; match if it contains the query. "
            "Ignore the left navigation sidebar icons and top bar. "
            "Do not click the avatar, timestamp, filters, or phone/calls icons."
            + _format_whatsapp_region_context(region)
        ),
    )
    if not point:
        point = locate_ui_point(instruction, llm_client)
    if point and _is_valid_whatsapp_chat_point(point):
        click_point = _whatsapp_row_click_point(point, region)
        pyautogui.click(click_point[0], click_point[1])
        time.sleep(0.2)
        if _is_whatsapp_message_ready(llm_client):
            return
        pyautogui.press("enter")
        time.sleep(0.3)
        if _is_whatsapp_message_ready(llm_client):
            return
        pyautogui.press("down")
        time.sleep(0.15)
        pyautogui.press("enter")
        time.sleep(0.3)
        if _is_whatsapp_message_ready(llm_client):
            return
        return
    _click_text(
        agent_context,
        query,
        llm_client,
        timeout_seconds=6,
        instruction=instruction,
    )


def _focus_whatsapp_search(agent_context: AgentContext, llm_client: Any) -> None:
    if not llm_client:
        raise ExecutionError("WhatsApp search focus requires llm_client")
    try:
        pyautogui.press("esc")
        time.sleep(0.15)
        pyautogui.press("esc")
        time.sleep(0.15)
    except Exception:
        pass
    if _try_focus_whatsapp_search_uia():
        agent_context.pc_state["search_context"] = True
        return
    if _click_whatsapp_search_fallback():
        agent_context.pc_state["search_context"] = True
        return
    instruction = (
        "click the WhatsApp search input at the top of the chats list "
        "(placeholder text like 'Ara' or 'Search'), not the gear/settings icon"
    )
    point = locate_ui_point(instruction, llm_client)
    if point and _is_valid_whatsapp_search_point(point):
        pyautogui.click(point[0], point[1])
        time.sleep(0.2)
        agent_context.pc_state["search_context"] = True
        return

    _try_open_whatsapp_chats(llm_client)
    point = locate_ui_point(instruction, llm_client)
    if point and _is_valid_whatsapp_search_point(point):
        pyautogui.click(point[0], point[1])
        time.sleep(0.2)
        agent_context.pc_state["search_context"] = True
        return

    raise ExecutionError("Unable to locate WhatsApp search input via vision.")


def _is_valid_whatsapp_search_point(point: tuple[int, int]) -> bool:
    x, y = point
    rect = _get_whatsapp_window_rect()
    if rect:
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if x < rect.left + int(width * 0.06):
            return False
        if y > rect.top + int(height * 0.35):
            return False
        return True
    width, height = pyautogui.size()
    if x < int(width * 0.06):
        return False
    if y > int(height * 0.35):
        return False
    return True


def _is_valid_whatsapp_chat_point(point: tuple[int, int]) -> bool:
    region = _whatsapp_chat_list_region()
    if region:
        left, top, right, bottom = region
        x, y = point
        return left <= x <= right and top <= y <= bottom
    x, y = point
    rect = _get_whatsapp_window_rect()
    if rect:
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if x < rect.left + int(width * 0.12):
            return False
        if x > rect.left + int(width * 0.55):
            return False
        if y < rect.top + int(height * 0.2):
            return False
        if y > rect.top + int(height * 0.92):
            return False
        return True
    width, height = pyautogui.size()
    if x < int(width * 0.12):
        return False
    if x > int(width * 0.55):
        return False
    if y < int(height * 0.2):
        return False
    if y > int(height * 0.92):
        return False
    return True


def _try_open_whatsapp_chats(llm_client: Any) -> None:
    if not llm_client:
        return
    instruction = "click the Chats tab (speech bubble icon) in the left sidebar"
    point = locate_ui_point(instruction, llm_client)
    if not point:
        return
    pyautogui.click(point[0], point[1])
    time.sleep(0.3)


def _try_select_whatsapp_chat_uia(query: str) -> bool:
    if not query:
        return False
    window = _get_whatsapp_window()
    if not window:
        return False
    region = _whatsapp_chat_list_region()
    try:
        title_pattern = rf"(?i).*{re.escape(query)}.*"
        match = window.child_window(title_re=title_pattern, control_type="Text")
        if not match.exists(timeout=0.2):
            return False
        rect = match.rectangle()
        if region:
            left, top, right, bottom = region
            if rect.right < left or rect.left > right or rect.bottom < top or rect.top > bottom:
                return False
        click_point = _whatsapp_row_click_point(_rect_center(rect), region)
        if not _is_valid_whatsapp_chat_point(click_point):
            return False
        try:
            match.click_input()
        except Exception:
            pyautogui.click(click_point[0], click_point[1])
        time.sleep(0.4)
        return _try_focus_whatsapp_message_uia()
    except Exception:
        return False


def _select_whatsapp_chat_result(query: str, agent_context: AgentContext, llm_client: Any) -> None:
    if not llm_client:
        raise ExecutionError("WhatsApp chat selection requires llm_client")
    _focus_whatsapp_window()
    if _is_whatsapp_message_ready(llm_client):
        return
    if _try_select_whatsapp_chat_uia(query):
        return

    region = _whatsapp_chat_list_region()

    # 1. Try Gemini Vision with optimized prompt for WhatsApp contact rows
    point = locate_whatsapp_search_result(query, llm_client)
    if point and not _is_valid_whatsapp_chat_point(point):
        point = None

    if not point:
        # Fallback to older methods if specific locator fails
        instruction = (
            f"click the chat result row for '{query}' in WhatsApp search results "
            "within the chats list (left panel), below the search bar and filters"
        )
        point = locate_text_center(
            query,
            llm_client,
            context=(
                "WhatsApp desktop. Use the left chats list. Search results appear below the search bar. "
                "Click the contact name text itself (e.g., 'Bebiss') inside the result row. "
                "The visible text can include extra emoji or symbols; match if it contains the query. "
                "Ignore the left navigation sidebar icons and top bar. "
                "Do not click the avatar, timestamp, filters, or phone/calls icons."
                + _format_whatsapp_region_context(region)
            ),
        )
        if not point:
            point = locate_ui_point(instruction, llm_client)

        if point and _is_valid_whatsapp_chat_point(point):
            point = _whatsapp_row_click_point(point, region)

    if point and _is_valid_whatsapp_chat_point(point):
        click_point = _whatsapp_row_click_point(point, region)
        logger.info("Clicking WhatsApp chat result at %s", click_point)
        pyautogui.click(click_point[0], click_point[1])
        time.sleep(0.5)

        # Validation and Retry logic
        for _ in range(3):
            if _is_whatsapp_message_ready(llm_client):
                return

            # Try Enter key
            pyautogui.press("enter")
            time.sleep(0.5)
            if _is_whatsapp_message_ready(llm_client):
                return

            # Try Down + Enter
            pyautogui.press("down")
            time.sleep(0.2)
            pyautogui.press("enter")
            time.sleep(0.5)

    # Final fallback if still not open
    if not _is_whatsapp_message_ready(llm_client):
        # Last resort: Try text click fallback
        _click_text(
            agent_context,
            query,
            llm_client,
            timeout_seconds=6,
            instruction=f"click the contact '{query}' in the left chat list",
        )
    if not _is_whatsapp_message_ready(llm_client):
        raise ExecutionError(f"Unable to open WhatsApp chat for '{query}'.")


def _is_whatsapp_message_ready(llm_client: Any) -> bool:
    if _try_focus_whatsapp_message_uia():
        return True
    if not llm_client:
        return False
    instruction = (
        "click the message input box in the chat area at the bottom "
        "(where you type messages), not the search bar"
    )
    point = locate_ui_point(instruction, llm_client)
    if point and _is_valid_whatsapp_message_point(point):
        pyautogui.click(point[0], point[1])
        time.sleep(0.15)
        return True
    return False


def _focus_whatsapp_message_input(llm_client: Any) -> None:
    if not llm_client:
        return
    _focus_whatsapp_window()
    if _try_focus_whatsapp_message_uia():
        return
    instruction = (
        "click the message input box in the chat area at the bottom "
        "(where you type messages), not the search bar"
    )
    point = locate_ui_point(instruction, llm_client)
    if point and _is_valid_whatsapp_message_point(point):
        pyautogui.click(point[0], point[1])
        time.sleep(0.2)
        return
    if _click_whatsapp_message_fallback():
        return
    raise ExecutionError("Unable to locate WhatsApp message input via vision.")


def _try_focus_whatsapp_message_uia() -> bool:
    try:
        window = Desktop(backend="uia").window(title_re=".*WhatsApp.*")
        if not window.exists(timeout=0.5):
            return False
        rect = window.rectangle()
        min_x = rect.left + int((rect.right - rect.left) * 0.4)
        min_y = rect.top + int((rect.bottom - rect.top) * 0.6)
        candidates = window.descendants(control_type="Edit")
        for control in candidates:
            ctrl_rect = control.rectangle()
            if ctrl_rect.left < min_x or ctrl_rect.top < min_y:
                continue
            control.click_input()
            return True
    except Exception:
        return False
    return False


def _is_valid_whatsapp_message_point(point: tuple[int, int]) -> bool:
    x, y = point
    rect = _get_whatsapp_window_rect()
    if rect:
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        min_x = rect.left + int(width * 0.4)
        region = _whatsapp_chat_list_region()
        if region:
            min_x = max(min_x, region[2] + 8)
        if x < min_x:
            return False
        if y < rect.top + int(height * 0.65):
            return False
        return True
    width, height = pyautogui.size()
    if x < int(width * 0.35):
        return False
    if y < int(height * 0.65):
        return False
    return True


def _click_whatsapp_message_fallback() -> bool:
    try:
        window = Desktop(backend="uia").window(title_re=".*WhatsApp.*")
        if not window.exists(timeout=0.5):
            return False
        rect = window.rectangle()
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        x = rect.left + int(width * 0.65)
        y = rect.top + int(height * 0.9)
        pyautogui.click(x, y)
        time.sleep(0.2)
        return True
    except Exception:
        return False


def _focus_whatsapp_window() -> None:
    try:
        window = Desktop(backend="uia").window(title_re=".*WhatsApp.*")
        if window.exists(timeout=0.5):
            window.set_focus()
            time.sleep(0.1)
            return
    except Exception:
        pass
    try:
        rect = _get_whatsapp_window_rect()
        if rect:
            pyautogui.click(rect.left + 40, rect.top + 40)
        else:
            pyautogui.click(200, 120)
        time.sleep(0.1)
    except Exception:
        pass


def _replace_input_text(value: str) -> None:
    try:
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.05)
    except Exception:
        pass
    if _paste_text(value):
        return
    pyautogui.typewrite(str(value), interval=0.02)


def _paste_text(value: str) -> bool:
    try:
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(str(value))
        root.update()
        root.destroy()
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.05)
        return True
    except Exception:
        return False


def _try_focus_whatsapp_search_uia() -> bool:
    try:
        window = Desktop(backend="uia").window(title_re=".*WhatsApp.*")
        if not window.exists(timeout=0.5):
            return False
        rect = window.rectangle()
        max_x = rect.left + int((rect.right - rect.left) * 0.5)
        max_y = rect.top + int((rect.bottom - rect.top) * 0.3)
        candidates = window.descendants(control_type="Edit")
        for control in candidates:
            name = (control.window_text() or "").strip().lower()
            ctrl_rect = control.rectangle()
            if ctrl_rect.left > max_x or ctrl_rect.top > max_y:
                continue
            if name and ("ara" in name or "search" in name):
                control.click_input()
                return True
        for control in candidates:
            ctrl_rect = control.rectangle()
            if ctrl_rect.left <= max_x and ctrl_rect.top <= max_y:
                control.click_input()
                return True
    except Exception:
        return False
    return False


def _click_whatsapp_search_fallback() -> bool:
    try:
        window = Desktop(backend="uia").window(title_re=".*WhatsApp.*")
        if not window.exists(timeout=0.5):
            return False
        rect = window.rectangle()
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        x = rect.left + int(width * 0.22)
        y = rect.top + int(height * 0.13)
        pyautogui.click(x, y)
        time.sleep(0.2)
        return True
    except Exception:
        return False


def _get_whatsapp_window_rect():
    try:
        window = _get_whatsapp_window()
        if window:
            rect = window.rectangle()
            if rect.right > rect.left and rect.bottom > rect.top:
                return rect
    except Exception:
        return None
    return None


def _whatsapp_chat_list_region() -> tuple[int, int, int, int] | None:
    rect = _get_whatsapp_window_rect()
    if not rect:
        return None
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    left = rect.left + int(width * 0.14)
    right = rect.left + int(width * 0.46)
    top = rect.top + int(height * 0.22)
    bottom = rect.top + int(height * 0.92)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _format_whatsapp_region_context(region: tuple[int, int, int, int] | None) -> str:
    if not region:
        return ""
    left, top, right, bottom = region
    return (
        f" Only click within the chats list rectangle "
        f"(left={left}, top={top}, right={right}, bottom={bottom})."
    )


def _whatsapp_row_click_point(
    point: tuple[int, int],
    region: tuple[int, int, int, int] | None,
) -> tuple[int, int]:
    if not region:
        return point
    left, top, right, bottom = region
    x = left + int((right - left) * 0.6)
    y = min(max(point[1], top + 4), bottom - 4)
    return x, y


def _rect_center(rect) -> tuple[int, int]:
    return int((rect.left + rect.right) / 2), int((rect.top + rect.bottom) / 2)


def _get_whatsapp_window():
    try:
        window = Desktop(backend="uia").get_active()
        if window and "whatsapp" in (window.window_text() or "").lower():
            return window
    except Exception:
        pass
    try:
        window = Desktop(backend="uia").window(title_re=".*WhatsApp.*")
        if window.exists(timeout=0.5):
            return window
    except Exception:
        return None
    return None


def _is_whatsapp_chat_open(query: str) -> bool:
    if not query:
        return False
    rect = _get_whatsapp_window_rect()
    if not rect:
        return False
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    min_x = rect.left + int(width * 0.45)
    max_y = rect.top + int(height * 0.35)
    normalized_query = _normalize_text(query)
    try:
        window = _get_whatsapp_window()
        if not window:
            return False
        for control in window.descendants(control_type="Text"):
            name = (control.window_text() or "").strip()
            if not name:
                continue
            if normalized_query not in _normalize_text(name):
                continue
            ctrl_rect = control.rectangle()
            if ctrl_rect.left < min_x or ctrl_rect.top > max_y:
                continue
            return True
    except Exception:
        return False
    return False


def _open_app_from_start_menu(
    app_name: str,
    window_title: str | None = None,
    timeout_seconds: float = 45,
) -> None:
    pyautogui.press("win")
    time.sleep(0.4)
    pyautogui.typewrite(app_name, interval=0.04)
    time.sleep(0.2)
    pyautogui.press("enter")

    title_pattern = window_title or f".*{re.escape(app_name)}.*"
    if not _wait_for_window(title_pattern, timeout_seconds=timeout_seconds, focus=True):
        raise ExecutionError(f"App window not found for: {app_name}")


def _wait_for_window(title_regex: str, timeout_seconds: float = 30, focus: bool = True) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            window = Desktop(backend="uia").window(title_re=title_regex)
            if window.exists(timeout=0.2):
                if focus:
                    try:
                        window.set_focus()
                    except Exception:
                        pass
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False
