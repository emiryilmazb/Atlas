from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv() -> None:
        return None


@dataclass(frozen=True)
class Settings:
    app_name: str = "ApplyWise"
    log_level: str = "INFO"
    sqlite_db_path: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    gemini_image_model: str = "gemini-3-pro-image-preview"
    gemini_enable_google_search: bool = True
    gemini_enable_code_execution: bool = True
    streaming_enabled: bool = True
    show_thoughts: bool = True
    screenshot_enabled: bool = True
    pc_use_system_browser: bool = True
    pc_browser_name: str = "Chrome"
    pc_browser_window_title: str = ".*Chrome.*"
    pc_browser_user_data_dir: str = ""
    pc_browser_executable_path: str = ""
    kariyer_job_url: str = ""
    kariyernet_username: str = ""
    kariyernet_password: str = ""


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        app_name=os.getenv("APPLYWISE_APP_NAME", "ApplyWise"),
        log_level=os.getenv("APPLYWISE_LOG_LEVEL", "INFO"),
        sqlite_db_path=os.getenv("SQLITE_DB_PATH", ""),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        gemini_image_model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview"),
        gemini_enable_google_search=_env_flag("GEMINI_ENABLE_GOOGLE_SEARCH", True),
        gemini_enable_code_execution=_env_flag("GEMINI_ENABLE_CODE_EXECUTION", True),
        streaming_enabled=_env_flag("STREAMING_ENABLED", True),
        show_thoughts=_env_flag("SHOW_THOUGHTS", True),
        screenshot_enabled=_env_flag("SCREENSHOT_ENABLED", True),
        pc_use_system_browser=_env_flag("PC_USE_SYSTEM_BROWSER", True),
        pc_browser_name=os.getenv("PC_BROWSER_NAME", "Chrome"),
        pc_browser_window_title=os.getenv("PC_BROWSER_WINDOW_TITLE", ".*Chrome.*"),
        pc_browser_user_data_dir=os.getenv("PC_BROWSER_USER_DATA_DIR", ""),
        pc_browser_executable_path=os.getenv("PC_BROWSER_EXECUTABLE_PATH", ""),
        kariyer_job_url=os.getenv("KARIYER_JOB_URL", ""),
        kariyernet_username=os.getenv("KARIYERNET_USERNAME", ""),
        kariyernet_password=os.getenv("KARIYERNET_PASSWORD", ""),
    )
