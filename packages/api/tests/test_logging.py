"""App logs reach both the rotating file and the console."""

import logging

from cell_explorer_api.config import Settings
from cell_explorer_api.main import CONSOLE_HANDLER_NAME, create_app


def _console_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if h.name == CONSOLE_HANDLER_NAME]


def test_console_handler_added_once(tmp_path):
    create_app(Settings(app_data_dir=tmp_path / "a", database_url="sqlite+aiosqlite://"))
    create_app(Settings(app_data_dir=tmp_path / "b", database_url="sqlite+aiosqlite://"))

    handlers = _console_handlers()
    assert len(handlers) == 1, "a console handler per app would print every line repeatedly"
    assert type(handlers[0]) is logging.StreamHandler


def test_app_logs_still_written_to_file(tmp_path):
    settings = Settings(app_data_dir=tmp_path, database_url="sqlite+aiosqlite://")
    create_app(settings)

    logging.getLogger("cell_explorer_api.test").warning("hello from the test")
    for h in logging.getLogger().handlers:
        h.flush()

    assert "hello from the test" in (settings.log_dir / settings.log_filename).read_text()
