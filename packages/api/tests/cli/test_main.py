from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from cell_explorer_api.cli.config import AuthConfig, auth_config_path, save_auth_config
from cell_explorer_api.cli.main import app


def test_app_help_lists_all_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Only 'login' exists after this task; more commands come in later tasks.
    assert "login" in result.stdout


def test_login_happy_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CELL_EXPLORER_API_URL", "http://localhost:8000")

    with patch("cell_explorer_api.cli.main.start_callback_server") as mock_start, \
         patch("cell_explorer_api.cli.main.webbrowser") as mock_browser, \
         patch("cell_explorer_api.cli.main._decode_username") as mock_decode:
        mock_start.return_value = (
            53412,
            lambda: {
                "access_token": "ACC",
                "refresh_token": "REF",
                "expires_in": 300,
            },
        )
        mock_decode.return_value = "jason@example.com"

        runner = CliRunner()
        result = runner.invoke(app, ["login"])

    assert result.exit_code == 0, result.stdout
    assert "Logged in as jason@example.com" in result.stdout
    mock_browser.open.assert_called_once()


def test_login_timeout_reports_clearly(monkeypatch, tmp_path):
    from cell_explorer_api.cli.callback_server import CallbackTimeout

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CELL_EXPLORER_API_URL", "http://localhost:8000")

    def _timeout_wait():
        raise CallbackTimeout("no callback in 5s")

    with patch("cell_explorer_api.cli.main.start_callback_server") as mock_start, \
         patch("cell_explorer_api.cli.main.webbrowser"):
        mock_start.return_value = (53412, _timeout_wait)

        runner = CliRunner()
        result = runner.invoke(app, ["login"])

    assert result.exit_code != 0
    assert "timed out" in result.stdout.lower() or "no callback" in result.stdout.lower()


def test_logout_when_logged_in(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = AuthConfig(
        api_url="http://localhost:8000",
        access_token="A", refresh_token="R",
        access_expires_at=datetime.now(timezone.utc),
        refresh_expires_at=datetime.now(timezone.utc),
        username="x",
    )
    save_auth_config(cfg)

    runner = CliRunner()
    result = runner.invoke(app, ["logout"])

    assert result.exit_code == 0
    assert "logged out" in result.stdout.lower()
    assert not auth_config_path().exists()


def test_logout_when_not_logged_in(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["logout"])

    assert result.exit_code == 0
    assert not auth_config_path().exists()


def test_datasets_lists_accessible(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = AuthConfig(
        api_url="http://localhost:8000",
        access_token="A", refresh_token="R",
        access_expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        refresh_expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        username="researcher@example.com",
    )
    save_auth_config(cfg)

    # Patch the in-process dataset loader to return a known list.
    fake_rows = [
        {"slug": "pbmc3k", "name": "PBMC 3k", "is_public": True,
         "required_roles": []},
        {"slug": "brca", "name": "BRCA", "is_public": False,
         "required_roles": ["researcher"]},
    ]
    with patch("cell_explorer_api.cli.main._list_datasets_sync") as mock_list, \
         patch("cell_explorer_api.cli.main._load_user_sync") as mock_user:
        mock_list.return_value = fake_rows
        mock_user.return_value = MagicMock(roles=["researcher"], username="researcher@example.com")

        runner = CliRunner()
        result = runner.invoke(app, ["datasets"])

    assert result.exit_code == 0, result.stdout
    assert "pbmc3k" in result.stdout
    assert "brca" in result.stdout


def test_datasets_not_logged_in(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    from cell_explorer_api.cli.errors import NotLoggedInError

    with patch("cell_explorer_api.cli.main._load_user_sync") as mock_user:
        mock_user.side_effect = NotLoggedInError("no auth.json")

        runner = CliRunner()
        result = runner.invoke(app, ["datasets"])

    assert result.exit_code == 6
    assert "not logged in" in result.stdout.lower()


def test_ask_streams_text_and_exits_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = AuthConfig(
        api_url="http://localhost:8000",
        access_token="A", refresh_token="R",
        access_expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        refresh_expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        username="u",
    )
    save_auth_config(cfg)

    fake_agent = MagicMock()
    async def _run(**_):
        from cell_explorer_agent.events import Done, TextDelta, TokenUsage
        yield TextDelta(text="the dataset has 100 cells")
        yield Done(usage=TokenUsage(input_tokens=10, output_tokens=5))
    fake_agent.run = _run

    with patch("cell_explorer_api.cli.main._load_user_sync") as mock_user, \
         patch("cell_explorer_api.cli.main._build_agent_sync") as mock_build:
        mock_user.return_value = MagicMock(roles=["researcher"])
        mock_build.return_value = fake_agent

        runner = CliRunner()
        result = runner.invoke(app, ["ask", "pbmc3k", "how big?"])

    assert result.exit_code == 0, result.stdout
    assert "100 cells" in result.stdout


def test_ask_dataset_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = AuthConfig(
        api_url="http://localhost:8000",
        access_token="A", refresh_token="R",
        access_expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        refresh_expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        username="u",
    )
    save_auth_config(cfg)

    from cell_explorer_api.services.chat_session import DatasetNotFoundError

    with patch("cell_explorer_api.cli.main._load_user_sync") as mock_user, \
         patch("cell_explorer_api.cli.main._build_agent_sync") as mock_build:
        mock_user.return_value = MagicMock(roles=[])
        mock_build.side_effect = DatasetNotFoundError("dataset 'nope' not found")

        runner = CliRunner()
        result = runner.invoke(app, ["ask", "nope", "hi"])

    assert result.exit_code == 2
    assert "nope" in result.stdout.lower()
