from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

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
