import pytest

from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.telemetry import langfuse_client


@pytest.fixture(autouse=True)
def reset_singleton():
    langfuse_client._reset_for_tests()
    yield
    langfuse_client._reset_for_tests()


def test_get_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    cfg = AgentConfig()
    assert langfuse_client.get(cfg) is None


def test_get_returns_client_when_configured(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    cfg = AgentConfig()
    client = langfuse_client.get(cfg)
    assert client is not None


def test_get_caches_instance(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    cfg = AgentConfig()
    assert langfuse_client.get(cfg) is langfuse_client.get(cfg)


def test_get_returns_none_when_kill_switch_off(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_TRACE_ENABLED", "false")
    cfg = AgentConfig()
    assert langfuse_client.get(cfg) is None


def test_release_kwarg_passed_when_git_sha_env_set(monkeypatch):
    """GIT_SHA env value flows into the Langfuse() constructor as `release`."""
    captured: dict = {}

    class _RecordingLangfuse:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langfuse as langfuse_mod  # noqa: PLC0415

    monkeypatch.setattr(langfuse_mod, "Langfuse", _RecordingLangfuse)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("GIT_SHA", "abc1234")

    cfg = AgentConfig()
    langfuse_client.get(cfg)

    assert captured.get("release") == "abc1234"


def test_release_kwarg_omitted_when_no_git_sha(monkeypatch):
    """When GIT_SHA is unset and not in a git repo, `release` is not passed
    so the SDK uses its own default."""
    captured: dict = {}

    class _RecordingLangfuse:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langfuse as langfuse_mod  # noqa: PLC0415

    monkeypatch.setattr(langfuse_mod, "Langfuse", _RecordingLangfuse)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("GIT_SHA", raising=False)

    # Force the subprocess fallback to return None by pointing PATH at /tmp.
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("no git")),
    )

    cfg = AgentConfig()
    langfuse_client.get(cfg)

    assert "release" not in captured
