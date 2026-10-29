"""PUBLIC_CORS_ORIGIN_REGEX: anonymous browser reads of the catalog from other sites."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

from cell_explorer_api.config import Settings
from cell_explorer_api.main import create_app
from cell_explorer_api.public_cors import PublicCatalogCorsMiddleware

REGEX = r"https://([a-z0-9-]+\.)*cbioportal\.org"
HUB = "https://hub.cbioportal.org"


def _client(cors_origins: list[str] | None = None) -> TestClient:
    app = FastAPI()

    @app.get("/api/datasets")
    @app.get("/api/collections")
    @app.get("/api/datasets/{slug}/access")
    def endpoint():
        return []

    if cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=cors_origins, allow_credentials=True
        )
    app.add_middleware(PublicCatalogCorsMiddleware, origin_regex=REGEX)
    return TestClient(app)


def test_catalog_listings_allow_matching_origins_without_credentials():
    client = _client()
    for path in ("/api/datasets", "/api/collections"):
        response = client.get(path, headers={"Origin": HUB})
        assert response.headers["access-control-allow-origin"] == HUB
        assert "access-control-allow-credentials" not in response.headers
        assert "Origin" in response.headers["vary"]


def test_other_origins_get_no_cors_headers():
    client = _client()
    for origin in (
        "https://evil.example.org",
        "https://cbioportal.org.evil.example",
        "http://hub.cbioportal.org",
    ):
        response = client.get("/api/datasets", headers={"Origin": origin})
        assert "access-control-allow-origin" not in response.headers


def test_other_endpoints_and_methods_get_no_cors_headers():
    client = _client()
    response = client.get("/api/datasets/x/access", headers={"Origin": HUB})
    assert "access-control-allow-origin" not in response.headers
    response = client.post("/api/datasets", headers={"Origin": HUB})
    assert "access-control-allow-origin" not in response.headers


def test_cors_origins_answer_wins_for_an_origin_in_both():
    client = _client(cors_origins=[HUB])
    response = client.get("/api/datasets", headers={"Origin": HUB})
    assert response.headers.get_list("access-control-allow-origin") == [HUB]
    assert response.headers["access-control-allow-credentials"] == "true"


def test_create_app_installs_middleware_only_when_set():
    def installed(settings: Settings) -> bool:
        app = create_app(settings)
        return any(m.cls is PublicCatalogCorsMiddleware for m in app.user_middleware)

    assert not installed(Settings())
    assert installed(Settings(public_cors_origin_regex=REGEX))
