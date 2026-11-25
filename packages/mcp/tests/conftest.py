"""Fixtures: zarr-access's test stores served over HTTP, plus a stub catalog."""

import socket
from pathlib import Path

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from aiohttp import web

from cell_explorer_mcp.catalog import CatalogClient
from cell_explorer_mcp.zarr_cache import open_dataset

FIXTURES_DIR = Path(__file__).parents[2] / "zarr-access" / "tests" / "fixtures"


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _write_generated(path: Path) -> None:
    """120 cells x 3 genes: a per-cell string column, a 60-value categorical,
    and two var IDs sharing the symbol HERC3."""
    n = 120
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "barcode": [f"cell-{i:04d}" for i in range(n)],
            "cluster": pd.Categorical([f"c{i // 2:02d}" for i in range(n)]),
            "cell_type": pd.Categorical(["T"] * 60 + ["B"] * 60),
        },
        index=[str(i) for i in range(n)],
    )
    var = pd.DataFrame(
        {"feature_name": ["HERC3", "HERC3", "CD3E"]},
        index=["ENSG00000287542", "ENSG00000138641", "ENSG00000198851"],
    )
    X = rng.random((n, 3), dtype="float32")
    ad.AnnData(X=X, obs=obs, var=var).write_zarr(path)


@pytest.fixture(scope="session")
def generated_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("generated")
    _write_generated(root / "generated.zarr")
    return root


# A private store: the fixtures again under /private/, served only with the
# credential the stub API's /access hands out.
PRIVATE_STORE_TOKEN = "zarr-ok"
PRIVATE_STORE_COOKIES = {
    "CloudFront-Policy": "policy",
    "CloudFront-Signature": "ok",
    "CloudFront-Key-Pair-Id": "kp",
}


async def _private_store(request: web.Request) -> web.StreamResponse:
    authorized = request.headers.get("Authorization") == f"Bearer {PRIVATE_STORE_TOKEN}"
    cookie = "; ".join(f"{k}={v}" for k, v in PRIVATE_STORE_COOKIES.items())
    if not authorized and request.headers.get("Cookie") != cookie:
        raise web.HTTPForbidden()
    root = FIXTURES_DIR.resolve()
    path = (root / request.match_info["tail"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def fixture_server(generated_dir):
    app = web.Application()
    app.router.add_get("/private/{tail:.*}", _private_store)
    app.router.add_static("/generated/", generated_dir)
    app.router.add_static("/", FIXTURES_DIR)
    port = _find_free_port()
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    yield f"http://127.0.0.1:{port}"
    await runner.cleanup()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def strata_tiny(fixture_server):
    """50 cells x 10 genes, dense X, coarse strata on cell_type only."""
    return await open_dataset("strata-tiny", f"{fixture_server}/strata-tiny.zarr")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pbmc(fixture_server):
    """2638 cells x 1838 genes, dense X, no strata."""
    return await open_dataset("pbmc3k", f"{fixture_server}/pbmc3k_v3.zarr")


@pytest_asyncio.fixture(loop_scope="session")
async def generated(fixture_server):
    """Opened fresh per test so its obs-column cache starts empty."""
    return await open_dataset("generated", f"{fixture_server}/generated/generated.zarr")


@pytest.fixture(scope="session")
def strata_tiny_adata():
    return ad.read_zarr(FIXTURES_DIR / "strata-tiny.zarr")


@pytest.fixture(scope="session")
def pbmc_adata():
    return ad.read_zarr(FIXTURES_DIR / "pbmc3k_v3.zarr")


# Shape of the API's DatasetCollectionRef (routes/datasets.py).
FIXTURES_COLLECTION = {
    "slug": "fixtures",
    "name": "Fixtures",
    "publication_citation": "Doe et al. 2026",
    "publication_url": None,
    "license_name": "CC BY 4.0",
    "license_url": None,
    "source_url": None,
    "attribution_note": None,
}


def catalog_payload(base_url: str) -> dict:
    return {
        "datasets": [
            {
                "slug": "strata-tiny",
                "name": "Strata tiny",
                "description": "Synthetic fixture",
                "is_public": True,
                "url": f"{base_url}/strata-tiny.zarr",
                "collection": FIXTURES_COLLECTION,
                "metadata": {"n_obs": 50, "n_vars": 10},
            },
            {
                "slug": "pbmc3k",
                "name": "PBMC 3k",
                "description": "Peripheral blood mononuclear cells",
                "is_public": True,
                "url": f"{base_url}/pbmc3k_v3.zarr",
                "collection": None,
                "metadata": {"n_obs": 2638, "n_vars": 1838},
            },
            {
                "slug": "secret",
                "name": "Private",
                "description": "",
                "is_public": False,
                "url": f"{base_url}/strata-tiny.zarr",
                "collection": None,
            },
        ],
        "collections": [{"slug": "fixtures", "name": "Fixtures", "dataset_count": 1}],
    }


# API tokens the stub API accepts: the private datasets each user sees in
# listings, and the ones /access lets them open (dave's grant was revoked
# after his listing was fetched).
API_USERS = {
    "api-alice": {"sees": {"secret", "secret-cookies"}, "opens": {"secret", "secret-cookies"}},
    "api-bob": {"sees": {"secret", "secret-cookies"}, "opens": {"secret", "secret-cookies"}},
    "api-carol": {"sees": set(), "opens": set()},
    "api-dave": {"sees": {"secret"}, "opens": set()},
}
PRIVATE_COLLECTION = {"slug": "private-studies", "name": "Private studies", "dataset_count": 2}


def private_datasets() -> list[dict]:
    """As the API lists them to a user allowed to see them: no URL."""
    return [
        {"slug": slug, "name": name, "description": "", "is_public": False, "url": None,
         "collection": None, "metadata": {"n_obs": 50, "n_vars": 10}}
        for slug, name in (("secret", "Private"), ("secret-cookies", "Private (cookies)"))
    ]


def cloudfront_set_cookies() -> list[tuple[str, str]]:
    return [
        ("set-cookie", f"{name}={value}; Domain=.test; HttpOnly; Max-Age=1800; "
                       "Path=/private/strata-tiny.zarr/; SameSite=lax; Secure")
        for name, value in PRIVATE_STORE_COOKIES.items()
    ]


def _access(base_url: str, payload: dict, slug: str, bearer: str | None, api: dict | None):
    user = API_USERS.get(bearer)
    if user is None:
        return httpx.Response(401)
    if api is not None and api.get("access_status"):
        return httpx.Response(api["access_status"])
    public = {d["slug"]: d for d in payload["datasets"] if d["is_public"]}
    if slug in public:
        return httpx.Response(200, json={"url": public[slug]["url"], "credential_type": "public"})
    if slug not in user["opens"]:
        return httpx.Response(404)
    url = f"{base_url}/private/strata-tiny.zarr"
    expires_at = "2026-11-25T12:30:00+00:00"
    if slug == "secret":
        return httpx.Response(200, json={
            "url": url, "credential_type": "bearer_token",
            "token": PRIVATE_STORE_TOKEN, "expires_at": expires_at,
        })
    # Signed cookies come back as Set-Cookie headers, not in the body.
    set_cookies = (api or {}).get("set_cookies") or cloudfront_set_cookies()
    return httpx.Response(200, headers=set_cookies, json={
        "url": url, "credential_type": "signed_cookies", "expires_at": expires_at,
    })


def stub_catalog(
    base_url: str,
    calls: list[str] | None = None,
    api: dict | None = None,
    log: list[tuple[str, str, str | None]] | None = None,
    **kwargs,
) -> CatalogClient:
    """api={"down": True} makes the stub answer 503 until it is set back;
    api={"access_status": 403} makes /access answer that status;
    api={"set_cookies": [...]} replaces its Set-Cookie headers. log records
    (method, path, bearer token) per request."""
    payload = catalog_payload(base_url)

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("Authorization")
        bearer = auth.removeprefix("Bearer ") if auth else None
        if calls is not None:
            calls.append(request.url.path)
        if log is not None:
            log.append((request.method, request.url.path, bearer))
        if api is not None and api.get("down"):
            return httpx.Response(503)
        parts = request.url.path.rsplit("/", 2)
        if request.method == "POST" and parts[-1] == "access":
            return _access(base_url, payload, parts[-2], bearer, api)
        key = parts[-1]
        if bearer is None:
            return httpx.Response(200, json={key: payload[key]})
        user = API_USERS.get(bearer)
        if user is None:
            return httpx.Response(401)
        if key == "datasets":
            public = [d for d in payload["datasets"] if d["is_public"]]
            private = [d for d in private_datasets() if d["slug"] in user["sees"]]
            return httpx.Response(200, json={"datasets": public + private})
        collections = payload["collections"] + ([PRIVATE_COLLECTION] if user["sees"] else [])
        return httpx.Response(200, json={"collections": collections})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return CatalogClient("http://catalog.test/api", client=client, **kwargs)
