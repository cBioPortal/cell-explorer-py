"""HTTP app: the open address, plus the signed-in address when an MCP key is set."""

import os

from fastmcp.server.auth import RemoteAuthProvider
from fastmcp.utilities.lifespan import combine_lifespans
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from cell_explorer_mcp import queries, server
from cell_explorer_mcp.auth import CellExplorerTokenVerifier
from cell_explorer_mcp.catalog import DEFAULT_API_URL

DEFAULT_HTTP_PATH = "/mcp"
DEFAULT_SIGNED_IN_HTTP_PATH = "/signed-in/mcp"


def build_http_app() -> Starlette:
    """Serve the open server at MCP_HTTP_PATH and, with CELL_EXPLORER_MCP_KEY
    set, the signed-in server at MCP_SIGNED_IN_HTTP_PATH."""
    open_app = server.mcp.http_app(path=os.getenv("MCP_HTTP_PATH", DEFAULT_HTTP_PATH))
    mcp_key = os.getenv("CELL_EXPLORER_MCP_KEY")
    if not mcp_key:
        return open_app

    signed_in_path = os.getenv("MCP_SIGNED_IN_HTTP_PATH", DEFAULT_SIGNED_IN_HTTP_PATH)
    public_url = os.getenv("MCP_PUBLIC_URL", "").rstrip("/")
    if not public_url:
        raise RuntimeError("MCP_PUBLIC_URL must be set when CELL_EXPLORER_MCP_KEY is set")
    if not public_url.endswith(signed_in_path):
        raise RuntimeError(f"MCP_PUBLIC_URL must end with MCP_SIGNED_IN_HTTP_PATH ({signed_in_path})")

    api_url = os.getenv("CELL_EXPLORER_API_URL", DEFAULT_API_URL)
    verifier = CellExplorerTokenVerifier(
        exchange_url=api_url.rstrip("/") + "/oauth/token-exchange",
        mcp_key=mcp_key,
    )
    auth = RemoteAuthProvider(
        token_verifier=verifier,
        # Public URL: it must equal the API's PUBLIC_URL + /api/oauth.
        authorization_servers=[f"{queries.UI_URL}/api/oauth"],
        # FastMCP appends the path, so the advertised resource is MCP_PUBLIC_URL.
        base_url=public_url[: -len(signed_in_path)],
    )
    signed_in_app = server.build_server(signed_in=True, auth=auth).http_app(path=signed_in_path)

    # Each app keeps its own middleware: the signed-in app's paths (the MCP
    # endpoint and its protected-resource metadata) go to it whole; everything
    # else goes to the open app.
    routes = [Route(route.path, endpoint=signed_in_app) for route in signed_in_app.routes]
    return Starlette(
        routes=[*routes, Mount("/", app=open_app)],
        lifespan=combine_lifespans(open_app.lifespan, signed_in_app.lifespan),
    )
