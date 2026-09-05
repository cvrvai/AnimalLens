"""
FastAPI application entrypoint for AnimalLens server.
"""
from __future__ import annotations

import os
from pathlib import Path
import secrets

from dotenv import load_dotenv

# Ensure .env is loaded when starting the FastAPI app
load_dotenv()
_env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_file.exists():
    load_dotenv(dotenv_path=_env_file)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from animallens.core.config import settings
from animallens.server.routes import router


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    API Key Authentication Middleware for AnimalLens.
    Protects endpoints when exposed over Cloudflare Tunnel or public internet.
    If ANIMALLENS_API_KEY is not set in .env, authentication is skipped (open local mode).
    """
    async def dispatch(self, request: Request, call_next):
        expected_key = os.getenv("ANIMALLENS_API_KEY", "").strip()

        # If no key configured in environment, allow open access (local development)
        if not expected_key:
            return await call_next(request)

        # Public routes that bypass API key:
        # - Health checks (used by Cloudflare Tunnel, Next.js auto-discovery, load balancers)
        # - API Docs & schemas
        # - Static media files
        # - CORS preflight requests
        path = request.url.path
        clean_path = path.rstrip("/") if path != "/" else path
        public_paths = {
            "/v1/health",
            "/docs",
            "/redoc",
            "/v1/openapi.json",
            "/openapi.json",
        }
        if (
            path in public_paths
            or clean_path in public_paths
            or path.startswith("/static/")
            or request.method == "OPTIONS"
        ):
            return await call_next(request)

        # Check X-API-Key header, Authorization: Bearer, or query parameter
        api_key = request.headers.get("X-API-Key")
        if api_key:
            api_key = api_key.strip()
        else:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:].strip()
        if not api_key:
            api_key = request.query_params.get("api_key")
            if api_key:
                api_key = api_key.strip()

        if not api_key or not secrets.compare_digest(api_key, expected_key):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "detail": "Invalid or missing AnimalLens API Key. Provide via 'X-API-Key' header or 'Authorization: Bearer <key>'.",
                },
            )

        return await call_next(request)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="AnimalLens API",
        description="Open Animal Behavior Intelligence Platform - Vision AI, Event Engine, and Multi-Species Taxonomy",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/v1/openapi.json",
    )

    # Enable CORS for cross-platform clients, Next.js web dashboards, and ERPs
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Key Authentication (active when ANIMALLENS_API_KEY is configured in .env)
    app.add_middleware(APIKeyAuthMiddleware)

    app.include_router(router)

    # Ensure static directories exist and mount them for frontend media access
    for dir_name in ["datasets", "models", "data"]:
        p = Path(dir_name)
        p.mkdir(parents=True, exist_ok=True)
        app.mount(f"/static/{dir_name}", StaticFiles(directory=str(p)), name=f"{dir_name}_static")

    return app


app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8088, reload: bool = False) -> None:
    """Run server with uvicorn."""
    import uvicorn
    uvicorn.run("animallens.server.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
