"""
FastAPI application entrypoint for AnimalLens server.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from animallens.core.config import settings
from animallens.server.routes import router


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

    # Enable CORS for cross-platform clients, web dashboards, and ERPs
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8088, reload: bool = False) -> None:
    """Run server with uvicorn."""
    import uvicorn
    uvicorn.run("animallens.server.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
