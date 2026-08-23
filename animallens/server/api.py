"""
FastAPI application module alias for AnimalLens server.
"""
from animallens.server.app import app, create_app, run_server

__all__ = ["app", "create_app", "run_server"]
