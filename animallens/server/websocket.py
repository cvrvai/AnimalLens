"""
WebSocket Connection Manager for real-time behavior event streaming.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from typing import Any, Dict, List, Optional, Set
from fastapi import WebSocket, status

logger = logging.getLogger(__name__)


def extract_websocket_api_key(websocket: WebSocket) -> Optional[str]:
    """Extract API key from WebSocket headers or query parameters."""
    # 1. Check X-API-Key header
    api_key = websocket.headers.get("x-api-key")
    if api_key:
        return api_key.strip()

    # 2. Check Authorization: Bearer <key> header
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    # 3. Check query parameters (?api_key=<key> or ?token=<key>)
    query_key = websocket.query_params.get("api_key") or websocket.query_params.get("token")
    if query_key:
        return query_key.strip()

    return None


async def authenticate_websocket(websocket: WebSocket) -> bool:
    """
    Authenticate WebSocket client against ANIMALLENS_API_KEY.
    Returns True if authorized (or if open local mode is active).
    If unauthorized, closes WebSocket with WS_1008_POLICY_VIOLATION and returns False.
    """
    expected_key = os.getenv("ANIMALLENS_API_KEY", "").strip()
    if not expected_key:
        return True  # Open local mode

    client_key = extract_websocket_api_key(websocket)
    if not client_key or not secrets.compare_digest(client_key, expected_key):
        logger.warning("Unauthorized WebSocket connection attempt rejected (code 1008)")
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Unauthorized: Invalid or missing AnimalLens API Key",
        )
        return False

    return True



class ConnectionManager:
    """Manages active WebSocket client connections and event broadcasting."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        if hasattr(websocket, "accept"):
            await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast structured event to all active WebSocket listeners."""
        if not self.active_connections:
            return

        message = {
            "type": event_type,
            "data": data,
        }
        text = json.dumps(message, default=str)
        dead_sockets = []

        for connection in list(self.active_connections):
            try:
                await connection.send_text(text)
            except Exception:
                dead_sockets.append(connection)

        for dead in dead_sockets:
            self.disconnect(dead)


# Global WebSocket connection manager
ws_manager = ConnectionManager()
